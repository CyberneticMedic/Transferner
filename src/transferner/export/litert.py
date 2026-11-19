"""A6: export the BioCLIP image tower to .tflite with litert-torch and check it on x86.

Runs in the separate CPU-torch venv (CLAUDE.md: export toolchains pin torch versions that
clash with the ROCm wheels):

  uv venv --python 3.12 .venv-export
  uv pip install --python .venv-export/bin/python "torch<2.15" torchvision \
      --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python .venv-export/bin/python -c configs/export-constraints.txt \
      litert-torch open_clip_torch pandas pyarrow
  cd /tmp && PYTHONPATH=~/Transferner/src ~/Transferner/.venv-export/bin/python -m transferner.export.litert --recipe w8

Peak RSS is about 16 GB (the FP32 graph is serialized once, >2 GB, before quantization).

Graph: image (1x3x224x224, already preprocessed) -> L2-normalized 1024-d embedding. The
linear head (probe.pt) is a single matmul and stays outside the graph here.
Recipes (litert_torch generative quant recipes, applied by ai-edge-quantizer after conversion):
  fp32  no quantization (ViT-H/14 is ~2.5 GB of FP32 weights: tests the 2 GB flatbuffer limit)
  fp16  FP16 weights
  w8    INT8 weight-only, per channel (the deployable W8A16 of the efficiency sweep)
  w8dyn INT8 weights + dynamic INT8 activations (XNNPACK-style, the sweep's w8a8dyn)
  int8static  NPU-style full INT8 via the PT2E quantizer: per-channel INT8 weights, static
        per-tensor INT8 activations (HistogramObserver) on every linear, conv and add, calibrated
        on --calib train images; --smooth A folds SmoothQuant into the weights first (D-028)

Check: --images regional val photos (fixed hash order) through PyTorch FP32 and through the
LiteRT interpreter (XNNPACK, --threads). Reports mean/min cosine, top-1 agreement under the
production head, file size and x86 latency (a proxy only: phone latency, memory and NPU
compile need the device). Appends a row to results/tables/export_check.csv.

A memory guard exits the process if MemAvailable drops below --min-free-gb (shared machine)."""
from __future__ import annotations

import argparse
import os
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..utils.config import RESULTS, data_root, load

RECIPES = ("fp32", "fp16", "w8", "w8dyn", "int8static")


def mem_guard(min_free_gb: float) -> None:
    def loop():
        while True:
            with open("/proc/meminfo") as f:
                kb = next(int(line.split()[1]) for line in f if line.startswith("MemAvailable"))
            if kb / 2**20 < min_free_gb:
                print(f"memory guard: MemAvailable {kb / 2**20:.1f} GB < {min_free_gb} GB, exiting", flush=True)
                os._exit(3)
            time.sleep(1)
    threading.Thread(target=loop, daemon=True).start()


def peak_rss_gb() -> float:
    with open("/proc/self/status") as f:
        kb = next(int(line.split()[1]) for line in f if line.startswith("VmHWM"))
    return kb / 2**20


class Encoder(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, x):
        return F.normalize(self.visual(x), dim=-1)


def quant_config(recipe: str):
    if recipe in ("fp32",):
        return None
    from litert_torch.generative.quantize import quant_attrs, quant_recipes
    if recipe == "fp16":
        return quant_recipes.full_fp16_recipe()
    if recipe == "w8":
        return quant_recipes.full_weight_only_recipe(weight_dtype=quant_attrs.Dtype.INT8)
    if recipe == "w8dyn":
        return quant_recipes.full_dynamic_recipe(weight_dtype=quant_attrs.Dtype.INT8)
    raise ValueError(recipe)


def check_images(root: Path, pre, n: int, split: str = "val") -> tuple[torch.Tensor, np.ndarray]:
    from ..eval.efficiency import hsort
    from ..train.embed_images import Images
    va = pd.read_parquet(root / f"artifacts/img_emb/{split}.parquet")
    if split == "val":
        va = va[va.source == "gbif_regional"]
    va = va.iloc[hsort(va.uuid)[:n]].merge(
        pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")
    ds = Images(root, list(va.path), pre)
    return torch.stack([ds[i] for i in range(len(ds))]), va.class_idx.to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=RECIPES, default="w8")
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="artifacts/export", help="relative to TRANSFERNER_DATA")
    ap.add_argument("--min-free-gb", type=float, default=16.0)
    ap.add_argument("--skip-convert", action="store_true", help="re-check an existing .tflite")
    ap.add_argument("--calib", type=int, default=32, help="int8static: calibration train images")
    ap.add_argument("--smooth", type=float, default=0.0, help="int8static: SmoothQuant alpha, 0 = off")
    ap.add_argument("--linear-only", action="store_true",
                    help="int8static: quantize only linears (the D-028 simulation's scope), not adds/muls/cat")
    args = ap.parse_args()
    mem_guard(args.min_free_gb)
    torch.set_num_threads(args.threads)
    root = data_root()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.recipe + (f"_sq{args.smooth}" if args.smooth else "") + ("_linear" if args.linear_only else "")
    path = out_dir / f"bioclip_visual_{tag}.tflite"
    row = {"recipe": tag, "torch": torch.__version__, "images": args.images, "threads": args.threads}

    import litert_torch
    import open_clip
    row["litert_torch"] = getattr(litert_torch, "__version__", "")
    hub = load("bioclip")["model"]["hf_hub"]
    model, _, pre = open_clip.create_model_and_transforms(hub)
    enc = Encoder(model.visual).eval()
    del model.transformer, model.token_embedding                  # text tower: not exported
    x, y = check_images(root, pre, args.images)
    with torch.no_grad():
        ref = torch.cat([enc(x[i:i + 16]) for i in range(0, len(x), 16)]).numpy()

    t0 = time.time()
    if not args.skip_convert:
        try:
            if args.recipe == "int8static":
                m, diag = convert_int8_static(litert_torch, enc, x[:1], check_images(root, pre, args.calib, "train")[0],
                                              args.smooth, args.linear_only, x[:16])
                row.update(calib=args.calib, **diag)
            else:
                m = litert_torch.convert(enc, (x[:1],), quant_config=quant_config(args.recipe))
            m.export(str(path))
        except Exception as e:                                        # noqa: BLE001 - the error IS the result
            row.update(status="convert_failed", error=" ".join(repr(e).split())[:400],
                       convert_s=round(time.time() - t0), peak_rss_gb=round(peak_rss_gb(), 1))
            traceback.print_exc()
            write(row)
            return
    row["convert_s"] = round(time.time() - t0)
    row["file_mb"] = round(path.stat().st_size / 2**20, 1)

    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=args.threads)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    got, times = [], []
    for i in range(len(x)):
        it.set_tensor(inp["index"], x[i:i + 1].numpy().astype(inp["dtype"]))
        t = time.perf_counter()
        it.invoke()
        times.append(time.perf_counter() - t)
        got.append(it.get_tensor(outp["index"]).reshape(-1).astype(np.float32))
    got = np.stack(got)
    got /= np.linalg.norm(got, axis=1, keepdims=True)
    cos = (got * ref).sum(1)
    pr = torch.load(root / "artifacts/probe.pt", map_location="cpu")
    W, b, s = pr["W"].float().numpy(), pr["b"].float().numpy(), float(pr["scale"])
    p_ref, p_got = (s * ref @ W.T + b).argmax(1), (s * got @ W.T + b).argmax(1)
    row.update(status="ok", cos_mean=float(cos.mean()), cos_min=float(cos.min()),
               top1_agree=float((p_ref == p_got).mean()), top1_ref=float((p_ref == y).mean()),
               top1_tflite=float((p_got == y).mean()),
               x86_ms_median=round(1000 * float(np.median(times[1:])), 1),
               peak_rss_gb=round(peak_rss_gb(), 1))
    write(row)


def convert_int8_static(litert_torch, enc: Encoder, sample: torch.Tensor, calib: torch.Tensor, smooth: float,
                        linear_only: bool, probe_x: torch.Tensor):
    """PT2E static INT8 -> LiteRT. Also returns the cosine of the PT2E-quantized PyTorch graph against
    the float encoder on probe_x, so quantization damage and conversion bugs can be told apart."""
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    if smooth:                       # exact in float: folds into LayerNorms / V rows of enc.visual in place
        from ..models.efficient import EfficientVisual, Variant, calibrate
        ev = EfficientVisual(enc.visual, Variant(w_bits=8, a_bits=8, a_mode="static", smooth=smooth)).eval()
        with torch.no_grad():
            calibrate(ev, [calib[i:i + 8] for i in range(0, len(calib), 8)], "calib_channel")
            ev.smooth(smooth)
        del ev
    with torch.no_grad():
        ref = torch.cat([enc(probe_x[i:i + 1]) for i in range(len(probe_x))])

    class LinearOnly(PT2EQuantizer):
        STATIC_OPS = ["linear", "addmm"]

    quantizer = (LinearOnly() if linear_only else PT2EQuantizer()).set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False))
    m = torch.export.export(enc, (sample,)).module()
    m = prepare_pt2e(m, quantizer)
    with torch.no_grad():
        for i in range(len(calib)):                # exported graph is specialised to batch 1
            m(calib[i:i + 1])
    m = convert_pt2e(m, fold_quantize=False)
    with torch.no_grad():
        q = F.normalize(torch.cat([m(probe_x[i:i + 1]) for i in range(len(probe_x))]), dim=-1)
    diag = {"pt2e_cos_mean": float((q * ref).sum(-1).mean())}
    print("PT2E (PyTorch side) cosine vs float:", diag, flush=True)
    return litert_torch.convert(m, (sample,), quant_config=QuantConfig(pt2e_quantizer=quantizer)), diag


def write(row: dict) -> None:
    print(row, flush=True)
    f = RESULTS / "tables/export_check.csv"
    df = pd.DataFrame([{"time": time.strftime("%Y-%m-%d %H:%M"), **row}])
    if f.exists():
        df = pd.concat([pd.read_csv(f), df], ignore_index=True)
    df.to_csv(f, index=False)


if __name__ == "__main__":
    main()

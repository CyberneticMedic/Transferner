"""Phase D: export the live detector (SSDLite on MNv4-Conv-Medium, D-012) to LiteRT and check it on x86.

Runs in the CPU-torch venv, like export/litert.py:
  cd /tmp && PYTHONPATH=~/Transferner/src ~/Transferner/.venv-export/bin/python \\
      -m transferner.export.detector_litert --name mnv4_ssdlite --recipe int8

Graph: normalised image (1, 3, 320, 320) -> cls_logits (1, A, G + 1), bbox_regression (1, A, 4). Softmax,
anchor decoding, top-k and NMS stay outside the graph (app side, as in train_detector.detect). Depthwise
convs are plain nn.Conv2d here (the trainer's shift-add class is a ROCm training speed-up only).
Recipes:
  fp32  float reference
  int8  full INT8 for the NPU (D-011): PT2E static, per-channel symmetric INT8 weights, per-tensor INT8
        activations on every conv and add (BatchNorm folded), calibrated on --calib train images
        (eval transform); --act-observer histogram|minmax.
--pt2e-only scores the PT2E-quantized PyTorch graph and skips the LiteRT conversion.

Check: --images images of --split through float PyTorch and through the .tflite (or PT2E graph), both scored
by train_detector.evaluate_with (genus mAP/AP50, class-agnostic AP50, image genus/family top-1), plus the
cosine of the class logits. x86 latency is a proxy only (phone latency and NPU compile need the device).
Appends a row to results/tables/export_check.csv (recipe detector_<name>_<tag>)."""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..train.train_detector import (BoxSet, Net, anchors_for, build, collate, evaluate_with, load_items)
from ..utils.config import data_root, load
from .litert import mem_guard, peak_rss_gb, static_quantizer, write


class Export(torch.nn.Module):
    def __init__(self, ssd):
        super().__init__()
        self.net = Net(ssd)

    def forward(self, x):
        o = self.net(x)
        return o["cls_logits"], o["bbox_regression"]


def per_image(fn):
    """Wrap a batch-1 callable (exported graphs are specialised to batch 1) as run(x) for evaluate_with."""
    def run(x):
        outs = [fn(x[i:i + 1]) for i in range(len(x))]
        return {"cls_logits": torch.cat([o[0] for o in outs]), "bbox_regression": torch.cat([o[1] for o in outs])}
    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_ssdlite", help="artifacts/detector/<name>/")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--boxes", default="manifests/boxes.parquet")
    ap.add_argument("--recipe", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--act-observer", choices=["histogram", "minmax"], default="histogram")
    ap.add_argument("--pt2e-only", action="store_true")
    ap.add_argument("--images", type=int, default=512)
    ap.add_argument("--split", default="val")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.pt2e_only and args.recipe != "int8":
        ap.error("--pt2e-only needs --recipe int8")
    mem_guard(args.min_free_gb)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    root, cfg = data_root(), load("detector")
    size = cfg["model"]["input"]
    d = root / "artifacts/detector" / args.name
    labels = json.loads((d / "labels.json").read_text())
    ck = torch.load(d / args.ckpt, map_location="cpu")
    model = build(len(labels["genera"]), cfg["model"]["backbone"], size, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    genera, family_of, _, splits = load_items(root, args.boxes, ck["args"].get("fallback", "keep"), 0, args.seed)
    assert genera == labels["genera"], "boxes.parquet genera differ from the checkpoint's labels.json"
    rng = np.random.default_rng(args.seed)
    pick = lambda items, n: [items[i] for i in sorted(rng.permutation(len(items))[:n])]
    check = pick(splits[args.split], args.images)
    calib = pick(splits["train"], args.calib)
    mean = torch.tensor(model.backbone.mean).view(1, 3, 1, 1)
    std = torch.tensor(model.backbone.std).view(1, 3, 1, 1)
    load_x = lambda items: torch.cat([((imgs.float() / 255) - mean) / std for imgs, _, _ in
                                      DataLoader(BoxSet(root, items, size, False), batch_size=32, num_workers=4,
                                                 collate_fn=collate)])
    dl = DataLoader(BoxSet(root, check, size, False), batch_size=16, num_workers=4, collate_fn=collate)
    anchors0 = anchors_for(model, size, "cpu")
    score = lambda run: evaluate_with(run, model.box_coder, dl, anchors0, size, family_of, "cpu", mean, std)

    exp = Export(model).eval()
    tag = args.recipe + ("_minmax" if args.act_observer == "minmax" and args.recipe == "int8" else "") + \
        ("_pt2e" if args.pt2e_only else "")
    row = {"recipe": f"detector_{args.name}_{tag}", "torch": torch.__version__, "images": args.images,
           "split": args.split, "threads": args.threads, "epoch": ck["epoch"]}
    t0 = time.time()
    with torch.no_grad():
        ref_logits = []

        def float_run(x):
            o = exp(x)
            ref_logits.append(o[0])
            return {"cls_logits": o[0], "bbox_regression": o[1]}
        ref = score(float_run)
    ref_logits = torch.cat(ref_logits)
    row.update({f"float_{k}": round(v, 4) for k, v in ref.items() if k != "n_images"})
    print("float:", ref, f"({time.time() - t0:.0f} s)", flush=True)

    sample = load_x(check[:1])
    quantizer, q_logits = None, []
    if args.recipe == "int8":
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        quantizer = static_quantizer(False, args.act_observer)
        m = prepare_pt2e(torch.export.export(exp, (sample,)).module(), quantizer)
        cx = load_x(calib)
        t = time.time()
        with torch.no_grad():
            for i in range(len(cx)):
                m(cx[i:i + 1])
        m = convert_pt2e(m, fold_quantize=False)
        row.update(calib=args.calib, act_observer=args.act_observer, calib_s=round(time.time() - t))
        if args.pt2e_only:
            def q_run(x):
                with torch.no_grad():
                    o = m(x)
                q_logits.append(o[0])
                return o
            got = score(per_image(q_run))
            row.update(status="pt2e_only", peak_rss_gb=round(peak_rss_gb(), 1))
            finish(row, ref, got, ref_logits, torch.cat(q_logits))
            return
    else:
        m = exp

    import litert_torch
    from litert_torch.quantize.quant_config import QuantConfig
    path = d / f"detector_{tag}.tflite"
    t = time.time()
    try:
        lm = litert_torch.convert(m, (sample,), quant_config=QuantConfig(pt2e_quantizer=quantizer) if quantizer else None)
        lm.export(str(path))
    except Exception as e:                                                 # noqa: BLE001 - the error IS the result
        row.update(status="convert_failed", error=" ".join(repr(e).split())[:400], peak_rss_gb=round(peak_rss_gb(), 1))
        traceback.print_exc()
        write(row)
        return
    row.update(convert_s=round(time.time() - t), file_mb=round(path.stat().st_size / 2**20, 2),
               litert_torch=getattr(litert_torch, "__version__", ""))

    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=args.threads)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    outs = sorted(it.get_output_details(), key=lambda o: -int(np.prod(o["shape"])))   # logits first (G+1 > 4)
    times = []

    def tfl(x):
        x = x.numpy()
        s, z = inp["quantization"]
        if np.issubdtype(inp["dtype"], np.integer):                       # full-INT8 graph: int8 in, int8 out
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / s) + z, info.min, info.max)
        it.set_tensor(inp["index"], x.astype(inp["dtype"]))
        t1 = time.perf_counter()
        it.invoke()
        times.append(time.perf_counter() - t1)
        o = []
        for od in outs:
            v = it.get_tensor(od["index"]).astype(np.float32)
            s, z = od["quantization"]
            if np.issubdtype(od["dtype"], np.integer):
                v = (v - z) * s
            o.append(torch.from_numpy(v))
        q_logits.append(o[0])
        return o
    got = score(per_image(tfl))
    row.update(status="ok", x86_ms_median=round(1000 * float(np.median(times[1:])), 1),
               peak_rss_gb=round(peak_rss_gb(), 1))
    finish(row, ref, got, ref_logits, torch.cat(q_logits))


def finish(row, ref, got, ref_logits, q_logits) -> None:
    cos = torch.nn.functional.cosine_similarity(ref_logits.flatten(1), q_logits.flatten(1), dim=1)
    row.update(logit_cos_mean=round(float(cos.mean()), 5), logit_cos_min=round(float(cos.min()), 5),
               **{f"q_{k}": round(v, 4) for k, v in got.items() if k != "n_images"},
               **{f"delta_{k}": round(got[k] - ref[k], 4) for k in ("ap50", "agnostic_ap50", "genus_top1", "family_top1")})
    print("quantized:", got, flush=True)
    write(row)


if __name__ == "__main__":
    main()

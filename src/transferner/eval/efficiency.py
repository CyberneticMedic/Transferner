"""Evaluate one efficient BioCLIP variant (models/efficient.py): accuracy + cost, one CSV row.

Embeds val_regional, val_ext and test with the variant (FP16 autocast) into
artifacts/efficiency/<name>/, applies the production head (artifacts/probe.pt + T), and
appends a row to results/tables/efficiency.csv:
  accuracy  top-1/top-5 per split, test macro recall, rare-species (n_train<100) top-1, ECE,
            agreement / McNemar / mean cosine vs the `fp16` reference variant
  cost      weight MB (deployable), GFLOPs per image, GPU img/s (float variants only)

Variants that change the feature space (--drop > 0, or --retrain) get a new head: a
stratified train subset (--train-subset per species) plus train_ext is embedded too and
linear_probe + calibrate run on the variant's embedding dir. `base_subset` (no knobs,
--retrain) is the matching reference for those.

--val-only skips test (used by the partial-INT8 search to pick k on val)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from ..models import bioclip
from ..models.efficient import Variant, build, preprocess_for, weight_bytes
from ..train.embed_images import Images
from ..utils.config import RESULTS, data_root
from ..utils.csvlog import RunLog
from . import metrics as M

REF = "fp16"


def prod_head_dir(root):
    """Head for every production-head variant: a frozen copy of artifacts/probe.pt + temperature.txt
    taken before the sweep's first job (artifacts/efficiency/_head/), so adopting a new production
    probe mid-sweep does not make the rows incomparable. Falls back to artifacts/."""
    d = root / "artifacts/efficiency/_head"
    return d if (d / "probe.pt").exists() else root / "artifacts"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--tome-r", type=int, default=0)
    ap.add_argument("--drop", type=int, default=0)
    ap.add_argument("--w-bits", type=int, default=16)
    ap.add_argument("--w-group", type=int, default=0)
    ap.add_argument("--a-bits", type=int, default=16)
    ap.add_argument("--a-mode", choices=["dynamic", "static"], default="dynamic")
    ap.add_argument("--a-pct", type=float, default=99.99)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--fp16-from", default="", help="sensitivity CSV in results/tables, most damaging first")
    ap.add_argument("--fp16-k", type=int, default=0, help="keep the k most sensitive layers in FP16")
    ap.add_argument("--retrain", action="store_true", help="train a new head on this variant's features")
    ap.add_argument("--train-subset", type=int, default=40, help="train images per species when retraining")
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--calib-images", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N images per split")
    ap.add_argument("--note", default="")
    ap.add_argument("--ref", default=REF, help="reference variant for agreement / McNemar")
    return ap


def variant_from(args) -> Variant:
    fp16 = frozenset()
    if args.fp16_from and args.fp16_k:
        fp16 = frozenset(pd.read_csv(RESULTS / "tables" / args.fp16_from).layer.head(args.fp16_k))
    return Variant(res=args.res, tome_r=args.tome_r, drop=args.drop, w_bits=args.w_bits, w_group=args.w_group,
                   a_bits=args.a_bits, a_mode=args.a_mode, a_pct=args.a_pct, smooth=args.smooth, fp16_layers=fp16)


def hsort(uuids: pd.Series) -> np.ndarray:
    return np.argsort([hashlib.sha1(u.encode()).hexdigest() for u in uuids])


def split_rows(root, limit: int, train_subset: int, retrain: bool, val_only: bool) -> dict[str, pd.DataFrame]:
    emb = root / "artifacts/img_emb"
    paths = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    val = pd.read_parquet(emb / "val.parquet")
    sets = {"val": val[val.source == "gbif_regional"], "val_ext": pd.read_parquet(emb / "val_ext.parquet")}
    if not val_only or retrain:                        # linear_probe needs test even when selecting on val
        sets["test"] = pd.read_parquet(emb / "test.parquet")
    if retrain:
        tr = pd.read_parquet(emb / "train.parquet")
        tr = tr.iloc[hsort(tr.uuid)]
        sets["train"] = tr[tr.groupby("class_idx").cumcount() < train_subset]
        sets["train_ext"] = pd.read_parquet(emb / "train_ext.parquet")
    out = {}
    for k, d in sets.items():
        d = d.merge(paths, on="uuid").reset_index(drop=True)
        if limit:
            d = d.iloc[hsort(d.uuid)[:limit]].reset_index(drop=True)
        out[k] = d
    return out


def loader(root, paths, pre, batch, workers):
    return DataLoader(Images(root, list(paths), pre), batch_size=batch, num_workers=workers,
                      pin_memory=True, prefetch_factor=4)


@torch.no_grad()
def embed(ev, dl) -> np.ndarray:
    out = []
    for x in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            e = ev(x.to("cuda", non_blocking=True))
        out.append(F.normalize(e.float(), dim=-1).half().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def cost(ev, res: int, quantized: bool) -> dict:
    x = torch.randn(1, 3, res, res, device="cuda")
    with FlopCounterMode(display=False) as fc, torch.autocast("cuda", dtype=torch.float16):
        ev(x)
    out = {"gflops": fc.get_total_flops() / 1e9}               # FlopCounterMode counts 2 FLOPs per MAC
    if not quantized:                                          # fake-quant overhead would distort speed
        xb = torch.randn(128, 3, res, res, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            for _ in range(3):
                ev(xb)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                ev(xb)
            torch.cuda.synchronize()
        out["gpu_img_s"] = 1280 / (time.perf_counter() - t)
    return out


def head_logits(root, head_dir):
    pr = torch.load(head_dir / "probe.pt")
    T = float((head_dir / "temperature.txt").read_text())
    W, b, s = pr["W"].cuda(), pr["b"].cuda(), pr["scale"]
    return (lambda e: s * e @ W.T + b), T


@torch.no_grad()
def predict(fn, e: np.ndarray, T: float):
    tops, confs = [], []
    et = torch.from_numpy(e)
    for i in range(0, len(et), 32768):
        p = (fn(et[i:i + 32768].cuda().float()) / T).softmax(-1)
        tops.append(p.topk(5, dim=-1).indices.cpu())
        confs.append(p.max(-1).values.cpu())
    return torch.cat(tops).numpy(), torch.cat(confs).numpy()


def run(args) -> dict:
    root = data_root()
    v = variant_from(args)
    out_dir = root / "artifacts/efficiency" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model, pre, _ = bioclip.load()
    pre = preprocess_for(pre, v.res)
    rows = split_rows(root, args.limit, args.train_subset, args.retrain or v.drop > 0, args.val_only)

    calib = None
    if v.a_bits < 16 and v.a_mode == "static":
        tr = pd.read_parquet(root / "artifacts/img_emb/train.parquet")
        tr = tr.iloc[hsort(tr.uuid)[: args.calib_images]].merge(
            pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")
        calib = [x.cuda() for x in loader(root, tr.path, pre, args.batch, args.workers)]
    ev = build(model.visual, v, calib)
    del calib

    for k, d in rows.items():
        e = embed(ev, loader(root, d.path, pre, args.batch, args.workers))
        np.save(out_dir / f"{k}.npy", e)
        d[["uuid", "class_idx", "is_specimen", "source"]].to_parquet(out_dir / f"{k}.parquet", index=False)
        print(f"[{args.name}] {k}: {len(d):,} embeddings ({time.time() - t0:.0f}s)", flush=True)

    c = cost(ev, v.res, v.quantized)
    c["weight_mb"] = weight_bytes(model.visual, v) / 2**20
    del ev, model
    torch.cuda.empty_cache()

    if args.retrain or v.drop > 0:                     # new feature space -> new head
        py = [sys.executable, "-m"]
        rel = f"artifacts/efficiency/{args.name}"
        subprocess.run(py + ["transferner.train.linear_probe", "--emb", rel, "--out", f"{rel}/probe.pt",
                             "--tag", f"eff_{args.name}"], check=True)
        if not args.val_only:
            subprocess.run(py + ["transferner.train.calibrate", "--emb", rel, "--probe", f"{rel}/probe.pt",
                                 "--out", f"{rel}/temperature.txt"], check=True)
        head_dir = out_dir
    else:
        head_dir = prod_head_dir(root)
        (out_dir / "temperature.txt").write_text((head_dir / "temperature.txt").read_text())
    fn, T = head_logits(root, head_dir)
    prod_fn, prod_T = head_logits(root, prod_head_dir(root))

    sp = pd.read_parquet(root / "manifests/species.parquet").set_index("class_idx")
    row = {"name": args.name, **{k: getattr(v, k) for k in ("res", "tome_r", "drop", "w_bits", "w_group",
                                                          "a_bits", "a_mode", "a_pct", "smooth")},
           "fp16_k": len(v.fp16_layers), "head": "retrained" if head_dir == out_dir else "production",
           "limit": args.limit, **{k: round(x, 3) for k, x in c.items()}}
    ref_dir = root / "artifacts/efficiency" / args.ref
    for k in ("val", "val_ext", "test"):
        if k not in rows:
            continue
        e = np.load(out_dir / f"{k}.npy")
        y = rows[k].class_idx.to_numpy()
        top, conf = predict(fn, e, T)
        hit = top[:, 0] == y
        tag = "val_regional" if k == "val" else k
        row[f"{tag}_top1"], row[f"{tag}_top5"] = hit.mean(), (top == y[:, None]).any(1).mean()
        if k == "test":
            row["test_macro_recall"] = M.per_class_f1(top[:, 0], y)[0]
            row["test_ece"] = M.ece(conf, hit)
            row["test_rare_top1"] = hit[(sp.loc[y].n_train < 100).to_numpy()].mean()
        ref_file = ref_dir / f"{k}.npy"
        if args.name != args.ref and ref_file.exists() and (ref_dir / f"{k}.parquet").exists():
            ref_uuid = pd.read_parquet(ref_dir / f"{k}.parquet").uuid
            if len(ref_uuid) == len(rows[k]) and (ref_uuid.to_numpy() == rows[k].uuid.to_numpy()).all():
                rtop, _ = predict(prod_fn, np.load(ref_file), prod_T)       # reference: production head
                re_ = np.load(ref_file)
                row[f"{tag}_agree_ref"] = (rtop[:, 0] == top[:, 0]).mean()
                if v.drop == 0:
                    row[f"{tag}_cos_ref"] = float((e.astype(np.float32) * re_.astype(np.float32)).sum(1).mean())
                if k == "test":
                    row["test_mcnemar_p_ref"] = M.mcnemar(rtop[:, 0] == y, hit)["p"]
    row["minutes"] = round((time.time() - t0) / 60, 1)
    row["note"] = args.note
    log = RunLog(f"eff_{args.name}")
    log.summary("efficiency" if not args.limit else "efficiency_smoke", **row)
    log.close()
    (out_dir / "result.json").write_text(json.dumps({k: (float(x) if isinstance(x, (np.floating, float)) else x)
                                                      for k, x in row.items()}, indent=1, default=str))
    print(json.dumps({k: (round(float(x), 4) if isinstance(x, (np.floating, float)) else x) for k, x in row.items()},
                     default=str), flush=True)
    return row


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()

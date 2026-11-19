"""Per-layer sensitivity to static W8A8 (D-011 "partial W8A8": sensitive layers stay FP16).

Quantizes ONE linear at a time (weights per-channel INT8 + calibrated per-tensor INT8 input),
everything else FP16, and measures the damage on a fixed batch of regional val photos:
mean cosine distance to the FP16 embeddings and top-1 agreement under the production head.
SmoothQuant (--smooth) is applied to all layers first; it is exact in float, so the FP16
reference is unchanged. Writes results/tables/quant_sensitivity[_tag].csv, most damaging first."""
from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from ..eval.efficiency import head_logits, hsort, loader, prod_head_dir
from ..models import bioclip
from ..models.efficient import EfficientVisual, Variant, calibrate, fq_weight
from ..utils.config import RESULTS, data_root


@torch.no_grad()
def embed(ev, batches):
    out = []
    for x in batches:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(F.normalize(ev(x).float(), dim=-1))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--a-pct", type=float, default=99.99)
    ap.add_argument("--images", type=int, default=1024)
    ap.add_argument("--calib-images", type=int, default=1024)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    root = data_root()
    model, pre, _ = bioclip.load()
    paths = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    tr = pd.read_parquet(root / "artifacts/img_emb/train.parquet")
    tr = tr.iloc[hsort(tr.uuid)[: args.calib_images]].merge(paths, on="uuid")
    va = pd.read_parquet(root / "artifacts/img_emb/val.parquet")
    va = va[va.source == "gbif_regional"]
    va = va.iloc[hsort(va.uuid)[: args.images]].merge(paths, on="uuid")
    calib = [x.cuda() for x in loader(root, tr.path, pre, 128, 10)]
    evalb = [x.cuda() for x in loader(root, va.path, pre, 128, 10)]
    y = torch.tensor(va.class_idx.to_numpy()).cuda()

    v = Variant(w_bits=8, a_bits=8, a_mode="static", a_pct=args.a_pct, smooth=args.smooth)
    ev = EfficientVisual(model.visual, v).eval()
    if args.smooth > 0:
        calibrate(ev, calib, "calib_channel")
        ev.smooth(args.smooth)
    calibrate(ev, calib, "calib_range")
    fn, _ = head_logits(root, prod_head_dir(root))
    ev.q.active = set()                                   # nothing quantized: FP16 reference
    ref = embed(ev, evalb)
    ref_pred = fn(ref).argmax(-1)
    print(f"FP16 reference top-1 on {len(y)} regional val photos: {(ref_pred == y).float().mean():.4f}", flush=True)

    rows = []
    for name in ev.names():
        w = ev.weight_of(name)
        orig = w.detach().clone()
        w.data.copy_(fq_weight(w, 8, 0))
        ev.q.active = {name}
        e = embed(ev, evalb)
        w.data.copy_(orig)
        pred = fn(e).argmax(-1)
        rows.append({"layer": name, "cos_dist": (1 - (e * ref).sum(-1)).mean().item(),
                     "agree": (pred == ref_pred).float().mean().item(), "top1": (pred == y).float().mean().item(),
                     "a_lo": ev.q.ranges[name][0], "a_hi": ev.q.ranges[name][1]})
        print(f"{name:8s} cos_dist {rows[-1]['cos_dist']:.5f} agree {rows[-1]['agree']:.4f}", flush=True)
    df = pd.DataFrame(rows).sort_values("cos_dist", ascending=False)
    out = RESULTS / "tables" / f"quant_sensitivity{('_' + args.tag) if args.tag else ''}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}; most sensitive:\n{df.head(12).to_string(index=False)}")


if __name__ == "__main__":
    main()

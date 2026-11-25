"""Live label tuning on cached whole-frame logits (app feedback 2026-11-23: "live accuracy needs improvement").

The detector's whole-frame genus head (the app's live label, D-033/D-034) was trained on at most 100 photos
per genus with a n^-0.5 sampler, so its implicit genus prior is close to uniform. Walking around the region,
the genera you meet are far from uniform. This script measures two post-hoc fixes on the REGIONAL photos
(selection on the regional slice of val, as D-024; test reported once per setting):
  prior   logits + tau * log p(genus), p from the regional GBIF observation counts (priors_state_month.parquet,
          summed over states), optionally per month (smoothed toward the annual prior);
  family  show the family instead of the genus when the top genus probability is below a threshold: the
          live label is then right more often, at a coarser level.
Image logits are computed once (GPU) and cached next to the checkpoint.
  python -m transferner.eval.detector_prior [--name mnv4_shared]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..train.train_detector import BoxSet, Net, build, collate, load_items
from ..utils.config import data_root, load


def logits_for(root, model, items, size, dev, mean, std) -> np.ndarray:
    net = Net(model).to(dev).eval()
    out = []
    dl = DataLoader(BoxSet(root, items, size, False), batch_size=128, num_workers=16, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, *_ in dl:
            x = ((imgs.to(dev).float() / 255) - mean) / std
            out.append(net(x)["image_logits"].float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_shared")
    ap.add_argument("--boxes", default="manifests/boxes.parquet")
    ap.add_argument("--ckpt", default="best.pt", help="cache: image_logits.npz for best.pt, else image_logits_<stem>.npz")
    args = ap.parse_args()
    root, cfg = data_root(), load("detector")
    d = root / "artifacts/detector" / args.name
    labels = json.loads((d / "labels.json").read_text())
    ck = torch.load(d / args.ckpt, map_location="cpu")
    size = ck["args"].get("size") or cfg["model"]["input"]
    genera, family_of, families, splits = load_items(root, args.boxes, ck["args"].get("fallback", "keep"), 0, 0)
    assert genera == labels["genera"]
    G = len(genera)
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path", "source", "event_date"]).set_index("path")

    cache = d / ("image_logits.npz" if args.ckpt == "best.pt" else f"image_logits_{Path(args.ckpt).stem}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        data = {s: (z[f"{s}_logits"], z[f"{s}_y"], z[f"{s}_path"]) for s in ("val", "test")}
    else:
        dev = "cuda"
        model = build(G, cfg["model"]["backbone"], size, pretrained=False, head="shared", image_aux=True,
                      image_head=ck["args"].get("image_head", 0), min_ratio=ck["args"].get("min_ratio", 0.2))
        model.load_state_dict(ck["model"])
        mean = torch.tensor(model.backbone.mean, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(model.backbone.std, device=dev).view(1, 3, 1, 1)
        data = {}
        for s in ("val", "test"):
            items = splits[s]
            data[s] = (logits_for(root, model, items, size, dev, mean, std),
                       np.array([int(lab[0]) - 1 for _, _, lab in items]), np.array([p for p, _, _ in items]))
            print(f"{s}: {len(items):,} images", flush=True)
        np.savez(cache, **{f"{s}_{k}": v for s, t in data.items() for k, v in zip(("logits", "y", "path"), t)})

    # regional genus prior, annual and per month (counts summed over the four states)
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["gbif_taxon_key", "genus"])
    pr = pd.read_parquet(root / "manifests/priors_state_month.parquet").merge(sp, on="gbif_taxon_key")
    gi = {g: i for i, g in enumerate(genera)}
    pr = pr[pr.genus.isin(gi)]
    cm = np.ones((13, G))                                                 # +1 smoothing; row 0 = annual
    for (g, m), c in pr.groupby(["genus", "month"])["count"].sum().items():
        cm[int(m), gi[g]] += c
    cm[0] = cm[1:].sum(0)
    annual = np.log(cm[0] / cm[0].sum())
    beta = 0.2 * cm[1:].sum(1).mean()                                     # month counts shrunk toward the annual prior
    monthly = {m: np.log((cm[m] + beta * cm[0] / cm[0].sum()) / (cm[m].sum() + beta)) for m in range(1, 13)}
    print(f"prior: {int(cm[0].sum() - G):,} regional observations over {G} genera; top genera "
          + ", ".join(genera[i] for i in np.argsort(-cm[0])[:5]), flush=True)

    fam_t = np.asarray(family_of)
    rows = []
    for s, (lg, y, paths) in data.items():
        meta = im.reindex(paths)
        regional = (meta.source == "gbif_regional").to_numpy()
        month = pd.to_datetime(meta.event_date, errors="coerce").dt.month.fillna(0).astype(int).to_numpy()
        for kind in ("none", "annual", "month"):
            for tau in ((0.0,) if kind == "none" else (0.25, 0.5, 0.75, 1.0)):
                if kind == "annual":
                    adj = lg + tau * annual
                elif kind == "month":
                    adj = lg + tau * np.stack([monthly[m] if m else annual for m in month])
                else:
                    adj = lg
                p = np.exp(adj - adj.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                top = np.argsort(-p, 1)[:, :5]
                fp = np.zeros((len(p), fam_t.max() + 1)); np.add.at(fp, (np.arange(len(p))[:, None], fam_t[None, :].repeat(len(p), 0)), p)
                for sub, m in (("regional", regional), ("tol", ~regional)):
                    if not m.any():
                        continue
                    r = {"split": s, "subset": sub, "n": int(m.sum()), "prior": kind, "tau": tau,
                         "genus_top1": float((top[m, 0] == y[m]).mean()), "genus_top5": float((top[m] == y[m, None]).any(1).mean()),
                         "family_top1": float((fp[m].argmax(1) == fam_t[y[m]]).mean())}
                    # hierarchical live label: genus when its probability >= t, else family
                    for t in (0.3, 0.5):
                        show_g = p[m].max(1) >= t
                        ok = np.where(show_g, top[m, 0] == y[m], fp[m].argmax(1) == fam_t[y[m]])
                        r[f"shown_correct_t{t}"] = float(ok.mean())
                        r[f"genus_shown_t{t}"] = float(show_g.mean())
                    rows.append(r)
    df = pd.DataFrame(rows)
    out = root / "artifacts/detector" / args.name / "prior_sweep.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df[(df.subset == "regional")].round(4).to_string(index=False))
    print(df[(df.subset == "tol") & (df.split == "val")][["prior", "tau", "genus_top1", "family_top1"]].round(4).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()

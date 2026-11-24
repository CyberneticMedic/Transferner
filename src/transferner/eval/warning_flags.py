"""Warning-flag rules, false positives preferred over false negatives (user, 2026-11-24; D-038).

Ground truth: a photo is "toxic" when its true species has a cited toxic row (data/traits/species_traits.csv);
"serious" when one of those rows has moderate/high severity. A rule MISSES a toxic photo when it raises no
flag, and FALSE-ALARMS on a non-toxic photo when it raises one. Regional photos only (val selects, test reports).

Capture (BioCLIP probe, cached embeddings) - a flag is a caution banner:
  top5          a cited toxic species in the top 5 (app 0.4)
  top5+rel      ... or a candidate without trait rows whose genus has a cited toxic species ("relative")
  p>=t (+rel)   any cited toxic (or relative) species with probability >= t anywhere in the 4,411
Live (detector whole-frame genus, cached logits) - a flag is the live warning row:
  live p>=t     a genus with probability >= t that contains a cited toxic species (any / serious only)
  python -m transferner.eval.warning_flags [--live mnv4_shared]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils.config import data_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="mnv4_shared", help="detector whose cached whole-frame logits to use")
    args = ap.parse_args()
    torch.set_num_threads(8)
    root = data_root()
    repo = Path(__file__).resolve().parents[3]
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx").reset_index(drop=True)
    tr = pd.read_csv(repo / "data/traits/species_traits.csv")
    tox_rows = tr[tr.category == "toxic"]
    toxic = sp.scientific_name.isin(set(tox_rows.scientific_name)).to_numpy()
    serious = sp.scientific_name.isin(set(tox_rows[tox_rows.ingestion_severity != "low"].scientific_name)
                                      | set(tox_rows[tox_rows.contact_toxic == 1].scientific_name)).to_numpy()
    has_rows = sp.scientific_name.isin(set(tr.scientific_name)).to_numpy()
    tox_genera = set(sp.genus[toxic])
    rel = (~has_rows) & sp.genus.isin(tox_genera).to_numpy()
    pr = torch.load(root / "artifacts/probe.pt", map_location="cpu")
    T = float((root / "artifacts/temperature.txt").read_text())
    rows = []

    def add(kind, split, rule, flag, y_toxic, y_serious):
        rows.append({"kind": kind, "split": split, "rule": rule, "n": len(flag),
                     "miss_toxic": round(float(np.mean(~flag[y_toxic])), 4),
                     "miss_serious": round(float(np.mean(~flag[y_serious])), 4),
                     "false_alarm": round(float(np.mean(flag[~y_toxic])), 4)})

    for split in ("val", "test"):
        m = pd.read_parquet(root / f"artifacts/img_emb/{split}.parquet")
        e = np.load(root / f"artifacts/img_emb/{split}.npy", mmap_mode="r")
        keep = np.where((m.source == "gbif_regional").to_numpy())[0]
        y = m.class_idx.to_numpy()[keep]
        p = np.concatenate([((pr["scale"] * torch.from_numpy(np.asarray(e[keep[s:s + 20000]], dtype=np.float32))
                              @ pr["W"].float().T + pr["b"].float()) / T).softmax(-1).numpy() for s in range(0, len(keep), 20000)])
        top5 = np.argsort(-p, 1)[:, :5]
        yt, ys = toxic[y], serious[y]
        add("capture", split, "top5", toxic[top5].any(1), yt, ys)
        add("capture", split, "top5+rel", (toxic | rel)[top5].any(1), yt, ys)
        for t in (0.02, 0.01, 0.005):
            add("capture", split, f"p>={t}", (p[:, toxic] >= t).any(1) | toxic[top5].any(1), yt, ys)
            add("capture", split, f"p>={t}+rel", (p[:, toxic | rel] >= t).any(1) | (toxic | rel)[top5].any(1), yt, ys)

    d = root / "artifacts/detector" / args.live
    genera = json.loads((d / "labels.json").read_text())["genera"]
    g_tox = np.array([g in tox_genera for g in genera])
    g_ser = np.array([g in set(sp.genus[serious]) for g in genera])
    z = np.load(d / "image_logits.npz", allow_pickle=True)
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path", "source", "class_idx"]).set_index("path")
    for split in ("val", "test"):
        lg, paths = z[f"{split}_logits"].astype(np.float64), z[f"{split}_path"]
        meta = im.reindex(paths)
        reg = (meta.source == "gbif_regional").to_numpy()
        lg, cls = lg[reg], meta.class_idx.to_numpy()[reg].astype(int)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        yt, ys = toxic[cls], serious[cls]
        for t in (0.2, 0.1, 0.05, 0.02):
            add("live", split, f"any toxic genus p>={t}", (p[:, g_tox] >= t).any(1), yt, ys)
            add("live", split, f"serious genus p>={t}", (p[:, g_ser] >= t).any(1), yt, ys)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    out = repo / ("results/tables/warning_flags.csv" if args.live == "mnv4_shared" else f"results/tables/warning_flags_{args.live}.csv")
    df.to_csv(out, index=False)
    print(f"relatives (no rows, genus has a cited toxic species): {rel.sum()} species; toxic genera {len(tox_genera)} "
          f"({g_tox.sum()} of the detector's {len(genera)}), serious {g_ser.sum()}; wrote {out}")


if __name__ == "__main__":
    main()

"""Phase B diagnosis: why bryophytes trail vascular plants (CPU only, cached embeddings).

  python -m transferner.eval.moss_diagnosis

Per phylum (Bryophyta, Marchantiophyta, Tracheophyta), on test and regional val, with the
production probe (artifacts/probe.pt) and the zero-shot text head:
  accuracy     top-1 probe / zero-shot
  error depth  share of probe errors whose prediction is the same genus / same family /
               same phylum / another phylum (moss called a vascular plant, ...)
  data         train images per species (median, share < 20 and < 100), specimen share of
               train, ToL images left in split `unused` by the per-species cap, train_ext
  by n_train   probe vs zero-shot top-1 per training-count bin (is it just fewer images?)
Writes results/tables/moss_diagnosis.csv and moss_diagnosis_bins.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..utils.config import RESULTS, data_root

PHYLA = ["Bryophyta", "Marchantiophyta", "Tracheophyta"]
BINS = [0, 20, 100, 400, 10**9]


@torch.no_grad()
def argmax(e: np.ndarray, W: torch.Tensor, b: torch.Tensor | float, s: float) -> np.ndarray:
    out = []
    for i in range(0, len(e), 16384):
        x = torch.from_numpy(np.asarray(e[i:i + 16384], np.float32))
        out.append((s * x @ W.T + b).argmax(-1).numpy())
    return np.concatenate(out)


def main() -> None:
    torch.set_num_threads(8)
    root = data_root()
    emb = root / "artifacts/img_emb"
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx").set_index("class_idx")
    pr = torch.load(root / "artifacts/probe.pt", map_location="cpu")
    text = torch.from_numpy(np.load(root / "artifacts/text_emb.npy")).float()
    zs_scale = float((root / "artifacts/logit_scale.txt").read_text())
    splits = pd.read_parquet(root / "manifests/splits.parquet", columns=["class_idx", "split", "is_specimen", "source"])

    full = sp.reindex(range(int(sp.index.max()) + 1))     # class_idx -> row, gaps stay NaN
    phylum, genus, family = (full[c].to_numpy() for c in ("phylum", "genus", "family"))

    rows, bins = [], []
    for split in ("test", "val"):
        meta = pd.read_parquet(emb / f"{split}.parquet")
        e = np.load(emb / f"{split}.npy", mmap_mode="r")
        if split == "val":                                # regional slice only (D-024)
            keep = (meta.source == "gbif_regional").to_numpy()
            meta, e = meta[keep].reset_index(drop=True), np.asarray(e)[keep]
        y = meta.class_idx.to_numpy()
        p = argmax(e, pr["W"].float(), pr["b"].float(), pr["scale"])
        z = argmax(e, text, 0.0, zs_scale)
        for ph in PHYLA:
            m = phylum[y] == ph
            if not m.any():
                continue
            yy, pp, zz = y[m], p[m], z[m]
            err = pp != yy
            same_g = genus[pp] == genus[yy]
            same_f = family[pp] == family[yy]
            same_p = phylum[pp] == phylum[yy]
            rows.append({"split": split, "phylum": ph, "n_img": int(m.sum()), "n_species": int(len(np.unique(yy))),
                         "probe_top1": (~err).mean(), "zero_shot_top1": (zz == yy).mean(),
                         "probe_genus_top1": same_g.mean(), "probe_family_top1": same_f.mean(),
                         "err_same_genus": (err & same_g).sum() / err.sum(),
                         "err_same_family_other_genus": (err & ~same_g & same_f).sum() / err.sum(),
                         "err_same_phylum_other_family": (err & ~same_f & same_p).sum() / err.sum(),
                         "err_other_phylum": (err & ~same_p).sum() / err.sum(),
                         "pred_into_phylum_from_others": int(((phylum[p] == ph) & (phylum[y] != ph)).sum())})
            ntr = sp.reindex(yy).n_train.to_numpy()
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                b = (ntr >= lo) & (ntr < hi)
                if b.any():
                    bins.append({"split": split, "phylum": ph, "n_train_bin": f"{lo}-{hi - 1}" if hi < 10**9 else f">={lo}",
                                 "n_img": int(b.sum()), "n_species": int(len(np.unique(yy[b]))),
                                 "probe_top1": (pp[b] == yy[b]).mean(), "zero_shot_top1": (zz[b] == yy[b]).mean()})
    diag = pd.DataFrame(rows)

    # training data per phylum (label set, all species)
    data = []
    for ph in PHYLA:
        s = sp[sp.phylum == ph]
        tr = splits[(splits.split == "train") & splits.class_idx.isin(s.index)]
        un = splits[(splits.split == "unused") & splits.class_idx.isin(s.index)]
        data.append({"phylum": ph, "species": len(s), "train_img": len(tr),
                     "train_median_per_species": float(s.n_train.median()),
                     "species_train_lt20": int((s.n_train < 20).sum()), "species_train_lt100": int((s.n_train < 100).sum()),
                     "train_specimen_share": float(tr.is_specimen.mean()) if len(tr) else np.nan,
                     "train_sources": tr.source.value_counts(normalize=True).round(3).to_dict(),
                     "unused_img": len(un), "species_with_unused": int(un.class_idx.nunique()),
                     "train_ext_img": int(s.n_train_ext.sum()), "val_ext_img": int(s.n_val_ext.sum()),
                     "test_img": int(s.n_test.sum())})
    data = pd.DataFrame(data)
    out = diag.merge(data, on="phylum", how="left")
    out.to_csv(RESULTS / "tables/moss_diagnosis.csv", index=False)
    pd.DataFrame(bins).to_csv(RESULTS / "tables/moss_diagnosis_bins.csv", index=False)
    pd.set_option("display.width", 250)
    print(diag.round(4).to_string(index=False))
    print(data.to_string(index=False))
    print(pd.DataFrame(bins).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

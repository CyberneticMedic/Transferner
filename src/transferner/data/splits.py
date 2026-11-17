"""Step 07. Train/val/test.

test  = regional Pl@ntNet/iNaturalist records with in_tol=False (no gbifID match and
        phash-distinct from every ToL image of the same species) and eventDate >=
        leakage.test_min_event_date; up to regional.test_per_species observations per species.
train/val = everything else that survived dedup, 70/15 (renormalised) grouped by
        observation_id and stratified by species, then capped per species
        (caps.default, caps.toxic_or_lookalike); overflow is split="unused".
Species with fewer than caps.min_images train+val images are flagged zero_shot_only
(kept in test, no probe/detector training weight).
Writes manifests/splits.parquet (uuid -> split) and updates species.parquet counts."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..utils.config import load, data_root


def h(s: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()[:12], 16)


def main() -> None:
    cfg = load("data")
    root = data_root()
    seed = cfg["splits"]["seed"]
    img = pd.read_parquet(root / "manifests/images.parquet")
    sp = pd.read_parquet(root / "manifests/species.parquet")
    img = img[img.keep].copy()
    img["obs_h"] = img.observation_id.map(lambda o: h(o, seed))
    img["split"] = None

    # ---- test: leakage-free regional observations
    since = str(cfg["leakage"]["test_min_event_date"])
    elig = (img.source == "gbif_regional") & ~img.in_tol & (img.event_date.fillna("") >= since)
    # an observation is test-eligible only if *all* its images are
    obs_ok = elig.groupby(img.observation_id).transform("all")
    cand = img[obs_ok & elig][["class_idx", "observation_id", "obs_h"]].drop_duplicates("observation_id")
    cand = cand.sort_values(["class_idx", "obs_h"])
    cand["r"] = cand.groupby("class_idx").cumcount()
    test_obs = set(cand.loc[cand.r < cfg["regional"]["test_per_species"], "observation_id"])
    img.loc[img.observation_id.isin(test_obs), "split"] = "test"

    # ---- train / val, grouped by observation, stratified by species
    tr, va = cfg["splits"]["ratios"]["train"], cfg["splits"]["ratios"]["val"]
    val_frac = va / (tr + va)
    rest = img[img.split.isna()]
    obs = rest[["class_idx", "observation_id", "obs_h"]].drop_duplicates("observation_id")
    obs = obs.sort_values(["class_idx", "obs_h"])
    obs["pos"] = obs.groupby("class_idx").cumcount()
    obs["n"] = obs.groupby("class_idx").observation_id.transform("size")
    # every val_period-th observation goes to val (deterministic, exactly stratified)
    period = max(int(round(1 / val_frac)), 2)
    obs["split"] = np.where((obs.pos % period == period - 1) & (obs.n >= 3), "val", "train")
    img.loc[rest.index, "split"] = rest.observation_id.map(dict(zip(obs.observation_id, obs.split)))

    # ---- per-species cap on train+val (regional extras first, then ToL by pick_rank)
    cap = sp.set_index("class_idx").pipe(
        lambda s: np.where(s.is_toxic | s.is_lookalike, cfg["caps"]["toxic_or_lookalike"], cfg["caps"]["default"]))
    cap = pd.Series(cap, index=sp.class_idx)
    tv = img[img.split.isin(["train", "val"])].copy()
    tv["_ord"] = (tv.source != "gbif_regional").astype(int)
    tv = tv.sort_values(["class_idx", "_ord", "pick_rank", "obs_h"])
    tv["r"] = tv.groupby("class_idx").cumcount()
    over = tv.index[tv.r >= tv.class_idx.map(cap)]
    img.loc[over, "split"] = "unused"

    out = img[["uuid", "class_idx", "observation_id", "source", "is_specimen", "split"]]
    out.to_parquet(root / "manifests/splits.parquet", index=False)

    counts = out.pivot_table(index="class_idx", columns="split", values="uuid", aggfunc="size", fill_value=0)
    for s in ("train", "val", "test", "unused"):
        sp[f"n_{s}"] = sp.class_idx.map(counts.get(s, pd.Series(dtype=int))).fillna(0).astype(int)
    sp["zero_shot_only"] = (sp.n_train + sp.n_val) < cfg["caps"]["min_images"]
    sp.to_parquet(root / "manifests/species.parquet", index=False)
    print(out.split.value_counts().to_dict())
    print(f"species with test images: {(sp.n_test > 0).sum()} | zero-shot only: {sp.zero_shot_only.sum()} "
          f"| median train/species: {int(sp.n_train.median())}")


if __name__ == "__main__":
    main()

"""Append the cultivated perennials to the label set (D-045; user answer of 2026-11-24: some cultivated species,
perennials only; data/extension/cultivated_perennials.csv, curated by data/cultivated_perennials.py).

build_species_list.py would also recompute every flag the D-045 repairs set (species_flags.py), so this
appends instead: one GBIF species/synonym lookup per new species (build_species_list.species_info, cached),
class indices continuing after the current maximum (existing indices never move), must_include = True and
cultivated = True. Training counts are zero until splits.py runs. Re-running is a no-op for species already in.
  python -m transferner.data.extend_species [--csv data/extension/cultivated_perennials.csv]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..utils.config import data_root
from .build_species_list import species_info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/extension/cultivated_perennials.csv")
    args = ap.parse_args()
    root = data_root()
    repo = Path(__file__).resolve().parents[3]
    p = root / "manifests/species.parquet"
    sp = pd.read_parquet(p)
    cur = pd.read_csv(repo / args.csv)
    new = cur[~cur.gbif_key.isin(set(sp.gbif_taxon_key))]
    if "cultivated" not in sp.columns:
        sp["cultivated"] = False
    if new.empty:
        print("nothing to add")
        return
    infos = [species_info(int(k)) for k in new.gbif_key]
    df = pd.DataFrame(infos)
    assert (df["rank"] == "SPECIES").all() and (df.kingdom_key == 6).all(), df[["scientific_name", "rank"]]
    clash = set(df.scientific_name) & set(sp.scientific_name)
    assert not clash, f"names already in the label set: {clash}"
    reg = pd.read_parquet(root / "manifests/regional_species.parquet")[["gbif_taxon_key", "n_regional", "n_regional_with_image"]]
    df = df.merge(reg, on="gbif_taxon_key", how="left")
    df[["n_regional", "n_regional_with_image"]] = df[["n_regional", "n_regional_with_image"]].fillna(0).astype(int)
    common = dict(zip(new.gbif_key, new.common_name))
    df["common_name"] = [common.get(k) or c for k, c in zip(df.gbif_taxon_key, df.common_name)]
    start = int(sp.class_idx.max()) + 1
    df["class_idx"] = range(start, start + len(df))
    df["must_include"] = True
    df["cultivated"] = True
    df["lookalike_groups"] = [[] for _ in range(len(df))]
    for c in ("is_lookalike", "has_traits", "is_toxic", "zero_shot_only"):
        df[c] = c == "zero_shot_only"
    for c in [c for c in sp.columns if c.startswith("n_") and c not in ("n_regional", "n_regional_with_image")]:
        df[c] = 0
    df = df.drop(columns=["rank", "kingdom_key"])
    out = pd.concat([sp, df[sp.columns]], ignore_index=True).sort_values("class_idx").reset_index(drop=True)
    assert out.class_idx.is_unique and out.gbif_taxon_key.is_unique and df.scientific_name.is_unique   # the 28 old twins stay (D-044)
    out.to_parquet(p, index=False)
    print(f"label set {len(sp):,} -> {len(out):,} species (classes {start}-{start + len(df) - 1}); new genera: "
          f"{sorted(set(df.genus) - set(sp.genus))[:60]}")


if __name__ == "__main__":
    main()

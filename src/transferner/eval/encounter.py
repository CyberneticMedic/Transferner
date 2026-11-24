"""Encounter weights for per-photo eval sets (D-040; audit evaluation-power-and-weighting, edible_targeting_app-01).

A capped or source-skewed photo set over-represents rare species. Weighting each photo by how often its
species (or genus) is met in the region, divided by how many photos of it the set holds, turns a per-photo
rate into the rate a walker would see. Counts are the regional GBIF observations in
manifests/priors_state_month.parquet over all states and months, plus 1, as export/bundle.forage computes e_g
(train_live_head.encounter_share is the genus version, also + 1). The counts are an observer-biased proxy for
what this user points the camera at; report per-photo rates next to them, never instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def species_counts(root: Path) -> np.ndarray:
    """Regional observations + 1 per species, indexed by class_idx (name twins keep their own taxon keys)."""
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "gbif_taxon_key"]).sort_values("class_idx")
    pr = pd.read_parquet(root / "manifests/priors_state_month.parquet").groupby("gbif_taxon_key")["count"].sum()
    return sp.gbif_taxon_key.map(pr).fillna(0).to_numpy(float) + 1.0


def photo_weights(cls: np.ndarray, counts: np.ndarray, genus_of: np.ndarray | None = None, by: str = "species") -> np.ndarray:
    """Per-photo weight = share of the photo's species (or genus) among encounters / its photos in this set.

    cls: class_idx per photo. counts: species_counts(). genus_of: genus id per class_idx (needed for by="genus").
    Every species (genus) present then carries its encounter share in total, however many photos it has, so a
    weighted mean of a per-photo outcome is the encounter-weighted rate. Weights sum to 1 over the set."""
    if by == "species":
        key, c = cls, counts
    elif by == "genus":
        assert genus_of is not None, "by='genus' needs genus_of"
        key = genus_of[cls]
        c = np.bincount(genus_of, weights=counts, minlength=int(genus_of.max()) + 1)
    else:
        raise ValueError(by)
    k = np.bincount(key, minlength=len(c)).astype(float)
    w = c[key] / k[key]
    return w / w.sum()


def wmean(x: np.ndarray, w: np.ndarray | None) -> float:
    """Mean of x, weighted by w when given; nan on an empty set."""
    if len(x) == 0:
        return float("nan")
    return float(np.mean(x)) if w is None else float(np.sum(w * x) / np.sum(w))


def n_eff(w: np.ndarray) -> float:
    """Kish effective sample size of a weighted set: how many photos the weighted rate is really worth."""
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0

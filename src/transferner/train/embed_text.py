"""Precompute BioCLIP text embeddings for species.parquet -> artifacts/text_emb.npy + labels.json."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..models import bioclip
from ..utils.config import data_root


def main() -> None:
    root = data_root()
    art = root / "artifacts"
    art.mkdir(exist_ok=True)
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx")
    assert (sp.class_idx.to_numpy() == np.arange(len(sp))).all(), "class_idx must be dense 0..N-1"
    model, _, tok = bioclip.load()
    emb = bioclip.embed_text(sp, model, tok)
    np.save(art / "text_emb.npy", emb.astype(np.float32))
    labels = [{"class_idx": int(r.class_idx), "scientific_name": r.scientific_name,
               "common_name": r.common_name, "genus": r.genus, "family": r.family,
               "gbif_taxon_key": int(r.gbif_taxon_key), "is_toxic": bool(r.is_toxic),
               "is_lookalike": bool(r.is_lookalike)} for r in sp.itertuples()]
    (art / "labels.json").write_text(json.dumps(labels, indent=1))
    (art / "logit_scale.txt").write_text(f"{model.logit_scale.exp().item():.6f}\n")
    print(f"text_emb {emb.shape}, logit_scale {model.logit_scale.exp().item():.2f}")


if __name__ == "__main__":
    main()

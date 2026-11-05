"""Step 07. Train/val/test.

test  = regional Pl@ntNet/GBIF records with in_tol=False, phash-distinct from all
        ToL images, and eventDate >= leakage.test_min_event_date.
train/val = everything else, 70/15 grouped by observation_id, stratified by species.
Writes data/manifests/splits.parquet (uuid -> split)."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

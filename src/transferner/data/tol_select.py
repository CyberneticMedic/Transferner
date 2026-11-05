"""Step 03. Select TreeOfLife-200M rows for the label set.

DuckDB over dataset/catalog.parquet (16.8 GB), never pandas:
  SELECT uuid, source_id, source_url, publisher, basis_of_record, img_type, <taxonomy>
  WHERE species IN label_set AND kingdom = 'Plantae'.
Drops museum specimens (img_type 'Museum Specimen: Plant') from detector training
but keeps them for BioCLIP probe training (configurable).
Writes data/manifests/tol_candidates.parquet."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

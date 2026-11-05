"""Step 01. Regional species universe from GBIF.

Submit a GBIF occurrence download (format=SPECIES_LIST) with predicate:
  gadm.level1Gid in region AND kingdomKey=Plantae AND occurrenceStatus=PRESENT.
SPECIES_LIST returns per-taxon occurrence counts without pulling 10^7 rows.
A second download (format=SIMPLE_CSV, mediaType=StillImage) per state x month
feeds the priors table (configs/data.yaml: priors).
Writes data/manifests/regional_species.parquet, priors_state_month.parquet."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

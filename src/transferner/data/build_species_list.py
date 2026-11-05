"""Step 02. Final label set.

label_set = {regional species with >= min_regional_occurrences}
          U must_include.txt U every species in lookalike_groups.csv
Reconcile names to accepted GBIF taxonKeys; attach genus/family.
Flags each species: has_traits, is_toxic, is_lookalike (drives the 2x cap).
Writes data/manifests/species.parquet (THE label table; class indices are frozen here)."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

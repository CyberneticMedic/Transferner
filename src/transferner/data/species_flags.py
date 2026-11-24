"""Recompute the label table's training flags from the cited trait rows (D-045; audit capture_lookalike_resolution-8).

species.parquet was built before species_traits.csv (D-027), so is_toxic marked 4 species and has_traits none,
and D-009's 2x training cap (configs/data.yaml caps.toxic_or_lookalike) reached only the 43 species of the
look-alike csv. This sets:
  has_traits    the species has any cited trait row
  is_toxic      it has a cited toxic row
  is_lookalike  it is in data/traits/lookalike_groups.csv (as before) OR in an empirical edible<->toxic confusion
                pair of the production probe on regional VAL (data/traits/lookalike_empirical.csv, both members:
                flagging one side only would leave a prior pulling predictions toward it)
Only these three columns change; class indices and everything else stay. tol_select.py and splits.py read them.
  python -m transferner.data.species_flags"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.config import data_root


def main() -> None:
    root = data_root()
    repo = Path(__file__).resolve().parents[3]
    p = root / "manifests/species.parquet"
    sp = pd.read_parquet(p)
    tr = pd.read_csv(repo / "data/traits/species_traits.csv")
    emp = set(pd.read_csv(repo / "data/traits/lookalike_empirical.csv").scientific_name)
    before = sp[["has_traits", "is_toxic", "is_lookalike"]].sum().to_dict()
    sp["has_traits"] = sp.scientific_name.isin(set(tr.scientific_name))
    sp["is_toxic"] = sp.scientific_name.isin(set(tr[tr.category == "toxic"].scientific_name))
    sp["is_lookalike"] = sp.is_lookalike.astype(bool) | sp.scientific_name.isin(emp)
    after = sp[["has_traits", "is_toxic", "is_lookalike"]].sum().to_dict()
    sp.to_parquet(p, index=False)
    print(f"before {before}; after {after}; 2x cap now applies to {int((sp.is_toxic | sp.is_lookalike).sum())} species")


if __name__ == "__main__":
    main()

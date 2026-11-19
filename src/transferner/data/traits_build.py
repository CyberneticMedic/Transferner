"""Phase C: build data/traits/species_traits.csv from cited sources, plus a coverage report.

  python -m transferner.data.traits_build      # after `traits_ncsu fetch` and `traits_ncsu parse`

Rows written here carry curator `auto:<source>` and are regenerated on every run; rows with any
other curator (hand-curated) are kept untouched. Nothing is inferred beyond what the source
page states (D-005, OVERNIGHT.md hard rule):

NC State Extension Gardener Plant Toolbox
  toxic        page has a "Poisonous to Humans" section -> one row per listed Poison Part
               (no parts listed -> part `whole`), ingestion_toxic = 1, ingestion_severity from
               Poison Severity (Low/Medium/High -> low/moderate/high), contact_toxic = 1 when
               "Causes Contact Dermatitis: Yes".
  edible       Plant Type "Edible" or an "#edible <part>" tag -> one row per tagged part
               (no part tag -> part `unspecified`), conditions = the page's Edibility text.
  conditional  the same edible rows when the page ALSO has a poison section: the source
               itself says both, so the edible claim is downgraded (more toxic wins).
  genus pages  a species with no page of its own takes its genus page (level=genus): toxic
               rows apply as stated (conservative), edible rows always become conditional,
               and notes say the species was not individually assessed.
Absence of a toxic row never means "safe": coverage is reported, not assumed.

Writes data/traits/species_traits.csv and results/tables/trait_coverage.csv, and prints
coverage by phylum / priority set / test-image share."""
from __future__ import annotations

import pandas as pd

from ..utils.config import REPO, RESULTS, data_root, load

SRC_NCSU = "NC State Extension Gardener Plant Toolbox"
CURATOR_NCSU = "auto:traits_ncsu"
COLS = ["scientific_name", "gbif_taxon_key", "common_name", "part", "category", "contact_toxic",
        "ingestion_toxic", "ingestion_severity", "conditions", "source_name", "source_url", "accessed",
        "curator", "notes"]

PART = {"Leaves": "leaf", "Fruits": "fruit", "Roots": "root", "Flowers": "flower", "Stems": "stem",
        "Bark": "bark", "Seeds": "seed", "Sap/Juice": "sap", "Bulbs": "root", "Nectar": "flower",
        "Pollen": "flower", "Shoots": "shoot", "Tubers": "root", "Rhizomes": "root", "Nuts": "seed"}
EDIBLE_TAG = {"edible fruits": "fruit", "edible fruit": "fruit", "edible flowers": "flower",
              "edible flower": "flower", "edible leaves": "leaf", "edible leaf": "leaf", "edible nuts": "seed",
              "edible nut": "seed", "edible seeds": "seed", "edible seed": "seed", "edible roots": "root",
              "edible root": "root", "edible stems": "stem", "edible shoots": "shoot", "edible tubers": "root",
              "edible bark": "bark", "edible sap": "sap"}
EDIBLE_GENERIC = {"edible", "native edible", "edible weed", "easy edibles"}   # no part named
SEVERITY = {"Low": "low", "Medium": "moderate", "High": "high"}


def _clip(s: str, n: int = 300) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def ncsu_rows(f: pd.DataFrame, common: dict[int, str]) -> list[dict]:
    rows = []
    for r in f.itertuples():
        cn = common.get(int(r.gbif_taxon_key)) or (r.common_names[0] if len(r.common_names) else "")
        base = {"scientific_name": r.scientific_name, "gbif_taxon_key": int(r.gbif_taxon_key),
                "common_name": cn, "source_name": SRC_NCSU,
                "source_url": r.url, "accessed": r.accessed, "curator": CURATOR_NCSU}
        genus = getattr(r, "level", "species") == "genus"
        syn = (f"genus-level page ({r.page_name}), species not individually assessed; " if genus else
               f"page filed under synonym {r.page_name}; " if r.page_name != r.scientific_name else "")
        toxic = bool(r.has_poison_section)
        contact = int(r.contact_dermatitis == "Yes")
        if toxic:
            sev = SEVERITY.get(r.poison_severity, "")
            parts = sorted({PART.get(p, p.lower()) for p in r.poison_parts}) or ["whole"]
            unknown = [p for p in r.poison_parts if p not in PART]
            note = (f"{syn}severity {r.poison_severity or 'not given'}; toxic principle: {_clip(r.poison_principle, 120)}; "
                    f"symptoms: {_clip(r.poison_symptoms)}" + (f"; unmapped parts {unknown}" if unknown else ""))
            for p in parts:
                rows.append({**base, "part": p, "category": "toxic", "contact_toxic": contact, "ingestion_toxic": 1,
                             "ingestion_severity": sev, "conditions": "", "notes": note})
        edible_parts = sorted({EDIBLE_TAG[t.lower()] for t in r.tags if t.lower() in EDIBLE_TAG})
        generic = any(t.lower() in EDIBLE_GENERIC for t in r.tags)
        if "Edible" in list(r.plant_type) or edible_parts or generic:
            cat = "conditional" if toxic or genus else "edible"
            note = syn + ("source also lists this plant as poisonous to humans; edible use only as the source "
                          "describes" if toxic else "edible claim made for the genus" if genus else "")
            for p in edible_parts or ["unspecified"]:
                rows.append({**base, "part": p, "category": cat, "contact_toxic": contact if toxic else 0,
                             "ingestion_toxic": 0, "ingestion_severity": "", "conditions": _clip(r.edibility, 400),
                             "notes": note.strip("; ")})
    return rows


def coverage(traits: pd.DataFrame, sp: pd.DataFrame, pages: pd.DataFrame) -> pd.DataFrame:
    g = traits.groupby("gbif_taxon_key")
    flags = pd.DataFrame({
        "any_row": g.size() > 0,
        "toxic": g.apply(lambda d: (d.category == "toxic").any()),
        "contact_toxic": g.contact_toxic.max() == 1,
        "ingestion_toxic": g.ingestion_toxic.max() == 1,
        "edible": g.apply(lambda d: (d.category == "edible").any()),
        "conditional": g.apply(lambda d: (d.category == "conditional").any()),
    })
    df = sp.set_index("gbif_taxon_key")[["scientific_name", "phylum", "n_regional", "n_test", "must_include",
                                          "is_lookalike"]].join(flags)
    df["has_source_page"] = df.index.isin(pages.gbif_taxon_key)
    df["has_species_page"] = df.index.isin(pages.loc[pages.get("level", "species") == "species", "gbif_taxon_key"])
    df[flags.columns] = df[flags.columns].fillna(False).astype(bool)
    sets = {"all": df.index == df.index, "must_include": df.must_include, "lookalike": df.is_lookalike,
            "top500_regional": df.n_regional.rank(ascending=False, method="first") <= 500}
    for ph in df.phylum.dropna().unique():
        sets[ph] = df.phylum == ph
    out = []
    for name, m in sets.items():
        d = df[m]
        out.append({"set": name, "species": len(d), "with_species_page": int(d.has_species_page.sum()),
                    "with_page": int(d.has_source_page.sum()),
                    "with_rows": int(d.any_row.sum()), "toxic": int(d.toxic.sum()),
                    "contact_toxic": int(d.contact_toxic.sum()), "ingestion_toxic": int(d.ingestion_toxic.sum()),
                    "edible": int(d.edible.sum()), "conditional": int(d.conditional.sum()),
                    "test_images": int(d.n_test.sum()),
                    "test_image_share_with_page": round(d.loc[d.has_source_page, "n_test"].sum() / max(d.n_test.sum(), 1), 4)})
    return pd.DataFrame(out)


def main() -> None:
    root = data_root()
    path = REPO / load("species")["traits"]
    old = pd.read_csv(path)
    keep = old[~old.curator.fillna("").str.startswith("auto:")]
    pages = pd.read_parquet(root / "raw/ncsu/ncsu_fields.parquet")
    sp = pd.read_parquet(root / "manifests/species.parquet")
    common = dict(zip(sp.gbif_taxon_key.astype(int), sp.common_name.fillna("")))
    new = pd.DataFrame(ncsu_rows(pages, common), columns=COLS)
    traits = pd.concat([keep, new], ignore_index=True)[COLS]
    traits = traits.sort_values(["scientific_name", "category", "part", "source_name"]).reset_index(drop=True)
    traits.to_csv(path, index=False)
    cov = coverage(traits, sp, pages)
    cov.to_csv(RESULTS / "tables" / "trait_coverage.csv", index=False)
    print(f"wrote {len(traits)} rows ({len(new)} from {SRC_NCSU}, {len(keep)} hand-curated kept) -> {path}")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()

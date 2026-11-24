"""Step 03. Select TreeOfLife-200M rows for the label set.

DuckDB over raw/tol/plantae.parquet (the Plantae slice built in step 00), never pandas
over the full catalog. ToL names are matched on accepted GBIF names plus synonyms.

Curation policy (D-021): every image carries a `curation_tier`; tiers outside
curation.allowed_tiers, machine IDs, fossils, mis-kingdomed museum images and
licenses that do not permit ML training are dropped here.

Per species, candidates are over-sampled (cap x oversample) so the cap still holds
after dedup and failed downloads. Images are spread across observations (one
image per source_id first), and museum specimens are limited to
curation.max_specimen_share of the quota. Specimens stay flagged is_specimen
so the detector can exclude them (D-020).
Writes manifests/tol_candidates.parquet."""
from __future__ import annotations

import duckdb
import pandas as pd

from ..utils.config import load, data_root

TIER_SQL = """
CASE
  WHEN basis_of_record IN ('PRESERVED_SPECIMEN','LIVING_SPECIMEN','MATERIAL_SAMPLE')
       OR img_type = 'Museum Specimen: Plant'              THEN 'institutional_specimen'
  WHEN data_source = 'eol'                                 THEN 'eol_curated'
  WHEN publisher IN ('iNaturalist.org','Pl@ntNet','observation.org') THEN 'community_verified'
  WHEN basis_of_record IN ('HUMAN_OBSERVATION','OBSERVATION','OCCURRENCE') THEN 'other_observation'
  ELSE 'unknown'
END"""


def main() -> None:
    cfg = load("data")
    cur = cfg["curation"]
    root = data_root()
    sp = pd.read_parquet(root / "manifests/species.parquet")

    accepted = set(sp.scientific_name)
    names = [(r.class_idx, r.scientific_name, 0) for r in sp.itertuples()]
    names += [(r.class_idx, s, 1) for r in sp.itertuples() for s in r.synonyms if s not in accepted]
    nm = pd.DataFrame(names, columns=["class_idx", "name", "is_synonym"])
    # a synonym claimed by two label species is ambiguous: drop it
    nm = nm[~(nm.is_synonym.astype(bool) & nm.duplicated("name", keep=False))].drop_duplicates("name")
    # D-045: a synonym that GBIF resolves to another taxon (later homonym, misapplied name) pulls that plant's
    # photos into the class (thimbleberry under Rubus cuneifolius): never match it (data/homonyms.py)
    hp = root / "manifests/homonym_synonyms.parquet"
    if hp.exists():
        bad = pd.read_parquet(hp, columns=["class_idx", "synonym", "decision"])
        bad = set(zip(bad[bad.decision == "drop"].class_idx, bad[bad.decision == "drop"].synonym))
        nm = nm[[(c, n) not in bad for c, n in zip(nm.class_idx, nm.name)]]

    caps = sp[["class_idx"]].copy()
    big = sp.is_toxic | sp.is_lookalike
    caps["cap"] = big.map({True: cfg["caps"]["toxic_or_lookalike"], False: cfg["caps"]["default"]}).values
    caps["quota"] = (caps.cap * cur["oversample"]).round().astype(int)
    caps["spec_quota"] = (caps.quota * cur["max_specimen_share"]).round().astype(int)

    c = duckdb.connect()
    c.execute("SET threads=16; SET memory_limit='40GB'")
    c.register("nm", nm)
    c.register("caps", caps)
    T = root / "raw/tol/plantae.parquet"
    tiers = ",".join(f"'{t}'" for t in cur["allowed_tiers"])
    bad_lic = ",".join(f"'{l}'" for l in cur["drop_licenses"])
    q = f"""
    WITH base AS (
      SELECT nm.class_idx, nm.is_synonym, t.*, {TIER_SQL} AS curation_tier
      FROM '{T}' t JOIN nm ON t.scientific_name = nm.name
      WHERE coalesce(t.basis_of_record, '') NOT IN ('MACHINE_OBSERVATION', 'FOSSIL_SPECIMEN')
        AND (t.img_type IS NULL OR t.img_type NOT LIKE 'Museum Specimen:%' OR t.img_type = 'Museum Specimen: Plant')
        AND t.license_name IS NOT NULL AND t.license_name NOT IN ({bad_lic})
        AND t.source_url IS NOT NULL
    ), tiered AS (
      SELECT *, curation_tier = 'institutional_specimen' AS is_specimen FROM base WHERE curation_tier IN ({tiers})
    ), ranked AS (
      SELECT *,
        row_number() OVER (PARTITION BY class_idx, source_id ORDER BY hash(uuid)) AS img_in_obs,
        count(*) OVER (PARTITION BY class_idx) AS n_avail
      FROM tiered
    ), ordered AS (
      SELECT *, row_number() OVER (PARTITION BY class_idx, is_specimen
                                   ORDER BY img_in_obs, is_synonym, hash(uuid || '{cfg['splits']['seed']}')) AS rk
      FROM ranked
    ), picked AS (
      -- specimens up to spec_quota; field photos fill the rest of the quota
      SELECT o.*, caps.cap, caps.quota FROM ordered o JOIN caps USING (class_idx)
      WHERE (o.is_specimen AND o.rk <= caps.spec_quota) OR (NOT o.is_specimen AND o.rk <= caps.quota)
    )
    SELECT * FROM (
      SELECT *, row_number() OVER (PARTITION BY class_idx ORDER BY is_specimen, rk) AS pick_rank FROM picked
    ) WHERE pick_rank <= quota
    """
    df = c.execute(q).df()
    df = df.rename(columns={"license_name": "license", "copyright_owner": "rights_holder",
                            "source_url": "url", "basis_of_record": "basis"})
    df["source"] = "tol"
    df["observation_id"] = df.data_source + ":" + df.source_id
    keep = ["uuid", "class_idx", "source", "data_source", "source_id", "observation_id", "url", "license",
            "rights_holder", "license_link", "publisher", "basis", "img_type", "curation_tier", "is_specimen",
            "is_synonym", "scientific_name", "n_avail", "cap", "pick_rank"]
    df[keep].to_parquet(root / "manifests/tol_candidates.parquet", index=False)

    per = df.groupby("class_idx").size().reindex(sp.class_idx, fill_value=0)
    print(f"ToL candidates: {len(df):,} images for {(per > 0).sum()}/{len(sp)} species")
    print(f"species with 0 ToL images: {(per == 0).sum()}  | below min_images ({cfg['caps']['min_images']}): "
          f"{(per < cfg['caps']['min_images']).sum()}")
    print("by tier:", df.curation_tier.value_counts().to_dict())
    print("specimen share:", round(df.is_specimen.mean(), 3))


if __name__ == "__main__":
    main()

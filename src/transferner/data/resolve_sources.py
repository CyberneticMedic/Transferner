"""Step 00. Resolve external identifiers before any download.

- Pl@ntNet DOI -> GBIF datasetKey (GET /dataset/doi/{doi}); iNaturalist research-grade datasetKey.
- Verify GADM level-1 IDs for SC/NC/GA/TN against live GBIF records.
- Resolve GBIF backbone keys for the included phyla.
- Record the ToL-200M and BioCLIP commit hashes and the GBIF snapshot ToL was built from.
- Build raw/tol/plantae.parquet: the Plantae slice of catalog.parquet joined with the
  per-image licenses in provenance.parquet (one 27 GB scan instead of one per query).
Writes manifests/sources.json."""
from __future__ import annotations

import json
import time

import duckdb

from ..utils.config import load, data_root
from ..utils.http import GBIF, get_json

INAT_RG_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"   # "iNaturalist Research-grade Observations"


def build_plantae(root) -> str:
    out = root / "raw/tol/plantae.parquet"
    if out.exists():
        return str(out)
    cat = root / "raw/tol/dataset/catalog.parquet"
    prov = root / "raw/tol/dataset/metadata/provenance.parquet"
    tmp = root / "cache/duckdb"
    tmp.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute(f"SET threads=16; SET memory_limit='40GB'; SET temp_directory='{tmp}'")
    c.execute(f"""COPY (SELECT c.uuid, c.source_url, c.phylum, c.class, c."order", c.family, c.genus,
        c.species, c.scientific_name, c.common, c.data_source, c.publisher, c.basis_of_record,
        c.img_type, c.source_id, p.license_name, p.copyright_owner, p.license_link
      FROM '{cat}' c LEFT JOIN '{prov}' p USING (uuid) WHERE c.kingdom = 'Plantae')
      TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    return str(out)


def main() -> None:
    cfg, sp = load("data"), load("species")
    root = data_root()
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    pn = get_json(f"{GBIF}/dataset/doi/{cfg['sources']['plantnet']['doi']}")["results"][0]
    inat = get_json(f"{GBIF}/dataset/{INAT_RG_DATASET}")

    gadm = {}
    for gid, name in zip(cfg["region"]["gadm_level1"], cfg["region"]["names"]):
        r = get_json(f"{GBIF}/occurrence/search", {"gadmGid": gid, "limit": 1}, cache=False)
        got = r["results"][0]["gadm"]["level1"]["name"] if r["results"] else None
        if got != name:
            raise SystemExit(f"GADM id {gid} resolves to {got!r}, expected {name!r}")
        gadm[gid] = name

    phyla = {}
    for name in sp["taxa"]["include_groups"]:
        m = get_json(f"{GBIF}/species/match", {"name": name, "rank": "PHYLUM", "kingdom": "Plantae"})
        if m.get("rank") != "PHYLUM":
            raise SystemExit(f"could not resolve phylum {name}: {m}")
        phyla[name] = m["usageKey"]

    tol = get_json(f"https://huggingface.co/api/datasets/{cfg['sources']['tol200m']['hf_repo']}", cache=False)
    bio_repo = load("bioclip")["model"]["hf_hub"].removeprefix("hf-hub:")
    bio = get_json(f"https://huggingface.co/api/models/{bio_repo}", cache=False)

    t = time.time()
    plantae = build_plantae(root)

    out = {
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plantnet": {"datasetKey": pn["key"], "title": pn["title"], "doi": cfg["sources"]["plantnet"]["doi"]},
        "inaturalist": {"datasetKey": INAT_RG_DATASET, "title": inat["title"], "doi": inat.get("doi")},
        "gadm_level1": gadm,
        "phylum_keys": phyla,
        "plantae_kingdom_key": 6,
        "tol200m": {"repo": tol["id"], "sha": tol["sha"], "lastModified": tol["lastModified"],
                    "gbif_snapshot": "2024-05-01", "gbif_snapshot_doi": "10.15468/dl.bfv433",
                    "plantae_parquet": plantae},
        "bioclip": {"repo": bio["id"], "sha": bio["sha"]},
    }
    (root / "manifests/sources.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), f"\n(plantae parquet ready in {time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

"""Step 04. Recent regional photos from Pl@ntNet + iNaturalist research grade (via GBIF).

For every label species: occurrences in the region (gadm level-1), eventDate >=
regional.min_event_date (after ToL's GBIF snapshot, so they cannot be in BioCLIP's
training data), with StillImage media. Up to regional.per_species observations per
species. Keeps gbifID, eventDate, stateProvince, organ (Pl@ntNet title suffix),
license, rightsHolder. These are the leakage-free test pool (step 07).
Uses the public search API (no account needed).
Writes manifests/gbif_candidates.parquet."""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from ..utils.config import load, data_root
from ..utils.http import GBIF, get_json

MAX_IMAGES_PER_OBS = 3
ORGANS = {"leaf", "flower", "fruit", "bark", "habit", "other"}


def norm_license(s: str | None) -> str | None:
    """Media license (URL or 'user (cc-by-sa)') -> short code like ToL's (cc-by-4.0, cc-0-1.0)."""
    if not s:
        return None
    s = s.lower()
    if "publicdomain/zero" in s or "cc0" in s:
        return "cc-0-1.0"
    if "publicdomain/mark" in s:
        return "public-domain-mark"
    m = re.search(r"licenses/([a-z\-]+)/(\d\.\d)", s)
    if m:
        return f"cc-{m.group(1)}-{m.group(2)}"
    m = re.search(r"\((cc-[a-z\-]+)\)", s)
    if m:
        return m.group(1)
    if "all rights reserved" in s:
        return "all-rights-reserved"
    return "other"


def fetch_species(key: int, datasets: list[str], gadm: list[str], since: str, n: int) -> list[dict]:
    rows, offset = [], 0
    while len({r["gbif_id"] for r in rows}) < n:
        r = get_json(f"{GBIF}/occurrence/search", {
            "taxonKey": key, "datasetKey": datasets, "gadmGid": gadm, "mediaType": "StillImage",
            "eventDate": f"{since},*", "occurrenceStatus": "PRESENT", "limit": 300, "offset": offset})
        for o in r["results"]:
            imgs = [m for m in o.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
            for i, m in enumerate(imgs[:MAX_IMAGES_PER_OBS]):
                title = (m.get("title") or "")
                organ = title.rsplit(":", 1)[-1].strip().lower() if ":" in title else None
                rows.append({
                    "uuid": f"gbif-{o['key']}-{i}", "gbif_id": str(o["key"]), "taxon_key": key,
                    "dataset_key": o["datasetKey"], "url": m["identifier"],
                    "license": norm_license(m.get("license") or o.get("license")),
                    "rights_holder": m.get("rightsHolder") or m.get("creator"),
                    "event_date": (o.get("eventDate") or "")[:10] or None,
                    "state": (o.get("gadm") or {}).get("level1", {}).get("name"),
                    "organ": organ if organ in ORGANS else None,
                    "gbif_scientific_name": o.get("species"),
                    "occurrence_ref": o.get("occurrenceID") or o.get("references"),
                })
        offset += 300
        if r["endOfRecords"]:
            break
    keep = set(list(dict.fromkeys(r["gbif_id"] for r in rows))[:n])   # first n observations, in order
    return [r for r in rows if r["gbif_id"] in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    cfg = load("data")
    reg = cfg["regional"]
    root = data_root()
    src = json.loads((root / "manifests/sources.json").read_text())
    datasets = [src[d]["datasetKey"] for d in reg["datasets"]]
    sp = pd.read_parquet(root / "manifests/species.parquet")
    since = str(reg["min_event_date"])

    failed = []

    def job(row):
        try:
            rows = fetch_species(row.gbif_taxon_key, datasets, cfg["region"]["gadm_level1"],
                                 since, reg["per_species"])
        except Exception as e:                       # noqa: BLE001 - keep going; re-run fills gaps (cached)
            failed.append((row.scientific_name, repr(e)[:120]))
            return []
        return [{**r, "class_idx": row.class_idx} for r in rows]

    with ThreadPoolExecutor(args.workers) as ex:
        out = [r for rows in tqdm(ex.map(job, sp.itertuples()), total=len(sp), desc="regional")
               for r in rows]
    if failed:
        print(f"warning: {len(failed)} species failed (re-run to fill; successes are cached): {failed[:3]}")
    df = pd.DataFrame(out)
    bad = set(cfg["curation"]["drop_licenses"])
    df = df[df.license.notna() & ~df.license.isin(bad)]
    df["source"] = "gbif_regional"
    df["observation_id"] = "gbif:" + df.gbif_id
    names = {v["datasetKey"]: k for k, v in src.items() if isinstance(v, dict) and "datasetKey" in v}
    df["publisher"] = df.dataset_key.map(names)
    df["curation_tier"] = "community_verified"
    df["is_specimen"] = False
    df.to_parquet(root / "manifests/gbif_candidates.parquet", index=False)
    per = df.groupby("class_idx").observation_id.nunique().reindex(sp.class_idx, fill_value=0)
    print(f"regional: {len(df):,} images / {df.observation_id.nunique():,} observations "
          f"for {(per > 0).sum()}/{len(sp)} species; by publisher {df.publisher.value_counts().to_dict()}")
    print(f"species with >= {reg['test_per_species']} recent regional observations: "
          f"{(per >= reg['test_per_species']).sum()}; organ-tagged images: {df.organ.notna().sum():,}")


if __name__ == "__main__":
    main()

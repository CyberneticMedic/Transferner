"""Month coverage of the regional photo pool (D-044; audit capture_model-3).

gbif_media.py kept the first 120 observations per species in GBIF's default order, so a common species'
regional photos cover a median 112-day window (yaupon: only January-February, though GBIF holds sightings in
every month; muscadine: March-April, not the fruiting season). This counts, per label species, the regional
observations with images in each month (same datasets, region and date cutoff as gbif_media.py), with GBIF's
month facet: one metadata request per species, no photos. The month-stratified re-pull uses the counts.
Writes manifests/gbif_month_counts.parquet (class_idx, gbif_taxon_key, month, count).
  python -m transferner.data.gbif_months [--workers 3]"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from ..utils.config import data_root, load
from ..utils.http import GBIF, get_json


def months(key: int, datasets: list[str], gadm: list[str], since: str) -> dict[int, int]:
    r = get_json(f"{GBIF}/occurrence/search", {"taxonKey": key, "datasetKey": datasets, "mediaType": "StillImage",
                                               "eventDate": f"{since},*", "occurrenceStatus": "PRESENT", "gadmGid": gadm,
                                               "limit": 0, "facet": "month", "facetLimit": 12})
    out = {m: 0 for m in range(1, 13)}
    for f in r.get("facets", []):
        for c in f.get("counts", []):
            out[int(c["name"])] = int(c["count"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3, help="GBIF throttles above ~3 concurrent requests")
    args = ap.parse_args()
    cfg = load("data")
    reg = cfg["regional"]
    root = data_root()
    src = json.loads((root / "manifests/sources.json").read_text())
    datasets = [src[d]["datasetKey"] for d in reg["datasets"]]
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "gbif_taxon_key", "scientific_name"])
    failed = []

    def job(row):
        try:
            return [(row.class_idx, row.gbif_taxon_key, m, c) for m, c in months(row.gbif_taxon_key, datasets,
                                                                                 cfg["region"]["gadm_level1"], str(reg["min_event_date"])).items()]
        except Exception as e:                       # noqa: BLE001 - re-run fills gaps (requests are cached)
            failed.append((row.scientific_name, repr(e)[:120]))
            return []

    with ThreadPoolExecutor(args.workers) as ex:
        rows = [r for rs in tqdm(ex.map(job, sp.itertuples()), total=len(sp), desc="month facets") for r in rs]
    df = pd.DataFrame(rows, columns=["class_idx", "gbif_taxon_key", "month", "count"])
    out = root / "manifests/gbif_month_counts.parquet"
    df.to_parquet(out, index=False)
    tot = df.groupby("month")["count"].sum()
    print(f"{df.class_idx.nunique():,} species, {int(tot.sum()):,} regional observations with images; by month: {tot.to_dict()}")
    if failed:
        print(f"warning: {len(failed)} species failed (re-run): {failed[:3]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

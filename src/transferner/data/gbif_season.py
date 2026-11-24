"""Month-balanced regional photos (D-045; audit capture_model-3).

gbif_media.py kept each species' first 120 observations in GBIF's default order, so a common species' regional
photos sit in a 2-3 month window (yaupon: January-February only; muscadine: March-April, not the fruiting
season); November and December were barely covered (1,335 December observations against 35,379 for April).
Using the per-month counts of data/gbif_months.py, this pulls two new candidate sets, both with the same
datasets, region and post-snapshot date cutoff as gbif_media.py, and never an observation already fetched:
  gbif_season       training rows for the 878 species with more than 120 regional observations: each month is
                    filled toward a quota proportional to sqrt(that month's sightings), --per-species in total.
                    splits.py puts them in train/val only, never test, so the frozen test set does not move.
  gbif_season_eval  an out-of-window evaluation slice: up to --eval-per-month observations in each of October to
                    February for species with cited trait rows and at least 10 sightings in those months.
                    splits.py makes them split "test_season" (dropped if a ToL near-duplicate exists).
Writes manifests/gbif_season.parquet (columns as gbif_candidates.parquet plus source); download.py queues it.
  python -m transferner.data.gbif_season [--per-species 80 --eval-per-month 4 --workers 3]"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.config import data_root, load
from ..utils.http import GBIF, get_json
from .gbif_media import MAX_IMAGES_PER_OBS, ORGANS, norm_license

WINTER = (10, 11, 12, 1, 2)


def fetch_month(key: int, datasets, gadm, since: str, month: int, n: int, skip: set, max_pages: int = 3) -> list[dict]:
    """First n new observations (with images) of taxon `key` in calendar month `month` (any year >= since),
    looking at most max_pages x 300 results deep: where a month's first observations were all fetched before,
    deep paging cost ~6 s per request (a 6-hour pull) for a handful of extra photos."""
    rows, seen, offset = [], set(), 0
    params = {"taxonKey": key, "datasetKey": datasets, "mediaType": "StillImage", "eventDate": f"{since},*",
              "occurrenceStatus": "PRESENT", "limit": 300, "gadmGid": gadm, "month": month}
    while len(seen) < n:
        r = get_json(f"{GBIF}/occurrence/search", {**params, "offset": offset})
        for o in r["results"]:
            gid = str(o["key"])
            if gid in skip or gid in seen or len(seen) >= n:
                continue
            imgs = [m for m in o.get("media", []) if m.get("type") == "StillImage" and m.get("identifier")]
            if not imgs:
                continue
            seen.add(gid)
            for i, m in enumerate(imgs[:MAX_IMAGES_PER_OBS]):
                title = m.get("title") or ""
                organ = title.rsplit(":", 1)[-1].strip().lower() if ":" in title else None
                rows.append({"uuid": f"gbif-{gid}-{i}", "gbif_id": gid, "taxon_key": key, "dataset_key": o["datasetKey"],
                             "url": m["identifier"], "license": norm_license(m.get("license") or o.get("license")),
                             "rights_holder": m.get("rightsHolder") or m.get("creator"),
                             "event_date": (o.get("eventDate") or "")[:10] or None,
                             "state": (o.get("gadm") or {}).get("level1", {}).get("name"),
                             "organ": organ if organ in ORGANS else None, "gbif_scientific_name": o.get("species"),
                             "occurrence_ref": o.get("occurrenceID") or o.get("references")})
        offset += 300
        if r["endOfRecords"] or offset >= 300 * max_pages:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=80, help="month-balanced observations per species (existing ones count)")
    ap.add_argument("--min-obs", type=int, default=121, help="species with at most this many - 1 sightings already have all of them")
    ap.add_argument("--eval-per-month", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3, help="GBIF throttles above ~3 concurrent requests")
    args = ap.parse_args()
    cfg = load("data")
    reg = cfg["regional"]
    root = data_root()
    src = json.loads((root / "manifests/sources.json").read_text())
    datasets = [src[d]["datasetKey"] for d in reg["datasets"]]
    gadm = cfg["region"]["gadm_level1"]
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "gbif_taxon_key", "scientific_name", "has_traits"])
    mc = pd.read_parquet(root / "manifests/gbif_month_counts.parquet").pivot_table(index="class_idx", columns="month", values="count", fill_value=0)
    prev = [pd.read_parquet(root / f"manifests/{n}.parquet", columns=["class_idx", "gbif_id", "event_date"])
            for n in ("gbif_candidates", "gbif_external") if (root / f"manifests/{n}.parquet").exists()]
    prev = pd.concat(prev, ignore_index=True).drop_duplicates("gbif_id")
    skip = set(prev.gbif_id)
    prev["m"] = pd.to_datetime(prev.event_date, errors="coerce").dt.month
    have = prev.groupby(["class_idx", "m"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)

    jobs = []
    tot = mc.sum(1)
    for c in tot.index[tot >= args.min_obs]:
        s = np.sqrt(mc.loc[c].reindex(range(1, 13), fill_value=0).to_numpy(float))
        target = np.round(args.per_species * s / s.sum()).astype(int)
        h = have.loc[c].to_numpy() if c in have.index else np.zeros(12, int)
        for m in range(1, 13):
            need = int(target[m - 1] - h[m - 1])
            if need > 0:
                jobs.append(("gbif_season", c, m, need))
    traits = set(sp[sp.has_traits].class_idx)
    win = mc[[m for m in WINTER if m in mc.columns]].sum(1)
    for c in win.index[(win >= 10) & win.index.isin(traits)]:
        for m in WINTER:
            if mc.loc[c].get(m, 0) > 0:
                jobs.append(("gbif_season_eval", c, m, args.eval_per_month))
    key = dict(zip(sp.class_idx, sp.gbif_taxon_key))
    print(f"{sum(j[0] == 'gbif_season' for j in jobs):,} training month-requests, "
          f"{sum(j[0] == 'gbif_season_eval' for j in jobs):,} evaluation month-requests", flush=True)

    failed = []

    def job(j):
        source, c, m, n = j
        since = str(reg["min_event_date"] if source == "gbif_season" else cfg["leakage"]["test_min_event_date"])
        try:
            return [{**r, "class_idx": c, "source": source} for r in fetch_month(int(key[c]), datasets, gadm, since, m, n, skip)]
        except Exception as e:                       # noqa: BLE001 - re-run fills gaps (requests are cached)
            failed.append((c, m, repr(e)[:120]))
            return []

    # evaluation requests first, so an observation picked for evaluation is never also a training row
    jobs.sort(key=lambda j: j[0] != "gbif_season_eval")
    with ThreadPoolExecutor(args.workers) as ex:
        rows = [r for rs in tqdm(ex.map(job, jobs), total=len(jobs), desc="month pulls") for r in rs]
    df = pd.DataFrame(rows)
    df = df.sort_values("source", key=lambda s: s != "gbif_season_eval").drop_duplicates("uuid")
    ev = set(df[df.source == "gbif_season_eval"].gbif_id)
    df = df[(df.source == "gbif_season_eval") | ~df.gbif_id.isin(ev)]
    bad = set(cfg["curation"]["drop_licenses"])
    df = df[df.license.notna() & ~df.license.isin(bad)]
    df["observation_id"] = "gbif:" + df.gbif_id
    names = {v["datasetKey"]: k for k, v in src.items() if isinstance(v, dict) and "datasetKey" in v}
    df["publisher"] = df.dataset_key.map(names)
    df["curation_tier"] = "community_verified"
    df["is_specimen"] = False
    out = root / "manifests/gbif_season.parquet"
    df.to_parquet(out, index=False)
    for s, d in df.groupby("source"):
        m = pd.to_datetime(d.drop_duplicates("gbif_id").event_date, errors="coerce").dt.month.value_counts().sort_index()
        print(f"{s}: {len(d):,} images / {d.gbif_id.nunique():,} observations / {d.class_idx.nunique()} species; by month {m.to_dict()}")
    if failed:
        print(f"warning: {len(failed)} requests failed (re-run): {failed[:3]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

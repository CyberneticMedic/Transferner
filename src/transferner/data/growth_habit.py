"""Growth habit per species from USDA PLANTS (D-005 source; D-037 focus chip).

For every label-set species: PlantSearch?searchText=<name> -> the USDA symbol of the exact name (else of a
GBIF synonym from species.parquet), then PlantProfile?symbol=<symbol> (following AcceptedId when the name is a
USDA synonym) -> GrowthHabits (Tree, Shrub, Subshrub, Vine, Forb/herb, Graminoid, Nonvascular, Lichenous),
Durations and the USDA common name. USDA PLANTS data are public domain; plantsservices has no robots.txt.
1 request/s (project rule), every response cached under $TRANSFERNER_DATA/cache/http, so re-runs are free.
Writes data/traits/growth_habit.csv (cited: one USDA profile URL per row).
  python -m transferner.data.growth_habit [--limit N]"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from ..utils.config import data_root
from ..utils.http import _cache_path, get_json

API = "https://plantsservices.sc.egov.usda.gov/api/"
PROFILE_URL = "https://plants.usda.gov/plant-profile/{symbol}"
_last = [0.0]


def fetch(url: str):
    """get_json with the project's 1 request/s rule for anything not already cached."""
    if not _cache_path(url, None).exists():
        wait = 1.0 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
    return get_json(url)


def plain(name: str) -> str:
    """'<i>Cardamine concatenata</i> (Michx.) Sw.' -> 'Cardamine concatenata' (italic part only)."""
    m = re.findall(r"<i>(.*?)</i>", name or "")
    return " ".join(x.strip() for x in m).replace("  ", " ") if m else (name or "").split(" (")[0]


def lookup(name: str):
    res = fetch(API + "PlantSearch?searchText=" + name.replace(" ", "%20")) or []
    for r in res:
        p = r.get("Plant") or {}
        if plain(p.get("ScientificName", "")).lower() == name.lower() and p.get("Rank") in ("Species", None):
            return p
    return None


def profile(symbol: str):
    p = fetch(API + "PlantProfile?symbol=" + symbol)
    if p and p.get("AcceptedId") and p.get("AcceptedId") != p.get("Id"):
        acc = fetch(API + f"PlantProfile/{p['AcceptedId']}")
        if acc:
            return acc, True
    return p, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/traits/growth_habit.csv")
    args = ap.parse_args()
    root = data_root()
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx")
    if args.limit:
        sp = sp.head(args.limit)
    rows, t0 = [], time.time()
    for k, r in enumerate(sp.itertuples()):
        hit, via = lookup(r.scientific_name), "name"
        if hit is None:
            syns = r.synonyms if r.synonyms is not None else []
            for s in list(syns)[:5]:
                s2 = " ".join(str(s).split()[:2])
                if s2.lower() != r.scientific_name.lower() and (hit := lookup(s2)) is not None:
                    via = f"synonym {s2}"
                    break
        row = {"scientific_name": r.scientific_name, "class_idx": int(r.class_idx), "usda_symbol": "", "match": "none",
               "growth_habits": "", "durations": "", "usda_common_name": "", "source_url": "",
               "accessed": time.strftime("%Y-%m-%d")}
        if hit is not None:
            p, followed = profile(hit["Symbol"])
            if p:
                row.update(usda_symbol=p.get("Symbol", ""), match=via + (" -> accepted" if followed else ""),
                           growth_habits=";".join(p.get("GrowthHabits") or []), durations=";".join(p.get("Durations") or []),
                           usda_common_name=p.get("CommonName") or "", source_url=PROFILE_URL.format(symbol=p.get("Symbol", "")))
        rows.append(row)
        if k % 200 == 0:
            got = sum(1 for x in rows if x["growth_habits"])
            print(f"{k + 1}/{len(sp)} species, {got} with growth habit ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[3] / args.out
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} species, {(df.growth_habits != '').sum()} with growth habit; matches "
          f"{df.match.str.split(' ').str[0].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()

"""Step 01. Regional species universe from GBIF.

Uses occurrence-search facets (no account needed): speciesKey counts for
gadm.level1Gid in region AND kingdomKey=Plantae AND occurrenceStatus=PRESENT,
once over the whole region and once per state x month for the priors table.
Facets give the same per-taxon counts a SPECIES_LIST download would, without
pulling 10^6 rows. (For a citable download DOI in the paper, run a SPECIES_LIST
download with the same predicate once GBIF credentials exist.)
Writes manifests/regional_species.parquet, manifests/priors_state_month.parquet."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..utils.config import load, data_root
from ..utils.http import GBIF, get_json

FACET_LIMIT = 50_000


def species_facets(**params) -> dict[int, int]:
    q = {"kingdomKey": 6, "occurrenceStatus": "PRESENT", "limit": 0,
         "facet": "speciesKey", "speciesKey.facetLimit": FACET_LIMIT, **params}
    r = get_json(f"{GBIF}/occurrence/search", q)
    counts = r["facets"][0]["counts"] if r["facets"] else []
    if len(counts) >= FACET_LIMIT:
        raise RuntimeError("facet limit reached; raise FACET_LIMIT")
    return {int(c["name"]): int(c["count"]) for c in counts}


def main() -> None:
    cfg = load("data")
    root = data_root()
    gids = cfg["region"]["gadm_level1"]
    names = dict(zip(gids, cfg["region"]["names"]))

    total = species_facets(gadmGid=gids)
    with_img = species_facets(gadmGid=gids, mediaType="StillImage")
    per_state = {g: species_facets(gadmGid=g) for g in gids}
    df = pd.DataFrame({"gbif_taxon_key": list(total), "n_regional": list(total.values())})
    df["n_regional_with_image"] = df.gbif_taxon_key.map(with_img).fillna(0).astype(int)
    for g in gids:
        df[f"n_{names[g].replace(' ', '_').lower()}"] = df.gbif_taxon_key.map(per_state[g]).fillna(0).astype(int)
    df = df.sort_values("n_regional", ascending=False).reset_index(drop=True)
    df.to_parquet(root / "manifests/regional_species.parquet", index=False)
    print(f"regional species: {len(df)}; >= {load('species')['min_regional_occurrences']}: "
          f"{(df.n_regional >= load('species')['min_regional_occurrences']).sum()}")

    if cfg["priors"]["record_state_month_counts"]:
        jobs = [(g, m) for g in gids for m in range(1, 13)]
        with ThreadPoolExecutor(6) as ex:
            res = list(ex.map(lambda gm: species_facets(gadmGid=gm[0], month=gm[1]), jobs))
        rows = [(k, names[g], m, n) for (g, m), d in zip(jobs, res) for k, n in d.items()]
        pri = pd.DataFrame(rows, columns=["gbif_taxon_key", "state", "month", "count"])
        pri.to_parquet(root / "manifests/priors_state_month.parquet", index=False)
        print(f"priors rows: {len(pri)}")


if __name__ == "__main__":
    main()

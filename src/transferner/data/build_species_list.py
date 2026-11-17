"""Step 02. Final label set.

label_set = {regional species with >= min_regional_occurrences}
          U must_include.txt U every species in lookalike_groups.csv
"Genus spp." entries expand to every regional species of that genus.
Names reconcile to accepted GBIF taxonKeys; genus/family are attached, and each
species' GBIF synonyms are kept so ToL rows filed under an older name still match.
Flags each species: has_traits, is_toxic, is_lookalike (drives the 2x cap).
Writes manifests/species.parquet (THE label table). Class indices are frozen:
re-running keeps every existing index and appends new species at the end."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..utils.config import REPO, load, data_root
from ..utils.http import GBIF, get_json

SPP_MIN_REGIONAL = 5   # floor for species pulled in by a "Genus spp." look-alike entry


def read_names(path) -> list[str]:
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def match(name: str) -> dict:
    m = get_json(f"{GBIF}/species/match", {"name": name, "kingdom": "Plantae", "strict": "true"})
    if m.get("matchType") == "NONE":
        raise SystemExit(f"GBIF has no match for {name!r}; fix the spelling in data/traits/")
    return m


def species_info(key: int) -> dict:
    s = get_json(f"{GBIF}/species/{key}")
    if s.get("taxonomicStatus") not in (None, "ACCEPTED", "DOUBTFUL") and s.get("acceptedKey"):
        s = get_json(f"{GBIF}/species/{s['acceptedKey']}")
    syn = get_json(f"{GBIF}/species/{s['key']}/synonyms", {"limit": 200}) or {"results": []}
    syn_names = sorted({r["canonicalName"] for r in syn["results"]
                        if r.get("canonicalName") and r.get("rank") == "SPECIES"} - {s.get("canonicalName")})
    return {"gbif_taxon_key": s["key"], "scientific_name": s.get("canonicalName"), "rank": s.get("rank"),
            "kingdom_key": s.get("kingdomKey"), "phylum_key": s.get("phylumKey"), "phylum": s.get("phylum"),
            "class": s.get("class"), "order": s.get("order"), "family": s.get("family"), "genus": s.get("genus"), "common_name": s.get("vernacularName"),
            "synonyms": syn_names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sp = load("species")
    root = data_root()
    sources = pd.read_json(root / "manifests/sources.json", typ="series")
    phylum_keys = set(sources["phylum_keys"].values())

    reg = pd.read_parquet(root / "manifests/regional_species.parquet")
    keys = set(reg.loc[reg.n_regional >= sp["min_regional_occurrences"], "gbif_taxon_key"])

    must = {match(n)["usageKey"] for n in read_names(REPO / sp["must_include"])}
    look = pd.read_csv(REPO / sp["lookalike_groups"])
    look_keys: dict[int, set[str]] = {}
    for _, row in look.iterrows():
        name = row.scientific_name.strip()
        if name.endswith(" spp."):
            g = match(name.removesuffix(" spp."))
            gkey = g.get("genusKey", g["usageKey"])
            # every regional species whose GBIF genusKey is this genus (responses are cached)
            cand = reg.loc[reg.n_regional >= SPP_MIN_REGIONAL, "gbif_taxon_key"].tolist()
            with ThreadPoolExecutor(args.workers) as ex:
                genus_of = dict(zip(cand, ex.map(lambda k: get_json(f"{GBIF}/species/{k}").get("genusKey"), cand)))
            hits = [k for k, gk in genus_of.items() if gk == gkey]
        else:
            hits = [match(name)["usageKey"]]
        for k in hits:
            look_keys.setdefault(k, set()).add(row.group_id)

    all_keys = sorted(keys | must | set(look_keys))
    with ThreadPoolExecutor(args.workers) as ex:
        infos = list(ex.map(species_info, all_keys))
    df = pd.DataFrame(infos).drop_duplicates("gbif_taxon_key")   # synonyms can collapse onto one accepted key
    df = df[(df["rank"] == "SPECIES") & (df.kingdom_key == 6) & df.phylum_key.isin(phylum_keys)]
    df = df.merge(reg[["gbif_taxon_key", "n_regional", "n_regional_with_image"]], on="gbif_taxon_key", how="left")
    df[["n_regional", "n_regional_with_image"]] = df[["n_regional", "n_regional_with_image"]].fillna(0).astype(int)

    # map look-alike / must-include keys through to their accepted key
    acc = dict(zip(all_keys, [i["gbif_taxon_key"] for i in infos]))
    must_acc = {acc[k] for k in must}
    look_acc: dict[int, set[str]] = {}
    for k, groups in look_keys.items():
        look_acc.setdefault(acc[k], set()).update(groups)

    traits = pd.read_csv(REPO / sp["traits"])
    toxic_names = set(traits.loc[(traits.contact_toxic == 1) | (traits.ingestion_toxic == 1), "scientific_name"])
    toxic_names |= set(look.loc[look.role == "toxic", "scientific_name"])
    df["must_include"] = df.gbif_taxon_key.isin(must_acc)
    df["lookalike_groups"] = df.gbif_taxon_key.map(lambda k: sorted(look_acc.get(k, [])))
    df["is_lookalike"] = df.lookalike_groups.str.len() > 0
    df["has_traits"] = df.scientific_name.isin(set(traits.scientific_name))
    df["is_toxic"] = df.scientific_name.isin(toxic_names)

    out = root / "manifests/species.parquet"
    if out.exists():                       # frozen class indices: keep old, append new
        old = pd.read_parquet(out)[["gbif_taxon_key", "class_idx"]]
        df = df.merge(old, on="gbif_taxon_key", how="left")
        start = int(old.class_idx.max()) + 1
        new = df.class_idx.isna()
        df.loc[new, "class_idx"] = range(start, start + int(new.sum()))
        dropped = set(old.gbif_taxon_key) - set(df.gbif_taxon_key)
        if dropped:
            print(f"warning: {len(dropped)} previously indexed species no longer qualify; their indices stay reserved")
    else:
        df = df.sort_values(["family", "genus", "scientific_name"]).reset_index(drop=True)
        df["class_idx"] = range(len(df))
    df["class_idx"] = df.class_idx.astype(int)
    df = df.sort_values("class_idx").reset_index(drop=True)
    df.drop(columns=["rank", "kingdom_key"]).to_parquet(out, index=False)
    print(f"label set: {len(df)} species | toxic {df.is_toxic.sum()} | look-alike {df.is_lookalike.sum()} "
          f"| must-include {df.must_include.sum()} | phyla {df.phylum.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

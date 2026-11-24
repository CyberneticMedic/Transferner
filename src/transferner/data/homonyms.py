"""GBIF later homonyms and misapplied names in the ToL matching (D-045; audit data_domain-02).

tol_select.py matches TreeOfLife rows to label species on bare canonical names, the accepted name plus every
GBIF synonym, and ToL rows carry no authorship. A synonym string that is also the ACCEPTED name of another
taxon (a later homonym, or a misapplied name) therefore pulls that other plant's photos into the class:
thimbleberry under Rubus cuneifolius, sloe under Prunus americana, wine grape under Vitis aestivalis, cape
gooseberry under Nicandra. The audit counted ~38.7K such train rows in ~300 classes, 56 classes at least half
contaminated, and the probe below zero-shot on them.

Rule (the audit's decisive test): a synonym is bad when GBIF's strict name match returns a taxon other than
the label species - ACCEPTED with a different key, or a synonym whose accepted taxon is a different key - or,
when GBIF knows the name only at a higher rank, when its genus is not the label species' genus. A synonym that
matches the label species (or one of its own synonyms) is kept. One cached metadata request per synonym name.
Writes manifests/homonym_synonyms.parquet (one row per synonym checked) and manifests/exclude_uuids.parquet
(uuid, class_idx, reason) for every candidate/downloaded row matched through a bad synonym; splits.py and the
trainers drop those rows.
  python -m transferner.data.homonyms [--workers 3]"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from ..utils.config import data_root
from ..utils.http import GBIF, get_json


def check(name: str) -> dict:
    r = get_json(f"{GBIF}/species/match", {"name": name, "strict": "true", "kingdom": "Plantae"}) or {}
    return {"match_type": r.get("matchType"), "status": r.get("status"), "usage_key": r.get("usageKey"),
            "accepted_key": r.get("acceptedUsageKey") or r.get("usageKey"), "matched_name": r.get("canonicalName"),
            "rank": r.get("rank")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3, help="GBIF throttles above ~3 concurrent requests")
    args = ap.parse_args()
    root = data_root()
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "gbif_taxon_key", "scientific_name", "synonyms"])
    key_of = dict(zip(sp.class_idx, sp.gbif_taxon_key))
    label = dict(zip(sp.class_idx, sp.scientific_name))
    cand = pd.read_parquet(root / "manifests/tol_candidates.parquet", columns=["uuid", "class_idx", "is_synonym", "scientific_name"])
    syn = cand[cand.is_synonym.astype(bool)]
    names = syn.groupby(["class_idx", "scientific_name"]).size().rename("n_candidates").reset_index()
    uniq = sorted(names.scientific_name.unique())
    with ThreadPoolExecutor(args.workers) as ex:
        res = dict(zip(uniq, tqdm(ex.map(check, uniq), total=len(uniq), desc="GBIF strict match")))
    rows = []
    for r in names.itertuples():
        m = res[r.scientific_name]
        own = int(key_of[r.class_idx])
        other = m["match_type"] == "EXACT" and m["accepted_key"] is not None and int(m["accepted_key"]) != own
        # GBIF does not know the name at species rank (HIGHERRANK): keep it only within the label genus, so a
        # misspelt look-alike genus cannot pass (Triplaris americana, a tree, under Triplasis americana, a grass)
        genus = r.scientific_name.split()[0]
        label_genus = label[r.class_idx].split()[0]
        higher = m["match_type"] == "HIGHERRANK" and genus != label_genus and (m["matched_name"] or "") != label_genus
        other = other or higher
        rows.append({"class_idx": r.class_idx, "label_species": sp.set_index("class_idx").scientific_name[r.class_idx],
                     "synonym": r.scientific_name, "n_candidates": r.n_candidates, **m,
                     "decision": "drop" if other else "keep"})
    df = pd.DataFrame(rows)
    df.to_parquet(root / "manifests/homonym_synonyms.parquet", index=False)
    bad = df[df.decision == "drop"][["class_idx", "synonym"]]
    drop = syn.merge(bad, left_on=["class_idx", "scientific_name"], right_on=["class_idx", "synonym"])[["uuid", "class_idx"]]
    drop["reason"] = "homonym_synonym"
    out = root / "manifests/exclude_uuids.parquet"
    if out.exists():                                   # keep exclusions for other reasons
        old = pd.read_parquet(out)
        drop = pd.concat([old[old.reason != "homonym_synonym"], drop], ignore_index=True)
    drop.drop_duplicates("uuid").to_parquet(out, index=False)
    img = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid"])
    spl = pd.read_parquet(root / "manifests/splits.parquet", columns=["uuid", "split"])
    used = spl[spl.uuid.isin(drop.uuid)].split.value_counts().to_dict()
    per = drop.groupby("class_idx").size()
    tv = spl[spl.split.isin(["train", "val"])].groupby(spl.loc[spl.split.isin(["train", "val"]), "uuid"].map(
        dict(zip(cand.uuid, cand.class_idx)))).size()
    share = (per / tv.reindex(per.index)).dropna()
    print(f"{len(uniq):,} synonym names checked; {int((df.decision == 'drop').sum()):,} (class, synonym) pairs are another "
          f"taxon's name; {len(drop):,} candidate rows excluded ({int(img.uuid.isin(drop.uuid).sum()):,} downloaded; by split {used})")
    print(f"classes affected: {per.size}; at least half of train+val contaminated: {int((share >= 0.5).sum())}")
    print(df[df.decision == "drop"].sort_values("n_candidates", ascending=False).head(25)[
        ["label_species", "synonym", "matched_name", "status", "n_candidates"]].to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

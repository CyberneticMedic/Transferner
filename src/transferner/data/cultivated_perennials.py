"""Cultivated perennials for the label set (D-044 user answer; audit findings capture_model-2,
data_domain-03, capture_open_set-1 and capture_open_set-5).

The label set takes a species at >= 20 regional GBIF records (configs/species.yaml), so planted
species that rarely escape (olive, pothos, sago palm, yew, citrus...) are missing even though a
yard walker meets them. The user asked for "some cultivated species, perennials only, no annual
crops". This module turns the hand-curated list data/extension/perennial_candidates.csv into

  data/extension/cultivated_perennials.csv           kept: absent from the label set, >= --min-tol images
  data/extension/cultivated_perennials_excluded.csv  everything else, with the reason

For every candidate it records:
  - the accepted GBIF name and key (species/match, strict, kingdom Plantae, then the accepted usage,
    exactly as build_species_list does) and the GBIF synonyms;
  - whether it is already a label species: same accepted key, or one of its names (queried or
    synonym) is a label species' accepted name AND GBIF resolves that name to this species. GBIF
    synonym lists keep misapplied and pro parte names (Buxus sempervirens under B. microphylla,
    Vitis labrusca under V. vinifera), so a shared name alone is reported, not taken as coverage;
  - usable TreeOfLife images: rows of raw/tol/plantae.parquet whose scientific_name is the accepted
    name or a synonym that GBIF resolves back to it (hybrid sign removed, as ToL writes them), under
    the same filters and curation tiers tol_select uses for training. capture_model-2 says to gate
    text rows on the count under the name BioCLIP actually saw, so the name with the most images is
    kept (tol_top_name);
  - perennial evidence from USDA PLANTS durations (growth_habit's lookup, 1 req/s, cached), the NCSU
    page's Life Cycle field and the curated habit;
  - the NC State Plant Toolbox page (raw/ncsu/sitemap.xml; species page by any name, else the genus
    page), fetched through traits_ncsu.fetch_pages at most once per second when not cached, and the
    poisonous parts / edible tags it lists.

Safety (D-038: false positives over false negatives): a blank or "not listed" toxic column only means
NCSU lists no poison section; it is never evidence that a plant is safe. "edible" in the group column
is a curation category (why a forager would look at the plant), not a claim that any part is safe to
eat. An edible-group species whose NCSU page lists poisonous parts becomes "both", and NCSU High
severity raises a species to priority 1.

  python -m transferner.data.cultivated_perennials [--min-tol 300] [--no-fetch]"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import duckdb
import pandas as pd

from ..utils.config import REPO, data_root, load
from ..utils.http import GBIF, get_json
from . import growth_habit, traits_ncsu
from .build_species_list import species_info
from .tol_select import TIER_SQL

EXT = REPO / "data/extension"
COLUMNS = ["scientific_name", "gbif_key", "common_name", "group", "perennial_evidence", "tol_images",
           "ncsu_slug", "ncsu_toxic_parts", "ncsu_edible_parts", "priority", "note", "tol_top_name"]
GROUP_ORDER = {"toxic": 0, "both": 1, "edible": 2}


def norm(name: str) -> str:
    """'Taxus × media' / 'Taxus x media' -> 'Taxus media' (ToL and GBIF canonical names drop the sign)."""
    s = re.sub(r"^(\S+)\s+x\s+(?=[a-z])", r"\1 ", name.replace("×", " "))
    return " ".join(s.split())


# ---------------------------------------------------------------- GBIF

def resolve(name: str) -> dict:
    """Strict GBIF match -> accepted species with synonyms; {'error': ...} when there is no species match."""
    m = get_json(f"{GBIF}/species/match", {"name": name, "kingdom": "Plantae", "strict": "true"}) or {}
    if m.get("matchType") in (None, "NONE") or m.get("rank") != "SPECIES":
        # GBIF writes hybrids as 'Taxus ×media'; retry that spelling before giving up
        alt = re.sub(r"^(\S+) × ", r"\1 ×", name)
        if alt != name:
            return resolve(alt)
        return {"error": f"no species match ({m.get('matchType')}, rank {m.get('rank')})"}
    info = species_info(m["usageKey"])
    rolled = ""
    if info.get("rank") in ("SUBSPECIES", "VARIETY", "FORM"):
        # e.g. Actinidia deliciosa -> A. chinensis var. deliciosa: the label set is species-rank
        sk = (get_json(f"{GBIF}/species/{info['gbif_taxon_key']}") or {}).get("speciesKey")
        if sk:
            rolled = f"GBIF accepts it as a {info['rank'].lower()} of its species"
            info = species_info(sk)
    if info.get("rank") != "SPECIES":
        return {"error": f"accepted usage is rank {info.get('rank')}"}
    info["match_status"] = m.get("status")
    info["rolled_up"] = rolled
    return info


def resolves_to(name: str, key: int) -> bool:
    """Does another name from a candidate's GBIF synonym list really stand for this species?

    GBIF synonym lists keep pro parte and misapplied names whose canonical form is also another
    accepted species: Chrysanthemum indicum under C. morifolium, Buxus sempervirens under
    B. microphylla, Vitis labrusca under V. vinifera, Citrus bergamia under C. limon. ToL files
    images, NCSU pages and the label set use the canonical name, so such a name belongs to the
    other species. A name counts when a strict GBIF match resolves it to `key`, or when GBIF has
    no species usage for it at all."""
    m = get_json(f"{GBIF}/species/match", {"name": name, "kingdom": "Plantae", "strict": "true"}) or {}
    if m.get("matchType") in (None, "NONE") or m.get("rank") != "SPECIES":
        return True
    return (m.get("acceptedUsageKey") or m.get("usageKey")) == key


# ---------------------------------------------------------------- ToL


def tol_counts(names: set[str]) -> pd.DataFrame:
    """Usable ToL images per scientific_name (tol_select's filters and allowed tiers), one DuckDB scan.

    Cached under $TRANSFERNER_DATA/cache/extension/ so re-runs only scan for new names."""
    cache = data_root() / "cache/extension/tol_name_counts.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(
        columns=["name", "tol_images", "tol_field_images"])
    todo = sorted(names - set(have.name))
    if todo:
        cur = load("data")["curation"]
        tiers = ",".join(f"'{t}'" for t in cur["allowed_tiers"])
        bad_lic = ",".join(f"'{x}'" for x in cur["drop_licenses"])
        c = duckdb.connect()
        c.execute("SET threads=8; SET memory_limit='8GB'")      # shared box: a training run is on
        c.register("nm", pd.DataFrame({"name": todo}))
        q = f"""
        SELECT nm.name, count(*) AS tol_images,
               count(*) FILTER (WHERE curation_tier <> 'institutional_specimen') AS tol_field_images
        FROM (SELECT scientific_name, {TIER_SQL} AS curation_tier
              FROM '{data_root() / "raw/tol/plantae.parquet"}' t
              WHERE coalesce(basis_of_record, '') NOT IN ('MACHINE_OBSERVATION', 'FOSSIL_SPECIMEN')
                AND (img_type IS NULL OR img_type NOT LIKE 'Museum Specimen:%' OR img_type = 'Museum Specimen: Plant')
                AND license_name IS NOT NULL AND license_name NOT IN ({bad_lic})
                AND source_url IS NOT NULL) t
        JOIN nm ON t.scientific_name = nm.name
        WHERE curation_tier IN ({tiers})
        GROUP BY nm.name"""
        got = c.execute(q).df()
        got = pd.DataFrame({"name": todo}).merge(got, on="name", how="left").fillna(0)
        have = pd.concat([have, got], ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(cache, index=False)
    have[["tol_images", "tol_field_images"]] = have[["tol_images", "tol_field_images"]].astype(int)
    return have


# ---------------------------------------------------------------- NCSU

def plant_slugs() -> set[str]:
    """Slugs of /plants/<slug>/ pages only. traits_ncsu.sitemap_slugs() takes the last path part of
    every sitemap URL, so genus names that only have a non-plant page there ('ligustrum',
    'elaeagnus') look like pages and fetch as 404."""
    traits_ncsu.sitemap_slugs()                          # downloads the sitemap if it is not cached
    text = (traits_ncsu.out_dir() / "sitemap.xml").read_text()
    return set(re.findall(r"<loc>[^<]*/plants/([^/<]+)/?</loc>", text))


def slug_variants(name: str) -> list[str]:
    """NCSU writes hybrids as genus-x-epithet (taxus-x-media, citrus-x-limon)."""
    s = traits_ncsu.slug(name)
    parts = s.split("-", 1)
    return [s] + ([f"{parts[0]}-x-{parts[1]}"] if len(parts) == 2 else [])


def ncsu_page(names: list[str], genus: str, slugs: set[str]) -> tuple[str, str]:
    """(slug, level): the species page under any of the names, else the genus page, else ('', '')."""
    for n in names:
        for s in slug_variants(n):
            if s in slugs:
                return s, "species"
    if genus and traits_ncsu.slug(genus) in slugs:
        return traits_ncsu.slug(genus), "genus"
    return "", ""


def ncsu_fields(slug_: str) -> dict | None:
    p = traits_ncsu.out_dir() / "pages" / f"{slug_}.html"
    if not slug_ or not p.exists():
        return None
    text = p.read_text()
    f = traits_ncsu.parse_page(text)
    f["life_cycle"] = traits_ncsu._dd_after(text, "Life Cycle:")
    return f


def toxic_text(f: dict | None) -> str:
    if f is None:
        return ""
    if not f["has_poison_section"]:
        return "not listed"
    s = f"{f['poison_severity'] or 'severity not given'}: {', '.join(f['poison_parts']) or 'parts not given'}"
    if f["contact_dermatitis"] == "Yes":
        s += "; contact dermatitis"
    return s


def edible_text(f: dict | None) -> str:
    """What the page lists, quoted, never more: edible tags, the Edible plant type, the Edibility text.

    When the same page has a poison section its severity goes first, so a quote such as the yew
    page's "the aril is OK to eat" (the seed inside is deadly) is never read without it (D-038)."""
    if f is None:
        return ""
    bits = sorted({t for t in f["tags"] if re.match(r"edible\b", t)})
    if "Edible" in f["plant_type"]:
        bits.append("plant type Edible")
    if f["edibility"]:
        e = " ".join(f["edibility"].split())
        bits.append(f"Edibility: \"{e[:110]}{'...' if len(e) > 110 else ''}\"")
    if not bits:
        return "not listed"
    if f["has_poison_section"]:
        bits.insert(0, f"SAME PAGE LISTS POISONOUS PARTS ({f['poison_severity'] or 'severity not given'})")
    return "; ".join(bits)


# ---------------------------------------------------------------- USDA

def usda(name: str) -> dict:
    try:
        hit = growth_habit.lookup(name)
        if hit is None:
            return {}
        p, _ = growth_habit.profile(hit["Symbol"])
    except Exception as e:                               # noqa: BLE001 - evidence only; the curated habit stays
        print(f"USDA lookup failed for {name}: {type(e).__name__}", flush=True)
        return {}
    if not p:
        return {}
    return {"symbol": p.get("Symbol", ""), "durations": p.get("Durations") or [],
            "habits": p.get("GrowthHabits") or []}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-tol", type=int, default=300, help="usable ToL images needed to keep a species")
    ap.add_argument("--no-fetch", action="store_true", help="NCSU: cached pages only")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = data_root()

    cand = pd.read_csv(EXT / "perennial_candidates.csv")
    sp = pd.read_parquet(root / "manifests/species.parquet")
    reg = pd.read_parquet(root / "manifests/regional_species.parquet")[["gbif_taxon_key", "n_regional"]]
    n_reg = dict(zip(reg.gbif_taxon_key, reg.n_regional))
    label_keys = dict(zip(sp.gbif_taxon_key, sp.scientific_name))
    label_names = {norm(n): n for n in sp.scientific_name}
    label_syn = {}                                   # synonym -> label species that lists it
    for r in sp.itertuples():
        for s in r.synonyms:
            label_syn.setdefault(norm(s), []).append(r.scientific_name)
    label_genera = set(sp.genus.dropna())
    hp = root / "manifests/homonym_synonyms.parquet"           # D-045: label synonyms tol_select ignores
    dropped = set()
    if hp.exists():
        h = pd.read_parquet(hp, columns=["label_species", "synonym", "decision"])
        dropped = set(zip(h.label_species[h.decision == "drop"], h.synonym[h.decision == "drop"]))

    with ThreadPoolExecutor(args.workers) as ex:
        infos = list(ex.map(resolve, cand.query_name))

    rows, seen = [], {}
    for c, info in zip(cand.itertuples(), infos):
        row = {"query_name": c.query_name, "common_name": c.common_name, "group": c.group, "kind": c.kind,
               "habit": c.habit, "priority": int(c.priority), "curated_note": c.note, "notes": [], "excluded": []}
        if "error" in info:
            row.update(scientific_name=norm(c.query_name), gbif_key="", names=[norm(c.query_name)], genus="")
            row["excluded"].append(f"GBIF: {info['error']}")
            rows.append(row)
            continue
        key, acc = int(info["gbif_taxon_key"]), info["scientific_name"]
        if key in seen:                                  # two queries collapse onto one accepted species
            seen[key]["notes"].append(f"also queried as {c.query_name}")
            continue
        names = list(dict.fromkeys([norm(acc), norm(c.query_name)] + [norm(s) for s in info["synonyms"]]))
        row.update(scientific_name=acc, gbif_key=key, names=names, genus=info.get("genus") or "")
        if norm(c.query_name) != norm(acc):
            row["notes"].append(f"GBIF accepted name for {c.query_name}")
        if info["rolled_up"]:
            row["notes"].append(info["rolled_up"])
        # already in the label set? A shared name only counts when GBIF resolves it to this species
        if key in label_keys:
            row["excluded"].append(f"in label set as {label_keys[key]}")
        else:
            shared = [n for n in names if n in label_names]
            hit = next((label_names[n] for n in shared if resolves_to(label_names[n], key)), None)
            if hit:
                row["excluded"].append(f"in label set as {hit} (GBIF resolves that name to this species)")
            elif shared:
                row["notes"].append("GBIF synonym list also holds label species "
                                    + ", ".join(label_names[n] for n in shared) + " (a different species in GBIF)")
        claimed = sorted(set(label_syn.get(norm(acc), [])))
        if claimed and not row["excluded"]:
            done = all((s, acc) in dropped for s in claimed)
            row["notes"].append(f"listed as a GBIF synonym of label species {', '.join(claimed)}; "
                                + ("D-045 homonym_synonyms.parquet already drops it there" if done else
                                   "tol_select would file its ToL images there: drop that synonym"))
        if row["genus"] and row["genus"] not in label_genera:
            row["notes"].append("genus absent from the label set")
        if key in n_reg:
            row["notes"].append(f"regional GBIF records: {n_reg[key]}")
        seen[key] = row
        rows.append(row)

    # The names that stand for each species in ToL and on NCSU: the accepted name plus synonyms that
    # GBIF resolves back to it (resolves_to; only checked for names with ToL images or an NCSU page).
    # A label species' accepted name never counts for a species outside the set.
    all_names = {n for r in rows for n in r["names"]}
    tol = tol_counts(all_names).set_index("name")
    slugs = plant_slugs()
    for r in rows:
        acc = norm(r["scientific_name"])
        own = [n for n in r["names"] if n == acc or r["excluded"] or n not in label_names]
        if r["gbif_key"] != "":
            used = [n for n in own if n != acc and (tol.tol_images.get(n, 0) > 0
                                                    or any(s in slugs for s in slug_variants(n)))]
            foreign = [n for n in used if not resolves_to(n, r["gbif_key"])]
            if foreign:
                r["notes"].append("ToL images / NCSU pages under " + ", ".join(foreign)
                                  + " not counted (GBIF: another species)")
            own = [n for n in own if n not in foreign]
        r["own_names"] = own
        per = {n: int(tol.tol_images.get(n, 0)) for n in own}
        r["tol_images"] = sum(per.values())
        r["tol_field_images"] = sum(int(tol.tol_field_images.get(n, 0)) for n in own)
        r["tol_top_name"] = max(per, key=per.get) if per and max(per.values()) > 0 else ""
        if r["tol_top_name"] and r["tol_top_name"] != norm(r["scientific_name"]):
            r["notes"].append(f"ToL files most images under {r['tol_top_name']} ({per[r['tol_top_name']]})")
        # tol_select caps specimens at 30% of a species' quota and the detector never trains on them
        # (D-020), so a later image pull needs field photos
        if r["tol_images"] and r["tol_field_images"] < args.min_tol:
            r["notes"].append(f"only {r['tol_field_images']} ToL field photos, the rest herbarium specimens")

    # NCSU: species page under any of its own names, else the genus page; fetch missing ones politely
    for r in rows:
        r["ncsu_slug"], r["ncsu_level"] = ncsu_page(r["own_names"], r["genus"], slugs)
    want = sorted({r["ncsu_slug"] for r in rows if r["ncsu_slug"] and not r["excluded"]})
    if not args.no_fetch:
        traits_ncsu.fetch_pages(want, min_gap=1.0)

    # USDA durations (1 req/s for anything not cached), only for species still in the running
    for r in rows:
        u = usda(r["scientific_name"]) if not r["excluded"] and r["kind"] != "annual_crop" else {}
        f = ncsu_fields(r["ncsu_slug"])
        if r["ncsu_slug"] and f is None:
            r["notes"].append("NCSU page not cached")
        pre = "[genus page] " if r["ncsu_level"] == "genus" else ""
        r["ncsu_toxic_parts"] = pre + toxic_text(f) if f is not None else ""
        r["ncsu_edible_parts"] = pre + edible_text(f) if f is not None else ""
        ev = []
        if u.get("durations"):
            ev.append(f"USDA {u['symbol']}: {'/'.join(u['durations'])}"
                      + (f" ({'/'.join(u['habits'])})" if u.get("habits") else ""))
        if f is not None and r["ncsu_level"] == "species" and f["life_cycle"]:
            ev.append(f"NCSU life cycle: {'/'.join(f['life_cycle'])}")
        ev.append(f"curated: {r['habit']}")
        r["perennial_evidence"] = "; ".join(ev)
        durs = set(u.get("durations") or []) | set(f["life_cycle"] if f is not None and r["ncsu_level"] == "species" else [])
        if durs and not durs & {"Perennial", "Woody", "Biennial", "Bulb"}:     # NCSU files bulbs as "Bulb"
            r["excluded"].append(f"sources list it as {'/'.join(sorted(durs))} only")
        # safety-first group and priority (D-038)
        if f is not None and r["ncsu_level"] == "species":
            if r["group"] == "edible" and f["has_poison_section"]:
                r["group"] = "both"
                r["notes"].append("NCSU lists poisonous parts")
            if r["group"] == "toxic" and ("Edible" in f["plant_type"] or any(t.startswith("edible") for t in f["tags"])):
                r["group"] = "both"
                r["notes"].append("NCSU tags an edible use")
            if r["group"] != "edible" and "High" in f["poison_severity"]:
                r["priority"] = 1
        if r["kind"] == "houseplant":
            r["excluded"].append("houseplant, rarely outdoors (lead: only if commonly outdoors)")
        if r["kind"] == "annual_crop":
            r["excluded"].append("annual crop (user: perennials only)")
        if r["kind"] == "tender":
            r["excluded"].append("not hardy in zones 6-9 (grown as an annual or in containers here)")
        if not r["excluded"] and r["tol_images"] < args.min_tol:
            r["excluded"].append(f"ToL images {r['tol_images']} < {args.min_tol}")

    out = pd.DataFrame(rows)
    out["note"] = [("; ".join([r["curated_note"]] + r["notes"])) for r in rows]
    out["gbif_key"] = out.gbif_key.astype(str)
    out["_g"] = out.group.map(GROUP_ORDER)
    out = out.sort_values(["priority", "_g", "tol_images"], ascending=[True, True, False])
    kept = out[out.excluded.str.len() == 0]
    exc = out[out.excluded.str.len() > 0].copy()
    exc["excluded_reason"] = exc.excluded.str.join("; ")
    kept[COLUMNS].to_csv(EXT / "cultivated_perennials.csv", index=False)
    exc[COLUMNS + ["excluded_reason"]].to_csv(EXT / "cultivated_perennials_excluded.csv", index=False)
    why = exc.excluded.str[0].str.extract(r"^(in label set|ToL images|houseplant|annual crop|not hardy|GBIF|sources)")[0]
    print(f"candidates {len(cand)} -> {len(out)} species | kept {len(kept)} "
          f"(priority {kept.priority.value_counts().sort_index().to_dict()}, group {kept.group.value_counts().to_dict()}) "
          f"| excluded {len(exc)}: {why.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

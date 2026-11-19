"""Phase C trait source: NC State Extension Gardener Plant Toolbox (plants.ces.ncsu.edu).

  python -m transferner.data.traits_ncsu fetch    # sitemap -> label-set pages, 1 req/s, cached
  python -m transferner.data.traits_ncsu parse    # cached pages -> raw/ncsu/ncsu_fields.parquet

The site has no robots.txt (404, checked 2026-11-19) and no terms page restricting automated
access. Pages are cached under $TRANSFERNER_DATA/raw/ncsu/pages/ and fetched at most once per
second with the project User-Agent. Only factual fields are kept (poison severity, poisonous
parts, contact dermatitis, edible plant type/tags), each with its page URL as the citation.

A species matches a page when its accepted name or one of its GBIF synonyms slugifies to a
sitemap slug ("Phytolacca americana" -> phytolacca-americana). Cultivar pages are not used.
traits_build turns the parsed fields into data/traits/species_traits.csv rows."""
from __future__ import annotations

import argparse
import html
import re
import time
from datetime import date

import pandas as pd

from ..utils.config import data_root
from ..utils.http import session

BASE = "https://plants.ces.ncsu.edu"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def out_dir():
    d = data_root() / "raw/ncsu"
    (d / "pages").mkdir(parents=True, exist_ok=True)
    return d


def sitemap_slugs() -> set[str]:
    p = out_dir() / "sitemap.xml"
    if not p.exists():
        r = session().get(f"{BASE}/sitemap.xml", timeout=60)
        r.raise_for_status()
        p.write_text(r.text)
    return {u.rstrip("/").split("/")[-1] for u in re.findall(r"<loc>([^<]*)</loc>", p.read_text())}


def targets() -> pd.DataFrame:
    """Label-set species with a toolbox page, in Phase C priority order."""
    sp = pd.read_parquet(data_root() / "manifests/species.parquet")
    slugs = sitemap_slugs()
    rows = []
    for r in sp.itertuples():
        names = [r.scientific_name] + [s for s in r.synonyms if s]
        for n in names:
            if slug(n) in slugs:
                rows.append({"class_idx": r.class_idx, "scientific_name": r.scientific_name,
                             "gbif_taxon_key": r.gbif_taxon_key, "page_name": n, "slug": slug(n),
                             "priority": int(r.must_include or r.is_lookalike or r.is_toxic),
                             "n_regional": r.n_regional})
                break
    t = pd.DataFrame(rows)
    return t.sort_values(["priority", "n_regional"], ascending=False).reset_index(drop=True)


def fetch(min_gap: float = 1.0) -> None:
    t = targets()
    d = out_dir()
    t.to_parquet(d / "targets.parquet", index=False)
    log_p = d / "fetch_log.tsv"
    s = session()
    last, n_new = 0.0, 0
    for i, r in enumerate(t.itertuples()):
        p = d / "pages" / f"{r.slug}.html"
        if p.exists():
            continue
        wait = min_gap - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.monotonic()
        url = f"{BASE}/plants/{r.slug}/"
        try:
            resp = s.get(url, timeout=60)
            code = resp.status_code
            if code == 200:
                tmp = p.with_suffix(".tmp")
                tmp.write_text(resp.text)
                tmp.replace(p)
                n_new += 1
        except Exception as e:                           # noqa: BLE001 - log and move on
            code = f"error:{type(e).__name__}"
        with open(log_p, "a") as f:
            f.write(f"{date.today()}\t{r.slug}\t{code}\n")
        if i % 100 == 0:
            print(f"{i}/{len(t)} {r.slug} {code} (new {n_new})", flush=True)
    print(f"done: {len(t)} targets, {n_new} new pages", flush=True)


# ---------------------------------------------------------------- parse

def _dd_after(block: str, label: str) -> list[str]:
    m = re.search(rf"<dt>\s*{re.escape(label)}\s*</dt>(.*?)(?=<dt|</dl>)", block, re.S)
    if not m:
        return []
    vals = re.findall(r'<span class="detail_display_attribute">(.*?)</span>', m.group(1), re.S)
    return [html.unescape(re.sub(r"<[^>]+>", "", v)).strip() for v in vals]


def parse_page(text: str) -> dict:
    poison = re.search(r'Poisonous to Humans:</span>(.*?)</dl>', text, re.S)
    pb = poison.group(1) + "</dl>" if poison else ""         # keep the terminator for the last <dt>
    tags = list(dict.fromkeys(html.unescape(x).strip().lstrip("#")       # page renders tags twice
                              for x in re.findall(r'href="/find_a_plant/\?tag=[^"]*"[^>]*>([^<]*)<', text)))
    return {
        "has_poison_section": bool(poison),
        "poison_severity": "; ".join(_dd_after(pb, "Poison Severity:")),
        "poison_symptoms": " ".join(_dd_after(pb, "Poison Symptoms:")),
        "poison_principle": " ".join(_dd_after(pb, "Poison Toxic Principle:")),
        "contact_dermatitis": "; ".join(_dd_after(pb, "Causes Contact Dermatitis:")),
        "poison_parts": _dd_after(pb, "Poison Part:"),
        "plant_type": _dd_after(text, "Plant Type:"),
        "edibility": " ".join(_dd_after(text, "Edibility:")),
        "uses": " ".join(_dd_after(text, "Uses (Ethnobotany):")),
        "tags": tags,
        "common_names": [html.unescape(x).strip() for x in re.findall(
            r'<a href="/plants/[^"]*/common-name/[^"]*">([^<]*)</a>', text)],
    }


def parse() -> pd.DataFrame:
    d = out_dir()
    t = pd.read_parquet(d / "targets.parquet")
    rows = []
    for r in t.itertuples():
        p = d / "pages" / f"{r.slug}.html"
        if not p.exists():
            continue
        f = parse_page(p.read_text())
        rows.append({"class_idx": r.class_idx, "scientific_name": r.scientific_name, "gbif_taxon_key": r.gbif_taxon_key,
                     "page_name": r.page_name, "url": f"{BASE}/plants/{r.slug}/",
                     "accessed": date.fromtimestamp(p.stat().st_mtime).isoformat(), **f})
    df = pd.DataFrame(rows)
    df.to_parquet(d / "ncsu_fields.parquet", index=False)
    print(f"parsed {len(df)} pages | poison section {df.has_poison_section.sum()} | "
          f"contact dermatitis yes {(df.contact_dermatitis == 'Yes').sum()}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "parse"])
    args = ap.parse_args()
    fetch() if args.step == "fetch" else parse()


if __name__ == "__main__":
    main()

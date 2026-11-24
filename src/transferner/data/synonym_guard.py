"""Which GBIF synonyms of a label species may stand in for it when a source is matched by name.

Read-only audit 2026-11-24 ($TRANSFERNER_DATA/ops/audit-20261124.json): findings data_domain-02 and
errors_found "Trait rows are matched through synonyms and lose their part information".

species.parquet keeps the bare canonical name of every SPECIES-rank GBIF synonym (build_species_list.species_info),
without authorship. A source keyed by bare name (an NC State page slug, a USDA PLANTS search) cannot tell a true
synonym from another plant that happens to carry the same name:
  (a) the synonym is the accepted name of ANOTHER label species: Trautvetteria carolinensis lists
      "Hydrastis canadensis", so it took goldenseal's toxic rows; Solidago tortifolia took S. odora's edible row;
      Plantago rugelii took P. major's page. tol_select.py already drops these for ToL rows; the trait and
      growth-habit matchers did not. Always rejected.
  (b) the synonym is a later homonym or a misapplied name (Ammi majus Walter under Ptilimnium capillaceum): GBIF's
      strict match on the bare name returns a different ACCEPTED species (Ammi majus L.), and that is the plant a
      name-keyed source describes. Rejected unless the source record itself confirms the label species.
GBIF's synonym status is not a usable filter (Illicium anisatum Bartram is a HOMOTYPIC_SYNONYM, yet the bare name is
the accepted Japanese star anise; audit checker notes), so (b) tests the strict match of the bare name itself.

The strict match is a proxy for what the source means. When the source says so directly (USDA's record carries the
same authorship as the label's own GBIF synonym record, or USDA's accepted name is the label species), the caller
passes source_confirms=True and (b) is skipped: GBIF's strict match can pick an obscure later homonym it keeps
ACCEPTED (Draba brachycarpa (DC.) Baikov, 2005 for USDA's Draba brachycarpa Nutt. ex Torr. & A. Gray, which is
Abdra brachycarpa). Rule (a) has no override.

    guard = SynonymGuard(species_df)
    why = guard.reject(label_key, label_name, synonym)    # "" when the synonym may be used, else the reason

Strict matches go through utils.http.get_json with the same parameters as build_species_list.match, so they share
its on-disk cache; uncached ones are spaced 0.2 s apart (GBIF metadata API)."""
from __future__ import annotations

import re
import time

import pandas as pd

from ..utils.http import GBIF, _cache_path, get_json

MATCH_URL = f"{GBIF}/species/match"
_last = [0.0]


def strict_match(name: str) -> dict:
    """GBIF backbone strict match of a bare canonical name in Plantae (cached; {} when GBIF returns nothing)."""
    params = {"name": name, "kingdom": "Plantae", "strict": "true"}     # same key as build_species_list.match
    if not _cache_path(MATCH_URL, params).exists():
        wait = 0.2 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
    return get_json(MATCH_URL, params) or {}


def norm_author(a: str) -> str:
    """'Nutt. ex Torr. & A. Gray' and 'Nutt. ex Torr. & A.Gray' -> the same key (case, spaces, dots ignored)."""
    return re.sub(r"[\s.]", "", (a or "").lower())


def synonym_authorships(label_key: int, synonym: str) -> set[str]:
    """Normalised authorships of the label species' own GBIF synonym records with this canonical name. Same
    request as build_species_list.species_info, so it is served from the cache."""
    syn = get_json(f"{GBIF}/species/{int(label_key)}/synonyms", {"limit": 200}) or {"results": []}
    return {norm_author(r.get("authorship", "")) for r in syn["results"] if r.get("canonicalName") == synonym}


class SynonymGuard:
    def __init__(self, sp: pd.DataFrame):
        self.accepted = set(sp.scientific_name)
        self.log: list[dict] = []          # every rejection, for the caller's report

    def reject(self, label_key: int, label_name: str, synonym: str, source_confirms: bool = False) -> str:
        """Reason this synonym must not stand in for the label species, or "" when it may."""
        why = ""
        if synonym in self.accepted and synonym != label_name:
            why = "accepted name of another label species"
        elif not source_confirms:
            m = strict_match(synonym)
            # Only a species-level EXACT match counts: for an ambiguous name GBIF answers HIGHERRANK with the
            # ACCEPTED genus or family ("Asarum virginicum" -> Aristolochiaceae), which says nothing about the
            # species. SYNONYM, DOUBTFUL or no match keep the synonym too: weaker signals, and the synonym came
            # from the label species' own GBIF synonym list.
            if (m.get("status") == "ACCEPTED" and m.get("matchType") == "EXACT" and m.get("rank") == "SPECIES"
                    and m.get("usageKey") and int(m["usageKey"]) != int(label_key)):
                why = f"GBIF strict match is ACCEPTED {m.get('scientificName', synonym)} (key {m['usageKey']})"
        if why:
            self.log.append({"gbif_taxon_key": int(label_key), "scientific_name": label_name, "synonym": synonym,
                             "reason": why})
        return why

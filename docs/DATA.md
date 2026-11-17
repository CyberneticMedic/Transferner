# Data

## Sources
- **GBIF regional occurrences** (SC/NC/GA/TN, all publishers): species universe + state×month priors.
- **Pl@ntNet via GBIF** (DOI 10.15468/gtebaa): images with organ tags (title suffix, e.g. `: leaf`).
- **iNaturalist research grade via GBIF** (DOI 10.15468/ab3s5x): together with Pl@ntNet, the recent
  (eventDate ≥ 2024-06-01) regional photos that form the leakage-free test pool (D-022).
- **TreeOfLife-200M** (imageomics/TreeOfLife-200M, CC0 catalog): `dataset/catalog.parquet`
  (uuid, source_id, source_url, publisher, basis_of_record, img_type, full taxonomy). Per-image licenses come from the original host.

## Manifests (`$TRANSFERNER_DATA/manifests/*.parquet`)
| File | Key | Contents |
|---|---|---|
| sources.json | | resolved dataset keys, GADM ids, phylum keys, ToL / BioCLIP commit hashes |
| regional_species.parquet | gbif_taxon_key | regional occurrence counts (total, with image, per state) |
| tol_candidates.parquet | uuid | ToL rows picked per species (curation_tier, license, pick_rank) |
| gbif_candidates.parquet | uuid | recent regional Pl@ntNet / iNaturalist photos (event_date, state, organ) |
| species.parquet | class_idx | accepted name, gbif_taxon_key, genus, family, is_toxic, is_lookalike, has_traits, n_regional |
| images.parquet | uuid | source, source_id/gbifID, observation_id, path, organ, eventDate, state, license, rights_holder, phash, in_tol |
| splits.parquet | uuid | split ∈ train/val/test |
| boxes.parquet | uuid, box_idx | x0,y0,x1,y1 (normalised), score, genus_idx |
| priors_state_month.parquet | taxon_key, state, month | count |

## Leakage control
BioCLIP 2.5 trained on ToL-200M **plus ~19M images not in the public catalog**, so test records must:
1. have a gbifID not matching any ToL `source_id`,
2. be phash-distinct (Hamming > 6) from every ToL image of the same species, and
3. have eventDate after BioCLIP 2.5's training-data cutoff (to confirm with Imageomics/paper).

## Curation (D-021)
Every image has `curation_tier` ∈ institutional_specimen · eol_curated · community_verified.
Licenses that do not permit ML training (all-rights-reserved, other, unknown) are dropped at selection.

## Caps
500 images/species, 1000 for toxic or look-alike species. Species with < 20 images are kept for
BioCLIP zero-shot only (no detector class / no probe weight training).

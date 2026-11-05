# Data

## Sources
- **GBIF regional occurrences** (SC/NC/GA/TN, all publishers): species universe + state×month priors.
- **Pl@ntNet via GBIF** (DOI 10.15468/gtebaa): images with organ tags. Main source of the leakage-free test set.
- **TreeOfLife-200M** (imageomics/TreeOfLife-200M, CC0 catalog): `dataset/catalog.parquet`
  (uuid, source_id, source_url, publisher, basis_of_record, img_type, full taxonomy). Per-image licenses come from the original host.

## Manifests (`$TRANSFERNER_DATA/manifests/*.parquet`)
| File | Key | Contents |
|---|---|---|
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

## Caps
500 images/species, 1000 for toxic or look-alike species. Species with < 20 images are kept for
BioCLIP zero-shot only (no detector class / no probe weight training).

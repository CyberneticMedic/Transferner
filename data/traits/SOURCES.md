# Trait sources (edibility / toxicity / utility)

Every row in `species_traits.csv` must cite a `source_url`. Reputable sources, in priority order:

1. USDA PLANTS Database (plants.usda.gov): accepted names, characteristics, Plant Guides
2. FDA Poisonous Plant Database (PLANTOX)
3. NC State Extension Plant Toolbox: "Causes Contact Dermatitis", poison severity, symptoms
4. Clemson Cooperative Extension / Home & Garden Information Center (HGIC)
5. UGA Extension, UT Extension
6. Peer-reviewed literature / poison-center publications for look-alike confusions

Conflict rule: if sources disagree on toxicity, the **more toxic** label wins and the
disagreement is recorded in `notes`.

## Column semantics
- `part`: whole | leaf | stem | root | flower | fruit | seed | sap | shoot | bark
- `category`: edible | utility | toxic | conditional. One row per (species, part, category);
  a species may have several rows (multi-label, D-004).
- `contact_toxic`, `ingestion_toxic`: 0/1, independent flags.
- `ingestion_severity`: none | low | moderate | high | lethal (NC State scale where available).
- `conditions`: free text for conditional rows (preparation, growth stage, ripeness).

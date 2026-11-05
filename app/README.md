# Android app (Kotlin) — placeholder

Model I/O contract (stable interface between `export/bundle.py` and the app):

| Artifact | Input | Output |
|---|---|---|
| detector.tflite | uint8/int8 [1,320,320,3] RGB | boxes [1,10,4] (y0,x0,y1,x1 norm), genus_scores [1,10,G] |
| bioclip_<variant>.tflite | float16 [1,3,224,224], CLIP mean/std | embedding [1,1024] L2-normalised |
| head.bin + T | embedding | logits = emb·Wᵀ / T → softmax → top-5 |
| labels.json | — | class_idx → {scientific, common, genus, family, lookalike_group} |
| genus_family.json | — | genus_idx → family_idx (rollup) |
| traits.json | — | species → [{part, category, contact, ingestion, conditions, source}] |
| thumbs/ + ATTRIBUTION.json | — | 3–5 licensed reference images per species |
| manifest.json | — | versions + sha256 of all of the above; per-device variant table |

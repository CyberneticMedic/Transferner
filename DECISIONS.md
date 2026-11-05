# Decision log
Newest at the bottom. Format: ID · date · decision · why.

- **D-001 · 2026-11-05 · Research focus:** the most efficient way to run BioCLIP 2.5 Huge on a phone, plus a live scanner. The forager-safety framing is application context only.
- **D-002 · 2026-11-05 · Two stages:** live MNv4 detector (genus/family) → user captures a still → BioCLIP species ID. All inference on-device.
- **D-003 · 2026-11-05 · Target hardware:** Pixel 9 Pro (Tensor G4). Fallbacks are Pixel 8/7 and generic arm64 through the GPU/CPU delegates. Training on the RX 7900 XTX with ROCm.
- **D-004 · 2026-11-05 · Labels:** the class is the species. Edible/utility/toxic/conditional is a multi-label lookup per (species, part), not predicted. Contact and ingestion toxicity are separate flags.
- **D-005 · 2026-11-05 · Trait sources:** USDA PLANTS, FDA PLANTOX, NC State Plant Toolbox, Clemson HGIC, other state extensions. Every row is cited. On conflicts, the more toxic label wins.
- **D-006 · 2026-11-05 · Species universe:** every forage-relevant regional species, including trees, vines, roots, sedges, mosses and liverworts. Regional GBIF count ≥ 20, plus a must-include list and look-alike groups.
- **D-007 · 2026-11-05 · Training images** can come from anywhere if labeled. The test set is regional, leakage-free, and has a date cutoff.
- **D-008 · 2026-11-05 · OOD:** plants not in the label set are ignored. No open-set rejection.
- **D-009 · 2026-11-05 · Caps:** 500 per species, 2× for toxic or look-alike species.
- **D-010 · 2026-11-05 · BioCLIP adaptation:** zero-shot baseline → text-initialised linear probe (main). LoRA is optional and experimental.
- **D-011 · 2026-11-05 · Quantization:** detector gets full INT8, with PTQ and QAT both reported. BioCLIP gets weight-only INT8/INT4 with FP16 activations by default, and partial W8A8 is tried for the NPU. Budget ≤ 10 s.
- **D-012 · 2026-11-05 · Detector:** SSDLite on MNv4-Conv-Medium (pure conv, NPU-friendly, Apache-licensed). YOLO is a comparison only (AGPL). Boxes are pseudo-labeled with OWLv2. 10–30 inferences/s over full-fps passthrough.
- **D-013 · 2026-11-05 · Output UX:** Pl@ntNet-style top-5 relative percentages with reference thumbnails, a caution sign plus info tabs, and no safe/unsafe verdict (liability). Temperature calibration is kept so the percentages mean something.
- **D-014 · 2026-11-05 · Later features designed in now:** GPS/season priors (state×month counts recorded during the data pull) and observation logging.
- **D-015 · 2026-11-05 · Organ/part:** the organ tag is stored in the manifest now. Part-aware BioCLIP (prompts or a small organ head) can be added later. The detector stays part-agnostic.
- **D-016 · 2026-11-05 · App:** native Kotlin, Android only, models downloaded on first launch. Logs are CSV.
- **D-017 · 2026-11-05 · No fungi or lichens, ever.** Mushroom mistakes kill, plant-contact mistakes mostly cause rashes. BioCLIP 2.5 is also weak on fungi (33.5% zero-shot).
- **D-018 · 2026-11-05 · Field test set:** Pixel video walks in the region with about 200+ hand-labelled frames (boxes + genus). This is the real test for the live detector, since close-up observation photos don't represent walking use.
- **D-019 · 2026-11-05 · Will ship as a real app:** thumbnails bundled in the app are CC0/CC-BY/public domain only, with attribution. Training images may use any license that allows ML training, and each image's license is recorded in images.parquet so the training set can be re-filtered later.
- **D-020 · 2026-11-05 · Museum/herbarium specimens:** kept for BioCLIP linear-probe training (helps generalization), excluded from detector training.

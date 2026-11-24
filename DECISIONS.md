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
- **D-021 · 2026-11-16 · Image curation:** training images must come from a curated or verified source, and every image carries a `curation_tier`. Allowed: `institutional_specimen` (herbaria, botanic gardens, museums), `eol_curated` (Encyclopedia of Life partners) and `community_verified` (iNaturalist research grade, which needs ≥ 2/3 community agreement and is the only iNat grade GBIF publishes; Pl@ntNet; observation.org). Dropped: other GBIF observations whose validation varies by publisher, machine IDs, fossils, and licenses that don't permit ML training (all-rights-reserved, "other", unknown). A strict institutional-only subset is one filter away (`configs/data.yaml: curation.allowed_tiers`) and can be reported as an ablation. Refines D-007.
- **D-022 · 2026-11-16 · Leakage-free test pool:** BioCLIP 2.5's ToL-200M images come from the GBIF snapshot of 2024-05-01 (doi:10.15468/dl.bfv433), so `test_min_event_date = 2024-06-01`. Only ~2.4K regional Pl@ntNet records are that recent, against ~670K iNaturalist research-grade ones, so the test pool is regional Pl@ntNet **and** iNaturalist, up to 50 observations per species, and phash-distinct from every ToL image of the species. Recent regional observations beyond the test quota go to train/val. Refines D-007 and the DATA.md note that Pl@ntNet is the main test source.
- **D-023 · 2026-11-17 · Download wide, select later:** keep the full D-006 label set (4,411 regional species: 3,890 vascular, 346 mosses, 175 liverworts) and download images for all of it. Disk is not the constraint, so narrowing the trained species set is a later selection step over manifests, not a reason to skip downloads. Trait data (USDA PLANTS etc.) and the GBIF download DOI are deferred.
- **D-024 · 2026-11-18 · Model selection on regional val:** probe checkpoints, hyperparameters and the calibration temperature are chosen on the regional slice of val (recent iNaturalist/Pl@ntNet photos that overflowed the test quota, ~20K images), never on test. 93% of val is TreeOfLife, which BioCLIP 2.5 was pretrained on: selecting on whole val picked an epoch-0 checkpoint while train loss kept falling, and a temperature fitted there under-corrects deployment-distribution confidence. Whole-val numbers are still reported for reference.
- **D-025 · 2026-11-18 · Probe recipe:** 10 epochs, AdamW lr 1e-3 with one-cycle, anchor 1.0 to the text init, logit adjustment τ = 0.5 (train on logits + τ·log train prior, predict without), all rows trainable, no label smoothing. Selected on regional val from a 17-run sweep (`results/tables/bioclip_probe.csv`). τ = 0.5 over τ = 0 trades 0.26 pt regional val for +5.2 pt on rare-species val; the two finalists were then both run on test (84.9% vs 84.8% top-1), so test saw two candidates, not one. Test (176,590 recent regional photos, 3,728 species): zero-shot 82.1% → probe 84.9% top-1 (95% CI 84.7–85.1, observation bootstrap), top-5 95.7% → 96.4%, ECE 0.085 → 0.005 after temperature 1.39. Open issue: species with < 20 training images (54 in test) fall from 26.9% zero-shot to 13.5%, and val cannot see it because their val images are ToL images; fixing it needs a regional validation slice for rare species.
- **D-026 · 2026-11-19 · Rare species: out-of-region validation and training photos.** The D-025 probe fell below zero-shot on species with < 20 training images (13.5% vs 26.9% test top-1), and val could not see it because those species have no regional val photos. Added `gbif_external`: iNaturalist research-grade and Pl@ntNet photos dated ≥ 2024-06-01 from **outside** SC/NC/GA/TN for the 681 species with < 100 train images. This yielded 9,046 images for 405 species, split by observation into val_ext (≤ 20 observations per species, 6,735 images) and train_ext (2,257). These photos are deduplicated against and after all other rows and never reach test, and the existing train/val/test assignment was verified unchanged row for row. val_ext reproduces the regression (zero-shot 22.9% vs probe 10.9% on < 20-image species). The probe now trains on train_ext (`train_ext: true`). Test: 85.0% top-1, 96.5% top-5, 78.1% per-species recall, < 20-image species 20.0% (from 13.5%; zero-shot 26.9%). Per-species shrinkage of rare rows toward the text embedding (`shrink_k: 30`) lifts < 20-image species to 40.7% but costs ≥ 0.34 pt regional-val top-1 at every threshold tried (test: −0.6 pt top-1, mosses 68.5% → 63.2%). Under the rule fixed in advance (≤ 0.1 pt regional cost), it is off. It remains a product choice to revisit once location/season priors (D-014) exist, since priors naturally down-weight rare species.
- **D-027 · 2026-11-19 · Trait table, first source: NC State Extension Gardener Plant Toolbox.**
  - **Sources:** FDA PLANTOX is gone (404), and USDA PLANTS returned 504 all night, so the first source is the NC State toolbox. It has no robots.txt and no terms restricting automated access. Pages are fetched at 1 request/s and cached under `$TRANSFERNER_DATA/raw/ncsu/`.
  - **Matching:** label-set species are matched to sitemap pages by accepted name or GBIF synonym (1,646 species). Species without their own page fall back to their genus page (705 more species, 160 genera).
  - **Rules** (`data/traits_build.py`; nothing is inferred beyond what the page states):
    - **Toxic:** a "Poisonous to Humans" section gives one `toxic` row per listed poison part, with `ingestion_toxic = 1` and severity Low/Medium/High → low/moderate/high. "Causes Contact Dermatitis: Yes" sets `contact_toxic`.
    - **Edible:** plant type "Edible" or `#edible <part>` tags give `edible` rows. If no part is named, the part is `unspecified` and the page's edibility text goes in `conditions`.
    - **Conditional:** an edible claim on a page that also has a poison section becomes `conditional`, and so does every edible claim from a genus page. Genus-page toxic rows apply to the congeners as stated.
    - **Missing data:** the absence of a toxic row never means "safe".
  - **Result:** 2,714 rows over 872 species (584 toxic, 190 contact-toxic, 226 edible, 176 conditional). Source pages cover 68% of test images, 89% of the top-500 regional species and 40/43 look-alike species. They cover only 2 of the 521 bryophytes.
  - **DMR baseline** (test, no abstention): zero-shot 6.44% contact / 4.63% ingestion → probe 5.45% / 4.38% (9,078 / 31,187 toxic-species images). With the calibrated probe abstaining below 0.8 confidence, it falls to 1.16% / 0.77% at 83% / 78% coverage.
  - **Open:**
    - USDA PLANTS (retry) and Clemson HGIC as second sources, with the more-toxic-wins merge.
    - Thirteen look-alike *Rubus* species rely on the genus page.
    - The pet-toxicity tags are not used.
- **D-028 · 2026-11-19 · NPU quantization (A2): full static W8A8 with SmoothQuant α = 0.5 passes, and no layer needs to stay FP16.**
  - **Recipe:** all 128 block linears (qkv, out, fc1, fc2) are quantized. Weights are INT8, symmetric per output channel. Activations are INT8, asymmetric per tensor, with static ranges at the 0.01/99.99 percentiles over 1,024 train images. SmoothQuant α = 0.5 is folded into the LayerNorms and V rows. Patch embedding, LayerNorms, attention matmuls, softmax, GELU and the final projection stay FP16 in the simulation (`models/efficient.py`).
  - **Selection (val):** `quant.partial_search` tries k = 0, 4, 8 … FP16 layers and stops at the first k whose regional val reaches 98% of FP16 (88.23%). k = 0 reached it: 88.90% in the search run and 88.75% in the test run, against FP16's 90.03%. The two runs differ because calibration subsampled activations without a seed, so the noise is about ±0.15 pt; the sampler is seeded from now on. val_ext is 49.8–50.0% (FP16: 48.9%).
  - **Test (reported, acceptance):** 83.52% top-1 against FP16's 84.95%. That is −1.43 pt, or 98.32% of FP16, against a threshold of 83.25%. Top-5 is 96.12% (FP16: 96.45%). Species with < 100 train images reach 58.1% (FP16: 58.0%), and ECE is 0.0054. Predictions agree with FP16 on 90.3% of photos, and McNemar p ≈ 3e-123, so the loss is real but inside the budget. Weights are 606 MB.
  - **Safety cost is larger than the top-1 cost:** the dangerous-miss rate (D-027 trait table, test, no abstention) rises from 5.46% to 6.20% for contact and from 4.41% to 4.68% for ingestion. That is +14% and +6% relative, against a −1.7% relative top-1 loss. `w8a8dyn` (5.41% / 4.36%) and `w8a16` (5.44% / 4.36%) leave DMR unchanged. If the NPU path ships, DMR should be checked again on the device build, and it is one more reason to prefer the k = 1 fallback if that recovers most of the loss on val.
  - **Comparison:** the same INT8 with per-token *dynamic* activations (`w8a8dyn`, XNNPACK-style) costs nothing: 84.96% on test, p = 0.69. W8A16 doesn't either: 84.94%, p = 0.74. The static loss therefore comes from per-tensor activation ranges, not from INT8 weights.
  - **Sensitivity (`results/tables/quant_sensitivity.csv`):** quantizing one layer at a time on 1,024 regional val photos, `b1.fc1` dominates. Its cosine distance is 0.0179, which is 3.3× the next layer (`b30.qkv`, 0.0055), and on its own it drops top-1 from 90.4% to 89.2%, about as much as quantizing the whole model does on full val.
  - **Margin caveat:** the test margin is only 0.27 pt. The simulation quantizes linear inputs only; a real NPU compiler may also quantize attention matmuls, softmax or LayerNorm, and its calibration differs. The NPU number therefore has to be measured on the phone. Tonight's fallback check is val-only runs with `b1.fc1` (k = 1) and the top four layers (k = 4) kept in FP16 (`w8a8p_s0.5_k1/k4`; on the device that would be FP16 or 16-bit activations for those layers), which tells the app how much margin one fallback layer buys. **Result for k = 1** (same round): regional val 88.79%, no better than k = 0, so one fallback layer buys nothing. The damage is spread across the network, and the ranking from one-layer-at-a-time sensitivity doesn't add up. k = 4 was dropped. Calibration looks like the better lever: the PT2E histogram-observer export (D-029) reaches cosine 0.974 against FP32, where the simulation's 99.99-percentile ranges reach 0.944. Tighter clipping is catastrophic: the 99.9th percentile gives 59.9% regional val (cosine 0.585), because ViT-H's outlier activation channels carry information and must stay inside the range. The 99.5 run was dropped, and plain min/max ranges (`w8a8_s0.5_p100`, val-only) are queued instead. The histogram observer's error-minimizing ranges sit between the two, and an MSE range search in `fq_act_static` calibration is the next candidate if p100 doesn't help.
  - **Status:** quantization is done per the OVERNIGHT.md gate. Work moves on to structure (A3), combinations (A4) and the per-runtime report (A5).
  - **Superseded in part by D-031:** min/max activation ranges replace the 99.99th percentile, giving test 84.82% (−0.12 pt) and DMR 5.56% / 4.42%.
- **D-029 · 2026-11-19 · Export (A6, x86 feasibility): the image tower converts to LiteRT, and the NPU graph must quantize only the block linears.**
  - **Toolchain:** litert-torch 0.10.0, torch 2.14.1+cpu and ai-edge-litert 2.3.0 in `.venv-export` (`configs/export-constraints.txt`). jax must stay below 0.12.1: jax 0.12.1 (2026-11-18) lowers GELU's erf with `vhlo.fma`, which the converter's StableHLO 1.19 serializer rejects. ExecuTorch 1.6 pins torch 2.15 and would need its own venv; it wasn't tried.
  - **Check** (`export/litert.py`, `results/tables/export_check.csv`): image tower → normalized embedding, compared with PyTorch FP32 on 64 regional val photos (cosine; top-1 under the production head). Latency is single-image x86 XNNPACK with 8 threads.

    | Recipe | Size (MB) | Cosine vs FP32 | Top-1 agreement | x86 latency |
    |---|---|---|---|---|
    | FP16 | 1,208 | 1.0000 | 64/64 | 322 ms |
    | W8 weight-only | 610 | 0.9997 | 64/64 | 325 ms |
    | W8 + dynamic INT8 activations | 610 | 0.998 | 63/64 | 110 ms |
    | Static INT8, linears only, SmoothQuant 0.5 (D-028 scope), 32 calibration images | 620 | 0.975 | 64/64 | 126 ms |

    The PT2E graph of the static INT8 recipe scores 0.974 before conversion, so the converter is faithful.
  - **Do not use the default full-INT8 recipe:** the stock PT2E static recipe also quantizes the residual `add`, `mul` and `cat`, and it destroys the embedding: cosine 0.25 with 0/64 agreement, and 0.245 with SmoothQuant. Per-tensor INT8 cannot hold ViT-H's residual stream.
  - **Decided:** the NPU graph quantizes the 128 block linears (weights per channel, activations static per tensor), and residual adds, LayerNorm, softmax, GELU and attention stay float. This is exactly the D-028 simulation, so D-028's accuracy numbers apply to this graph.
  - **Conversion needs memory:** it peaks at 14–17 GB RSS, because the FP32 module (> 2 GB) is serialized once before quantization. FP32 itself isn't a shipping format; FP16 at 1.2 GB fits in one flatbuffer.
  - **Proxy only; these need the phone:**
    - Tensor G4 latency on the GPU delegate and the NPU.
    - Peak memory.
    - Whether the NPU compiler accepts INT8 linears mixed with float residual ops, or splits the graph into many partitions or CPU fallbacks.
    - On-device accuracy.
  - **On x86 XNNPACK,** weight-only W8 runs at FP16 speed (it saves memory only), and dynamic INT8 is 3× faster.
- **D-030 · 2026-11-19 · Probe recipe: field photos weighted 8× in the loss (Phase B).**
  - **Why:** the round-1 moss diagnosis found that training is 99% ToL. Field photos (regional train rows and train_ext) are 0.9–1.3% of bryophyte training and 7.6% of vascular training, but they're what the phone sees.
  - **Runs** (`results/tables/bioclip_probe.csv`, tags `phaseB_*`, seed 0):
    - Bryophyte weights 2 and 4 lowered `val_ext_bryo`, and the 14.6K unused ToL moss images added +0.26 pt (within noise), so both fail the Phase B rule.
    - A field weight w passes at every value tried. From w = 1 to 32, regional val climbs steadily: 90.03 → 90.62 (2) → 90.93 (3) → 91.27 (5) → 91.50 (8) → 91.68 (12) → 91.77 (16) → 91.91 (24) → 92.00 (32). val_ext rises from 48.83 to 53.2 and `val_ext_bryo` from 30.94 to 33.9.
  - **Selection rule (val only):** regional val and val_ext share observers, places and time with the training rows being upweighted, so they keep rewarding a larger w and can't locate the optimum. The slice that doesn't share them is ToL val of rare species (`val_rare`). It gains slightly at first and then falls: 70.39 → 70.56 (5) → 70.38 (8) → 70.06 (12) → 69.15 (24). The rule is the D-026 guard applied to that slice: take the largest regional val for which `val_rare` loses ≤ 0.1 pt. That gives **w = 8**.
  - **Disclosure:** `linear_probe` always prints test, so test saw all nine candidates, and this guard was written with those numbers on screen. Test peaks at the same w: 84.96 → 85.15 (2), 85.21 (3), 85.28 (5), **85.32 (8)**, 85.29 (12), 85.24 (16), 85.00 (24), 85.20 (32).
  - **Test at w = 8 (reported):** +0.36 pt top-1 overall. Mosses go from 68.47% to 69.27%, liverworts from 67.84% to 69.41%, vascular plants from 85.42% to 85.76%, and species with < 100 train images from 58.09% to 59.58%. Field weighting helps mosses less than it helps everything else; moss domain shift is still the open problem. The next lever is recent field photos for all bryophytes (OVERNIGHT.md Phase B).
  - **Adopted:** `configs/bioclip.yaml: heads.linear_probe.w_field: 8`. Production is retrained, recalibrated and evaluated through the queue (`--tag d030_w_field8`, then `train.calibrate` and `eval.evaluate`). The D-026 head is kept as `artifacts/probe_d026.pt` and `temperature_d026.txt`. **Production result** (`eval.csv`, 05:27): test top-1 85.32% (D-026: 84.95%) and ECE 0.0053 at T = 1.424. DMR without abstention is 5.40% contact and 4.12% ingestion (D-026: 5.45% and 4.37%).
  - **Efficiency sweep:** it still scores every production-head variant with the frozen D-026 head in `artifacts/efficiency/_head/`, so its rows stay comparable. Its retrained-head variants (`base_subset`, `drop2`, `drop4`) now train with w = 8 and stay comparable with one another.
- **D-031 · 2026-11-19 · NPU calibration: per-tensor min/max activation ranges replace D-028's 0.01/99.99 percentiles.**
  - **Why:** D-028's recipe passed the gate by only 0.27 pt and raised contact DMR from 5.46% to 6.20%. Keeping `b1.fc1` in FP16 bought nothing (k = 1). Tighter clipping at the 99.9th percentile collapsed regional val to 59.9%, which pointed at clipping, not rounding, as the source of the loss. ViT-H has a few outlier activation channels. They make up more than 0.01% of the elements and carry information, so any percentile range clips them.
  - **Change:** `--a-pct 100` (`efficient.py`: the range is the exact min/max over the 1,024 calibration images). Everything else is as in D-028: all 128 block linears, per-channel INT8 weights, per-tensor asymmetric INT8 activations, SmoothQuant α = 0.5, k = 0.
  - **Val (selection), `w8a8_s0.5_p100`:** regional val 90.10% (FP16 90.03%; D-028 88.75–88.90%). val_ext 48.86% (FP16 48.89%). Agreement with FP16 is 97.8% and cosine 0.9955 (D-028: 93.0%, 0.944).
  - **Test (reported):** 84.82% top-1, against FP16's 84.95% and D-028's 83.52%. That is −0.12 pt and 99.85% of FP16, which clears the 83.25% threshold by 1.57 pt (D-028: 0.27 pt). Top-5 is 96.41% (FP16 96.45%), species with < 100 train images reach 57.54% (57.98%), and ECE is 0.0045. Predictions agree with FP16 on 97.0% of photos (D-028: 90.3%) at cosine 0.995. McNemar p = 1.6e-4: with n = 176,590 a small loss is still detectable, but it is a twelfth of D-028's. **DMR is 5.56% contact and 4.42% ingestion** (FP16 5.46% / 4.41%; D-028 6.20% / 4.68%), so most of D-028's safety cost is gone.
  - **Deployable graph (x86, `export/litert.py --pt2e-only`, `export_check.csv`):** this is the PT2E linear-only graph that D-029 converts faithfully, run with 128 calibration and 128 check images. With `MinMaxObserver` on activations: cosine 0.9951 (min 0.984), 127/128 agreement, and the same top-1 as FP32 on those images. With the default `HistogramObserver`: 0.9746 (min 0.931) and 125/128. The histogram observer minimizes the L2 error of the whole distribution, so it clips the outlier channels as well.
  - **Decided:** the NPU graph uses min/max activation observers (`static_quantizer(..., "minmax")` in `export/litert.py`). If an NPU toolchain recalibrates on its own, it must be told to use min/max ranges, not percentile or entropy ranges.
  - **Caveat:** min/max ranges are set by the most extreme calibration image, so a different calibration set can only widen them. Keep the calibration set as hash-ordered train images (1,024 in the sweep, 128 in the export check; both work). The on-device NPU check from D-028/D-029 still stands.
- **D-032 · 2026-11-19 · BioCLIP deployment variant per runtime (A5).**
  - **Frame:** BioCLIP runs once per captured still, with a ≤ 10 s budget (D-002, D-011). It doesn't run per frame, so FLOPs matter only if the phone misses the budget. Accuracy, safety (DMR) and memory or download size come first.
  - **LiteRT GPU delegate (first choice, `export.yaml: delegate_order`): W8A16,** INT8 weights per channel with FP16 activations. 606 MB; test 84.94% (Δ −0.01 pt, McNemar p 0.74); DMR 5.44% / 4.36% (FP16 5.46% / 4.41%). The memory-saving fallback is **W4A16 g32**: 343 MB, 84.52% (−0.43 pt). g128 is 315 MB at 84.37% (−0.58 pt), so g32 buys 0.15 pt for 28 MB.
  - **NPU: static W8A8 per D-031** (128 block linears, everything else float, min/max ranges, SmoothQuant 0.5). Test 84.82% (−0.12 pt), DMR 5.56% / 4.42%, 606 MB. No layer needs to stay FP16, so D-011's "partial" W8A8 is full W8A8 over the linears.
  - **XNNPACK CPU (generic arm64 fallback, D-003): W8A8 dynamic (8da8w).** 606 MB; test 84.96% (+0.01 pt); DMR unchanged; on x86 it runs 3× faster than FP16 or W8 weight-only (110 vs 322 ms, D-029). **8da4w** (`w4a8dyn_g32`, 343 MB) is the CPU choice when 606 MB is too much. Val-only (round 4): regional val 89.66% against `w4a16_g32`'s 89.63%, val_ext 48.05% (48.34%). Dynamic INT8 activations add nothing on top of W4, so its test top-1 is taken as `w4a16_g32`'s 84.52% (not run).
  - **Structure: none by default.** Every structural variant measured costs at least 0.8 pt of test top-1 (`res210`), and DMR rises faster still (`res196`: −1.81 pt, contact DMR 5.46% → 6.40%). If the phone misses the 10 s budget, the fallback order is lowering the input resolution (`res210`, then `res196`: −24% GFLOPs, +31% GPU throughput, a plain static shape) before token merging. ToMe's `tome_r2` costs less (−0.94 pt at −13%), but its gather/scatter, masked attention and data-dependent token selection are the hardest ops for a GPU or NPU delegate, and on ROCm `tome_r2` ran no faster than FP16 while `tome_r4` gained only 15% for 26% fewer FLOPs. **Round 4 update:**
    - **`res210` is the first fallback.** It costs 0.80 pt of test (84.15%) for −12% GFLOPs (293 vs 335) and +7% GPU img/s, with DMR 5.68% / 4.67%. At `tome_r2`'s FLOPs it scores 0.14 pt more (paired McNemar p = 0.019) and keeps a plain static graph. The order is `res210`, then `res196`, and ToMe only if a delegate turns out to handle it.
    - **Block dropping is off the list.** Against `base_subset`, the same 40-per-species retrained head on intact features (test 83.22%):
      - `drop2` loses 2.92 pt (80.30%) for −6% GFLOPs.
      - `drop4` loses 5.84 pt (77.39%) for −12.5% GFLOPs, which is `res210`'s saving at seven times the loss.
      - Contact DMR rises from 6.16% to 8.63% (`drop2`) and 10.40% (`drop4`).
  - **Static W8A8 on top of a structural cut (an NPU fallback) costs out-of-region photos extra.** The A4 screens are additive on regional val (`res196` 88.25 → 88.25%, `tome_r2` 89.23 → 89.23%), but not on val_ext:
    - `res196`: −0.76 pt (paired McNemar p = 0.004).
    - `tome_r2`: −0.97 pt (p = 0.001).
    - Static W8A8 alone costs val_ext nothing (−0.03 pt, p = 0.95).
    - Calibration already runs through the structural model at its own resolution, and min/max ranges don't depend on the sampling seed, so this is an interaction, not calibration noise. Test is regional and can't see it.
    - Before an NPU build ships with a structural cut, screen that exact combination on val_ext, and try calibrating on a pool that includes out-of-region field photos (`train_ext`).
  - **Needs the phone:**
    - Latency per delegate (GPU, NPU, CPU) on Tensor G4, against the 10 s budget.
    - Peak memory, including whether the GPU delegate keeps INT8 weights or expands them to FP16 at load (606 MB vs 1.2 GB resident).
    - Whether the NPU compiler accepts INT8 linears mixed with float residual and LayerNorm ops without splitting the graph into many partitions.
    - On-device top-1 and DMR for each shipped variant.
    - Energy per identification.
    None of these can be measured on this machine; x86 timings are proxies only.
- **D-033 · 2026-11-20 · Live detector: pseudo-box training data and recipe (Phase D).**
  - **Labels:** OWLv2 (`google/owlv2-base-patch16-ensemble`, Apache-2.0) over at most 100 non-specimen photos per genus (D-020), taken from BioCLIP's train/val/test. Each photo keeps its observation-grouped split.
    - That gives 132,302 photos, 1,359 genera with train boxes (309 families) and 199,073 raw boxes.
    - The planned 20 per genus would have left about 14 training photos per class.
    - 31.6% of photos get no box and carry one full-frame box instead: mosses 57%, liverworts 64%, vascular plants 27%. These are mostly frame-filling subjects.
    - `drop_nested()` removes part boxes inside a whole-plant box at load, taking boxes per photo from 1.75 to 1.35.
    - Box AP on val/test measures agreement with OWLv2 plus the genus label, not field accuracy. The real test is still the field video set (D-018).
  - **Head: one shared genus classifier** (`model.head: shared`) instead of torchvision's SSDLite head, which has a separate classifier per (level, anchor shape): 36 weight vectors per genus, each fed a sliver of about 70 photos.
    - Background is still scored per anchor; genus is scored per location from a 256-d embedding through one 1×1 classifier. Output layout and loss are unchanged.
    - Size: 9.1M parameters against 23.7M, and a much cheaper head at 1,359 classes.
    - Evidence, all at backbone lr 2e-3 (before the fix below): the stock head stalled at 0.6% val genus top-1 after 4 epochs. At epoch 1 the shared head reached top-1 0.35% and top-5 1.6%, against the stock head's 0.20% and 0.85%.
    - The two heads were not re-compared at the final learning rate.
  - **Image-level auxiliary loss** (`aux_image: 1.0`): globally pooled stride-32 features go through the same genus classifier, trained on the photo's genus. Every photo supervises the classifier every step. It also gives a cheap whole-frame genus readout (pool + 1×1).
  - **Backbone learning rate is the decisive setting** (`backbone_lr_mult: 0.03`, so 6e-5 for the pretrained MNv4 layers and 2e-3 for new layers).
    - At 2e-3 everywhere, image-level genus top-1 was 0.95% after 2 epochs.
    - A linear probe on *frozen* ImageNet MNv4 features through the same data path reaches 20.4% on clean training views and 14.7% on augmented ones. So the labels are right, and the field augmentation costs about a quarter of the accuracy but is not the problem.
    - Screens on the first 2 of 60 epochs, epoch-1 val, selected on genus AP50:
      - ×0.1: AP50 0.040, image genus top-1 13.3%, box genus top-1 3.8%.
      - **×0.03: AP50 0.051, 17.0%, 2.8%.**
    - Short screens favour lower rates, which is noted but not corrected for.
  - **Training:** 60 epochs, batch 64, AdamW, weight decay 0.05, 1,000-step warmup then cosine. The sampler weights each photo by (train photos of its genus)^-0.5. Field augmentation: SSD photometric distortion, zoom-out up to 3×, IoU crop, flip, then downscale-upscale, motion blur and JPEG at 320 px. The epoch is selected on val genus AP50 and test is run once.
  - **Engineering:** on this ROCm stack, MIOpen runs depthwise convs through per-group Winograd or naive kernels.
    - Depthwise convs are shift-and-add under `torch.compile`. The state_dict is unchanged and outputs and gradients match to 1e-13. Training goes from 239 to 393 img/s at batch 64 on an otherwise idle GPU, about 456 img/s in the real run.
    - The SSD loss is vectorized over the batch (identical to torchvision's, 87 → 30 ms).
    - Postprocessing is our own top-k plus NMS, because torchvision loops over every class. Objectness is 1 − P(background); the best single genus probability as the score gave class-agnostic AP50 0.045 instead of 0.408.
    - `coco_map` matches pycocotools to 1e-6 (`tests/test_coco_map.py`).
    - Triton needs Python.h, which comes from a uv-managed CPython 3.12. No sudo, nothing system-wide.
  - **Export** (`export/detector_litert.py`, CPU venv): backbone and head go to LiteRT; softmax, decoding and NMS stay app-side. FP32 is exact. Full INT8 PT2E (per-channel weights, per-tensor activations, BatchNorm folded) converts, and its input and outputs are INT8. On the smoke checkpoint it is 24.9 MB and 4.1 ms on 4 x86 threads.
  - **Result (2026-11-20, `mnv4_shared`, 4.3 h):** best val genus AP50 at epoch 57 of 60; val had flattened from about epoch 50 (0.365 → 0.371). Test, run once (12,358 photos): genus mAP 0.282, AP50 0.353; class-agnostic AP50 0.685; best-box genus top-1 49.1% (top-5 69.8%); family top-1 60.2%; whole-photo genus top-1 56.2% (top-5 76.6%), family 66.3%.
    - **Val is not test-like.** Detector train/val are 93.5% ToL photos, test is 100% recent regional field photos. Genus accuracy holds (val 49.5% → test 49.1%), localisation drops (class-agnostic AP50 0.744 → 0.685). Epoch selection used all of val, unlike D-024's regional-val rule; with a monotone curve it made no difference here, but future detector selection should report the regional slice.
    - **The whole-photo readout beats the best box** for genus (56.2% vs 49.1% top-1), so the live label should come from it (app prototype, 2026-11-23).
  - **PTQ (2026-11-23, 512 val photos, 256 calibration photos, x86 LiteRT):** full INT8 is 9.9 MB (FP32 34 MB). Histogram observer: AP50 0.557 → 0.541 (−1.6 pt), class-agnostic AP50 −0.7 pt, genus top-1 46.1% → 43.2% (−2.9 pt), family −2.7 pt, logit cosine 0.994. Min/max ranges are worse (genus −3.9 pt), the opposite of BioCLIP's D-031. FP32 LiteRT is exact. The genus loss justifies QAT (D-011); the export does not yet include the whole-photo genus output.
- **D-034 · 2026-11-23 · Android prototype 0.1: first app build, and the app's detector graph.**
  - **Design first:** the user reviewed a clickable screen mock (claude.ai artifact, built from a real test photo, the trained models' outputs on it and NC State trait text, quoted as written). The flow is setup → live scanner → capture or gallery → top 5 → species page, plus a log and settings.
  - **Live label = whole-frame genus** (D-033: 56.2% vs 49.1% test top-1). The detector export for the app (`export/detector_app.py`) emits whole-frame genus logits, per-anchor objectness and boxes instead of SSD's 3,234 × 1,360 class logits. Objectness is `sigmoid(logsumexp(g) − bg)`, which equals 1 − P(background) to 1e-6. INT8 PTQ on 512 val photos: whole-frame genus top-1 49.4% → 50.0%, family 61.3% → 61.1%, class-agnostic AP50 −0.7 pt. The live label loses nothing under PTQ, unlike the box genus (−2.9 pt, D-033), so QAT is not needed for the prototype. 11.9 MB.
  - **Caution (D-013):**
    - A banner appears when any of the top 5 has a cited toxic or conditional row. It names the species and what its source says.
    - Species without rows show "No trait data" (D-027).
    - Toxicity comes from `species_traits.csv` only, never from `species.parquet`'s `is_toxic`, which disagrees with it (e.g. *Delphinium tricorne*: flag False, cited rows poisonous if eaten, high severity).
    - No live caution yet; that is an open question for the user.
  - **Stack:** Kotlin, Compose, CameraX 1.7, LiteRT 2.4 `CompiledModel` (GPU → CPU fallback; NPU after the phone tests), minSdk 29, arm64 only, AGP 9.5 / Gradle 9.9 / compileSdk 37.2.
    - The toolchain is in user space under `/models/ml/android`, with no sudo. Accepting Google's SDK licence was approved by the user.
    - The APK is debug-signed for sideloading.
  - **Bundle (`export/bundle.py`, 708 MB):**
    - detector app graph + sidecar
    - BioCLIP W8A16 (D-032 GPU variant, 610 MB)
    - production probe as float16 + T
    - `catalog.json`: 4,411 species; cited trait rows for 872; 4,075 CC0 / CC BY thumbnails with credits
    - SHA-256 manifest

    It is served over plain HTTP on the home network by `scripts/serve_bundle.sh` (port 8765, this account), as the user chose. The app verifies every file's hash before use.
  - **Verification without a phone:**
    - JVM unit tests show the Kotlin preprocessing reproduces the Python inputs exactly: detector int8 input and BioCLIP float input, zero difference on the golden photo.
    - Post-processing (softmax, anchor decode, NMS) and the species head match Python to < 1e-4 and 2e-3.
    - The bundle download passes end to end against the server.
  - **Not verified (needs the Pixel):** LiteRT on device (GPU compile of the ViT-H graph, latency, memory), CameraX, and the UI on a real screen.
- **D-035 · 2026-11-23 · App 0.2: device plans and a fallback chain down to budget phones (Moto G Power 2025).**
  - **Target added by the user:** Moto G Power 2025: MediaTek Dimensity 6300 (2× Cortex-A76 2.4 GHz + 6× A55 2.0 GHz), Mali-G57 MC2, 8 GB, Android 15, no NPU confirmed by a primary source.
    - The 0.1 build would have tried the W8A16 ViT-H on that GPU (an out-of-memory risk), then fallen back to it on the CPU, where its INT8 weights are expanded to float.
  - **Three species-model variants in one bundle** (`export/bundle.py VARIANTS`). Manifest groups mean a phone downloads 99 MB of shared files plus one variant: 708 or 443 MB in total.
    - `gpu`: W8A16 (D-032).
    - `cpu`: dynamic INT8 (8da8w, D-032's XNNPACK choice). Logit cosine 0.9992 vs W8A16, same top 5 on the golden photo, 3.3× faster on x86 CPU.
    - `cpu-small`: new export `--recipe w4dyn`, INT4 weights in blocks of 32 + dynamic INT8 (8da4w, 345 MB). On 128 regional val photos: embedding cosine 0.985 (min 0.954) to FP32, top-1 agreement 97.7%, and top-1 accuracy equal to FP32 (96.1%).
  - **Plan from the device profile** (`core/DevicePlan.kt`, unit-tested on Pixel 9 Pro, Moto G Power 2025 and a 4 GB Moto G 2025):
    - The GPU variant only on SoCs whose GPUs are known to hold ViT-H (Google Tensor, Snapdragon 8 / SM8xxx, Dimensity 9xxx) with ≥ 10 GB.
    - Otherwise the `cpu` variant at ≥ 6 GB.
    - Otherwise `cpu-small`.
    - The live detector's runtime is measured on the phone (6 synthetic frames per runtime) rather than assumed.
  - **Fallback chain:** `gpu` → `cpu` → `cpu-small` → 2 threads, on a load or run exception, or on restart after a native crash. A crash flag is committed synchronously before every native load and run.
  - **Budget-phone behaviour:**
    - XNNPACK weight cache file: packs once, then file-backed loads.
    - The live detector pauses during identification and is capped at 8 fps (6 fps below 6 GB).
    - Captures are about 3 MP.
    - Camera binding drops the live analysis if the camera can't run three use cases together.
    - Storage and memory are checked before download and load.
  - **Not verified:** all of this behaviour on the Moto itself. The identification time on 2 A76 cores against the 10 s budget is the open number; Settings shows it and lets the user compare 2 and 4 threads.
- **D-036 · 2026-11-23 · Phone feedback on 0.2: live label retrain, source-faithful caution wording, field-log upload.**
  - **User report (first phone run):**
    - Fast, and BioCLIP ID is "mostly fine and well within the time budget".
    - The live label needs to be more accurate.
    - Fig and white mulberry were labelled poisonous.
    - BioCLIP failed on yellowing elderberry leaves in low light.
  - **Caution wording was not faithful to the source (fixed in app 0.3):**
    - NC State lists *Ficus carica* (sap: phytophotodermatitis, low) and *Morus alba* (unripe fruit, leaf, sap: low) as poisonous, and lists their fruit as edible. 0.2 collapsed that into "Poisonous if eaten".
    - The summary is now per part and per severity. 86 species are edible with low-toxicity parts and read "Fruit edible as described · some parts mildly toxic", with the banner naming the parts and severity.
    - Moderate or high (or unstated) severity stays "Poisonous if eaten · <severity>", or "Toxic to touch and eat · <severity>" when contact is listed (poison ivy).
    - The banner headline separates poisonous candidates from mildly toxic parts. The Poison Control button stays for moderate and high only.
    - Rows are unchanged; only the summary wording changed.
  - **Post-hoc live fixes, measured on cached whole-frame logits** (`eval/detector_prior.py`; regional val 1,304 photos, test 12,358):
    - **Regional genus prior** (`logits + τ·log p`, from 2.27M regional observations): annual prior +0.9 pt on val, −0.5 pt on test at τ 0.25. A month prior at τ 0.5 gives +1.8 pt on val but −0.4 pt on test.
    - Test is capped per species, so it can't show the real encounter skew. The evidence is too weak to ship a prior.
    - **Family fallback (shipped in 0.3):** showing the family when the top genus is under 0.5 makes the shown label right 62.3% of the time on val (61.9% on test), against 57.1% (56.2%) for always showing the genus. The genus is shown on 69% of frames.
  - **Live label retrain (`train/train_live_head.py`, run `mnv4_live`):**
    - The whole-frame head saw only the 100K boxed photos. It now trains on all 1.37M non-specimen training photos, with no boxes needed.
    - Each step mixes 32 boxed photos (SSD losses + image CE, to keep detection) with 96 unboxed photos.
    - The unboxed image loss is 0.5·CE(label) + 0.5·CE(BioCLIP genus distribution), the teacher being the production probe on the cached embeddings, calibrated, summed to the 1,359 genera, top 10 kept.
    - The head becomes MobileNetV4's own ImageNet classifier head (960 → 1280 → genera), warm-started from timm.
    - Low-light frames are added to the augmentation (`train_detector.low_light`: 15–60% of the light, sensor noise, gain back up).
    - Fine-tuned from `mnv4_shared`: lr 1e-3 for the new head, 3e-4 for the old heads, 1e-4 for the backbone, 3 passes (42.8K steps).
    - Selection is on val whole-frame genus top-1, with the regional slice reported. Result: below when it lands.
  - **Field logs:**
    - App 0.3 sends saved captures (photos + CSV with an optional note of what the plant really is) to `scripts/bundle_server.py`, which writes them under `$TRANSFERNER_DATA/field_logs/<device>/`. Names are restricted, files are capped at 25 MB, nothing is overwritten.
    - These are the beginning of the D-018 field set.
    - Result screens under 0.5 top-1 now show a low-confidence hint (light, framing, organs).
- **D-037 · 2026-11-23 · One live model plus a user-chosen focus, not a rotation of specialist live models; capture-side look-alike specialists deferred.**
  - **The question (user):** a small rotation of live models specialised by growth form (bushes, trees, vines) or edible part (berries, nuts, tubers), instead of one lightweight generalist.
  - **Measured on the live model's cached whole-frame logits** (`mnv4_shared`, regional test, 12,358 photos):
    - Genus top-1 is 56.2%. Only 21% of its genus errors stay inside the right family and 30% inside the right order, so the errors are coarse, not fine distinctions within a group.
    - Told the true group, the same model reaches 79.5% (family) or 72.8% (order).
    - Routed by its own group guess, it gets 56.2%, no gain: the router is the hard part (family 66%, order 70%).
    - Rotating k models costs k× compute or a 1/k update rate.
    - Specialists confidently mislabel out-of-category plants, and there is no open-set rejection (D-008).
    - Growth forms are fuzzy for key species: USDA lists poison ivy as herb, shrub, subshrub and vine.
    - Edible-part categories are unsafe: tubers are underground, and a "berries" set must still hold every toxic berry.
  - **Decision:**
    - **One live model, improved:** D-036's distillation retrain, then a higher input resolution (`--size 384`), then crop-and-classify once boxes localise.
    - **An optional focus chip:** tree, shrub, vine, herb, grass-like, fern, moss.
      - It adds a soft log-prior over genera; it never masks. A genus is in a category if any of its species can take the form; genera without habit data are never down-weighted.
      - Categories come from USDA PLANTS growth habits (`data/growth_habit.py` → `data/traits/growth_habit.csv`, cited per species) plus taxonomy for ferns and mosses.
      - The prior strength is chosen on regional val by `eval/focus_prior.py` (right-focus gain vs wrong-focus harm), shipped as `focus.json`. Result: below when it lands.
  - **Capture-side look-alike specialists: not now.** On the species model (production probe, regional photos):
    - The app's caution (any toxic or conditional candidate in the top 5) already fires for all but 0.44% (val) / 0.70% (test) of toxic-species photos. Serious-severity misses: 0.67% / 0.71%.
    - The remaining misses are scattered (at most 5 photos per toxic → non-toxic pair on val). Many are a toxic species confused with a congener that has no trait rows: *Solanum carolinense* → *S. dimidiatum*, *Asarum naniflorum* → *Hexastylis contracta*, *Caulophyllum*, *Trillium*, *Allium*. There is no cluster for a specialist to target.
    - Revisit with field logs.
  - **Open for the user: a congener caution.** Flagging candidates with no trait rows whose genus has a cited toxic member (230 species) cuts caution misses by a quarter (val 0.44% → 0.33%, test 0.70% → 0.53%). The cost: the banner then shows on 34% / 31% of non-toxic photos instead of 27% / 23%.
- **D-038 · 2026-11-24 · Warning flags, false positives over false negatives (user); live budget 200 ms on the Moto.**
  - **User:** "add warning flags. FP over FN". This settles D-037's open congener question in favour of more warnings. Rules measured on regional photos (`eval/warning_flags.py` → `results/tables/warning_flags.csv`). A miss is a toxic-species photo with no flag; a false alarm is a flag on a non-toxic photo.
  - **Capture (species model):**
    - The banner now fires on any cited toxic species, or any "relative", with probability ≥ 0.5% anywhere among the 4,411, not only in the top 5. A relative is a species with no trait rows of its own whose genus has a cited toxic species (230 species).
    - Misses fall from 0.44% to 0.17% on val and from 0.70% to 0.18% on test (serious only: 0.56% → 0.22%, 0.73% → 0.16%). False alarms rise from 27% to 36% and from 23% to 33%.
    - Flagged species outside the top 5 are listed as "also possible".
    - Relatives read "Relative of a poisonous / mildly toxic species", naming the cited congeners. They are never called toxic themselves.
  - **Live:**
    - A warning row appears when any whole-frame guess with probability ≥ 2% is a genus holding a cited toxic species. It reads "Possibly poisonous" if any such species is moderate/high or contact-toxic, else "Possibly mildly toxic", followed by "capture to check".
    - On `mnv4_shared`, regional val: misses 5.5%, false alarms 37% (test 5.6%, 35%). 206 of the detector's 1,359 genera hold a cited toxic species (140 serious).
    - Re-measure on the retrained live model.
  - **Latency:** the user allows up to 200 ms per live frame on the Moto G Power 2025 for a better accuracy trade-off. Next live model: `mnv4_live` fine-tuned at 448 px (`train_live_head --size 448`, about 2× the 320 px compute). The training photos are stored at about 480 px, so 448 is the useful ceiling. Measured Moto timings come from the app's device report (0.4+).
- **D-039 · 2026-11-24 · Adjustable plant-score sensitivity (user request).**
  - The live detector now returns its top 3 boxes after NMS unfiltered. The screen draws those scoring at least the user's threshold: 0.05 to 0.9, default 0.25, stored in prefs.
  - Controls:
    - A slider on the live sheet, shown with the prototype controls (timings on). Right means more sensitive.
    - The same slider in Settings, with a reset to 0.25.
    - The status pill counts the boxes actually drawn and keeps showing the frame's best score, so the threshold can be set against what the Moto actually scores.
  - **D-037 result (2026-11-24):**
    - USDA PLANTS growth habits were found for 4,198 of 4,411 species: 4,052 by exact name, 167 via a GBIF synonym, 192 unmatched. They are in `data/traits/growth_habit.csv`, with one profile URL per row.
    - The focus prior was tuned on `mnv4_shared`'s regional photos with the rule fixed in `eval/focus_prior.py`, which chose eps 0.01.
      - **Right chip:** genus top-1 +4.4 pt on val, +3.9 pt on test, compared at eps 1.
      - **Wrong chip:** −8.2 pt.
    - Per form on test (right chip): vine 52.6% → 66.0%, fern 60.7% → 69.5%, shrub 53.4% → 59.8%, tree 47.6% → 52.6%, grass-like 41.5% → 44.9%, moss 44.4% → 46.4%, herb 60.5% → 62.0% (herb covers 726 genera, too broad to help).
    - **Safety:** live warnings are computed from the unfocused distribution (app 0.4.3), so a wrong chip never silences a toxic genus.
    - **Shipped** in bundle 0.3 (`focus.json`, 9.9 KB). Bundle updates move unchanged files from the installed version (tested 0.2 → 0.3: only the manifest, `species.json` and `focus.json` were fetched).
- **D-041 · 2026-11-24 · Live retrain `mnv4_live` result (D-036/D-040 recipe); INT8 loses 3.7 pt; staged as bundle 0.4.**
  - **Run:** 3 passes over 1.37M unboxed photos plus the boxed stream with 50% mosaics, from `mnv4_shared`; 3.3 h. Val whole-frame genus top-1 was still rising slowly at the last step (72.6 → 72.7% over the final 2,849 steps), and the last step was selected. Test, run once on 12,358 regional photos, against `mnv4_shared`:

    | Metric | `mnv4_shared` | `mnv4_live` |
    |---|---|---|
    | Whole-frame genus top-1 | 56.2% | 72.8% |
    | Genus top-5 | 76.6% | 87.9% |
    | Family top-1 | 66.3% | 80.0% |
    | Mosaic class-agnostic AP50 | 0.31 | 0.64 |
    | Class-agnostic AP50 | 0.685 | 0.692 |
    | Box genus AP50 | 0.353 | 0.419 |
    | Best-box genus top-1 | 49.1% | 56.2% |

  - **Live warnings** (toxic genus ≥ 2%, regional): misses 3.0% / 1.1% (val / test) instead of 5.5% / 5.6%; false alarms 30% / 29% instead of 37% / 35% (`results/tables/warning_flags_mnv4_live.csv`).
  - **Focus chip** (eps 0.01, same rule): right chip +2.7 / +2.8 pt, wrong chip −7.7 / −7.5 pt.
  - **INT8 PTQ** (512 val photos): whole-frame genus top-1 72.1% → 68.4% (−3.7 pt), family −1.8 pt, class-agnostic AP50 −0.5 pt. `mnv4_shared` lost nothing.
    - **Not the cause:** the output rounding (simulated INT8 logits: 72.7% → 72.7%), the histogram observer (min/max: 68.9%), or the new head (the head rewritten as float `mean` + `matmul`, which the quantizer does not annotate: 68.75%, via `export/detector_app.py --float-image-head`).
    - **The cause:** the retrained backbone's INT8 features.
    - **Fixes:** QAT (D-011), or run the float graph on the Moto's CPU through XNNPACK in FP16. Its latency is not measured: an estimated 1.5–2.5× the INT8 model, inside the user's 200 ms budget.
  - **Staged, not published:** bundle 0.4 = `mnv4_live` INT8 (`detector_app_int8`, 14.0 MB) plus the same `focus.json`; a phone on 0.3 would fetch about 15 MB. `latest.json` stays at 0.3 until the user asks (they said the model can be updated later). `export/bundle.py --no-publish` builds without publishing.
  - **Published 2026-11-24 at the user's request:** `latest.json` → bundle 0.4 (`mnv4_live` INT8). The 0.3 pointer is kept in the session scratchpad; reverting means writing `{"version": "0.3", "manifest": "0.3/manifest.json"}` back to `$TRANSFERNER_DATA/bundle/latest.json`.
- **D-042 · 2026-11-24 · In-app updates for the sideloaded app (user request).**
  - The bundle server also publishes `app.json` (versionCode, versionName, APK file, size, SHA-256, notes), written by `scripts/publish_apk.py`. The script reads the version with aapt2, refuses an APK not signed with the sideload key or without a higher versionCode, writes `app.json` only after the APK is in place, and removes older APKs.
  - **The app (0.5+)** checks `app.json` and `latest.json` at launch and from Settings → Updates:
    - It downloads a newer APK into its cache, verifies the SHA-256, and opens the system installer. The first time, Android asks to allow installs from Transferner.
    - The update replaces the app in place: same package, same key, data kept (settings, model bundle, log).
    - A newer model bundle is offered the same way and fetches only changed files (D-041).
    - A banner on the live screen links to Settings when anything is new.
  - **Signing:** every APK is signed with one key, a copy of the debug keystore kept at `/models/ml/android/signing/transferner-sideload.keystore` (SHA-256 35a2f1fd…380e), set explicitly in `app/app/build.gradle.kts`. If that key were lost, updates would require an uninstall, which deletes the app's data.
  - 0.5 itself must be installed once by hand from the web page; later versions update in-app.
- **D-043 · 2026-11-24 · Live input 448 px (`mnv4_live448`), staged as bundle 0.5; selection fix; D-041's INT8 loss corrected.**
  - **Resolution.** The `mnv4_live` weights read larger frames with no retraining. Test (12,358 regional photos), whole-frame genus top-1 and mosaic box AP50:

    | Model | Input | Genus top-1 | Mosaic AP50 |
    |---|---|---|---|
    | `mnv4_live` | 320 | 72.8% | 0.643 |
    | `mnv4_live` | 384 | 75.7% | 0.634 |
    | `mnv4_live` | 448 | 76.9% | 0.535 |
    | `mnv4_live448`, 1 pass at 448 (step 12,000) | 448 | 77.0% | 0.643 |

    The pass at 448 (`--size 448 --epochs 1`, from `mnv4_live`) repaired the boxes (test class-agnostic AP50 0.391 → 0.692) but did not raise the label (val 76.5% → 76.4%).
  - **Selection bug.** `train_live_head.py` selected on the label alone, so step 0 (the 320 weights at 448, broken boxes) won by 0.05 pt and the final step was never saved. Now: the selection key adds `--box-weight` (0.25) × val mosaic AP50, the last step is always saved as `final.pt`, and `--eval-only CKPT --size N` scores a checkpoint without training. `mnv4_live448/step12000.pt` is `last.pt` with the full args. The mosaic metric is mostly seam detection (audit below), so this guards against broken boxes but does not rank good ones.
  - **INT8 at 448** (`detector_app_int8`, 14.0 MB, full regional test, not the 512-photo check): 77.0% float → 75.7% INT8 (−1.3 pt); class-agnostic AP50 0.692 → 0.682. Keeping the whole-frame head in float (`_fh`) gives 75.8% for 22.5 MB, so not worth it. Against the shipped 320 INT8 (71.8% on the same photos) that is about +4 pt on the phone. Moto latency not measured: about 2× the 320 graph's compute, an estimated 100–120 ms per frame against the 200 ms budget.
  - **D-041 correction.** The −3.7 pt INT8 loss for `mnv4_live` came from the 512-photo check set. Paired on the full regional test it is −1.08 ± 0.38 pt (72.84% → 71.77%; 361 photos right only in float, 228 only in INT8), found by the read-only audit (`$TRANSFERNER_DATA/ops/audit-20261124.json`, finding `live_int8_precision-loss-overstated`). D-041's observer and float-head diagnoses were read from the same 512 photos. Export checks now use `--split test --images 20000`.
  - **INT8 logit ties.** The INT8 `image_logits` scale (0.27) makes some genera tie exactly; the 0.5 golden photo has Polemonium and Mertensia both at 0.0949. `GoldenTest` now accepts a different order only for exact ties. The audit's fix (shift and clamp the logits before the int8 output) is for the next export.
  - **Live warnings at 448** (toxic genus ≥ 2%, float logits, regional; `results/tables/warning_flags_mnv4_live448.csv`): misses 1.7% / 1.2% (val / test) against 3.0% / 1.1% for `mnv4_live`; false alarms 25.6% / 25.7% against 30.1% / 29.3%. Same threshold kept. `eval/detector_prior.py --ckpt` and `eval/warning_flags.py --logits` score a checkpoint other than `best.pt`.
  - **Bundle 0.5 staged, not published:** `mnv4_live448` INT8 plus the same `focus.json` (model-independent: same genera). The app reads the input size from the sidecar, so no app change is needed; all unit tests pass against the 0.5 and 0.4 goldens. `latest.json` stays at 0.4 until the user asks.
- **D-040 · 2026-11-24 (recorded late, after its result) · Encounter-weighted sampling for the live retrain, and an encounter-weighted metric.**
  - **Why:** the 50 genera met most often in the field (34% of 2.27M regional GBIF observations) got 3.8% of the first live model's training and 8.1% of `mnv4_live`'s (the sampler balances genera at power 0.5). Their live recall was 41% against 57% overall.
  - **What:** `train_live_head.py --encounter-mix M --encounter-power P` mixes the balanced per-genus share with the genus' share of regional observations (`priors_state_month`). Every eval also reports `encounter_top1` (per-genus recall on BioCLIP's ~20K regional val photos, weighted by how often each genus is met), `pool_top1` and `top50_recall`; `--select encounter` selects on it.
  - **Result (`mnv4_live_enc`: from `mnv4_live`, 1 pass at 320, mix 0.5, power 1, lr 5e-4).** Test, regional; the trait columns are genus-balanced recall on all non-specimen val photos of the 339 genera with cited trait rows, 40 per genus (`eval/trait_recall.py`):

    | Model | Genus top-1 | Encounter-weighted | Top-50 genera | Pool (176K) | Edible genera | Toxic genera |
    |---|---|---|---|---|---|---|
    | `mnv4_live` @ 320 | 72.8% | 72.4% | 70.5% | 72.4% | 74.0% | 79.1% |
    | `mnv4_live_enc` @ 320 | 71.9% | 76.1% | 77.5% | 73.3% | 74.3% | 78.7% |
    | `mnv4_live448` @ 448 | 77.0% | 77.0% | 75.4% | 77.1% | 78.7% | 82.8% |

    - Encounter weighting buys +3.7 pt on what a walker meets and +7 pt on the commonest genera, for −0.9 pt per photo, and leaves the trait genera flat on average.
    - **Cost on the genera it down-samples** (the read-only audit's warning; 40 photos each, so ±7 pt per genus): fig 70 → 60%, pomegranate 82.5 → 77.5%, feijoa 67.5 → 65%, Cicuta 67.5 → 65%, Conium 70 → 65%. The regional pool cannot see this (0–1 photos of the cultivated genera).
    - 448 px alone beats it on every column except the top-50 genera.
  - **Next:** if encounter weighting is kept, exempt the 339 trait genera (target = max(mixed, balanced), renormalised, as the audit proposes) and combine it with 448 px in the next live retrain; select on encounter or pool top-1 guarded by the trait-genus recall. Not shipped.
- **D-044 · 2026-11-24 · App 0.6 + bundle 0.6: the read-only audit's app fixes, walk mode, background imagery; the D-044 live recipe.**
  - **User answers (2026-11-24):** night flashlight use is not typical; fields and woods are; add some cultivated perennials (not annual crops); passive walk mode is fine; downloads approved, including "a good chunk of background / true-negative imagery". Bundle 0.5 (448 px, D-043) published at the user's request.
  - **Live screen:** the 3:4 analysed frame is shown whole between the top bar and the sheet (0.5 hid about two thirds of it). Boxes after the first need max(slider + 0.15, 0.4); a box at least 80% inside a kept box is dropped unless that box fills the frame; an IoU tracker (2 hits to show, 2 misses to drop, EMA 0.5) steadies them. The label uses a time-based EMA (tau 0.56 s). Warnings keep their own faster EMA (tau 0.245 s, 0.5's alpha 0.4 at 8 fps) and are held 4 frames. Tap to focus; torch button; stills at MAXIMIZE_QUALITY.
  - **Plant-presence gate** (`eval/presence_gate.py`, scored with the shipped INT8 graph on held-out frames): top-5 whole-frame mass of the current frame. AUROC plant vs plant-free 0.97 (objectness: ~0.45).
    - Boxes need 0.464: kept on 97% of degraded plant photos; plant-free frames with a box 98% → 28%.
    - The toxic row needs 0.197: the highest threshold that keeps 99.5% of plant frames and adds at most 0.1 pt of toxic-warning misses (2.35% → 2.45% on 4,000 toxic-genus photos); it is suppressed on 26% of plant-free frames.
  - **Walk mode:** screen on and dimmed, 4 fps, a forage score S = Σ p_g · e_g over alert-tier genera (e_g = share of the genus' regional sightings that are cited edible species), alert when the 1 s EMA of S stays ≥ 0.5 for 0.75 s (clears < 0.35). BioCLIP then checks the frame; a confirmed genus (≥ 25% genus probability, or BioCLIP's top species in that genus) buzzes twice and is quiet for 10 min, a rejected one for 30 s, an unchecked one is released. The window of the new `region_logits` output (9 overlapping windows, `detector_app.py --regions`, +1.4% compute, 14.7 MB) where the genus scores highest is drawn. A serious-toxic genus mass ≥ 0.3 held 1 s gives a long buzz. Walks are recorded (JSON line per frame, a JPEG per second, 1.5 GB cap) with a "missed plant" marker, and uploaded with the log.
  - **Forage list** (`data/forage/default_tiers.csv` → `forage.json`): 47 alert genera (fruit, nuts, tubers, shoots, high-value greens, and perennial yard and orchard plants such as fig, pomegranate, feijoa, citrus, jujube, kiwi), 57 show (low-value or preparation-dependent: dandelion, plantain, clover, acorns, mayapple...), 23 annual crops and the 65 unlisted edible genera off (several are mostly toxic: Hydrangea, Trillium, Lathyrus). Editable in Settings → Walk mode. A genus is never called edible; each carries its cited caution note.
  - **Capture:** name twins summed (28 names); levels: species when p1 ≥ 0.8, "<Genus> sp." when the genus sum ≥ 0.8, else no confident match; one level lower when the text score (max cosine to the species-name embeddings, `text_emb.bin`, 9 MB) is below 0.561, and then "maybe not a plant I know, toxicity unknown". Caution banner when cited-toxic species and relatives together reach 0.2% (stronger at 2%) or a conditional species in the top 5 reaches 2% (audit: regional test misses 57 → 43, false alarms 32.9% → 23.7%). A second photo of another part is asked on edible/toxic conflicts (p1 < 0.9) and offered when p1 < 0.8; the two are fused by mean log-probability and the banner is the union of the fused and both single-photo banners.
  - **Logging:** a JSON sidecar per capture (levels, text score, banner sum, the live model's last 16 frames); new CSV columns id_ms, text_score, level, photos; uploads send the CSV first (the 0.5 upload at 02:07 left five photos without rows) and skip files the server refuses. The device report has per-stage times, fps, thermal state, battery drain, GPU errors and walk alert counts; the detector benchmark also picks 2 or 4 CPU threads and reruns after a bundle change.
  - **Review:** three independent reviewers (safety, Android runtime, spec/Python) found 18 defects before release, all fixed: among them the warning EMA (a brief toxic pass no longer reached 2%), the two-photo banner (now a union), capture rules waiting on the detector, a shared crash flag overwritten across threads, an empty walk file blocking every later upload, alert lockouts without a BioCLIP check, sports fields counted as plant-free, missing Places category names, and unlisted forage genera showing as edible chips.
  - **Background imagery** (`scripts/fetch_negatives.sh`, `data/negatives.py`, `manifests/negatives.parquet`): 101,600 frames, split 80/10/10 by id hash. Plant-free: 55,524 COCO 2017 photos with no plant, turf or slope pixels (stuff labels) and no plants or plant food (things), and 30,000 Open Images photos with "Plant" verified absent. Vegetation scenes for composites: 9,146 COCO, 3,830 Open Images, 3,100 Places365.
  - **D-044 live recipe** (`train_live_head.py --preset d044`): seam-free composites (1-3 OWLv2-boxed plants, 15-50% of the frame, feathered, on real vegetation; 40% of boxed samples) with area-weighted whole-frame targets; plant-free frames (15%) with a uniform whole-frame target; SSD loss with a 32-anchor negative floor and no negatives ≥ 50% inside a GT box (`train_detector.ssd_loss_batched`); vascular full-frame fallback boxes out of the box loss (18,964 of 100,018 photos); ColorJitter without channel permutation plus white balance; encounter weighting with the 339 trait genera exempt; selection on pool top-1 + 0.25 × composite AP50 − 0.25 × plant-free box rate. Baseline (`mnv4_live448`): composite AP50 0.29, plant-free frames with a box 98.9%, trait-genus recall 80.4%. Run `mnv4_live_d044` started 12:27 (1 pass at 448, lr 5e-4).
  - **Season coverage:** `data/gbif_months.py` counts each species' regional sightings with images per month (yaupon: every month, 194 in November; the photos cover only January-February), for a month-stratified re-pull.

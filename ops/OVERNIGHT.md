# Overnight plan and journal

Started 2026-11-19 by the interactive session. Every unattended round reads this file first and
appends to the journal at the bottom.

## Goal (the user's instructions, 2026-11-19)
> Do efficiency. See what you can tackle fully autonomously [...] proceed beyond quant when good
> performance (0-2% loss acceptable in partial int8) is achieved. Do 2, 3, 4 as needed and evaluate
> if parallel tasking would be more efficient.

The numbering refers to the open issues the user was given:
1. efficiency (this plan's Phase A),
2. mosses (Phase B),
3. trait table (Phase C),
4. live detector (Phase D).

**Acceptance for the quantization stage:** partial W8A8 test top-1 must be at least 0.98 × the FP16 reference's test top-1. That is a loss of no more than 2% relative, which is stricter than 2 points. Checkpoints are chosen on val; test is only reported.

## Rules
These come on top of ~/.claude/CLAUDE.md, which still applies in full: no sudo, no apt or ROCm repo, no amdgpu-dkms, no flash-attn, and leave the media stack, Docker and the game servers alone.

**GPU work**
- **Everything on the GPU goes through the queue.** Add jobs with `scripts/q.sh add "python -m transferner.<module> <args>"`, or with `add --front` to run next. Never start GPU work directly.
- **Jobs run from the last commit of `setup/data-pipeline`.** The runner checks out a separate worktree, so commit before you queue a job that needs new code. Uncommitted edits are invisible to jobs.
- **CPU-only analysis is fine, with limits.** Use numpy, or torch with `torch.set_num_threads(8)` and no `.cuda()`. The machine also hosts game servers, so cap threads and RAM: keep at least 16 GB available.

**Processes, installs and data**
- **Killing processes:** only ones you started, and only by exact PID. Never use `pkill -f` or `killall`.
- **Packages:** install them with `uv` into `.venv`, or into a new separate venv such as `.venv-export` for the export toolchains (CLAUDE.md). Never touch system Python. Never add flash-attn.
- **Data** goes in `$TRANSFERNER_DATA`, never in git. Commit code, configs, docs, `results/tables` and `results/figures`.

**Method**
- Select on val (`val_regional`, `val_ext`) and report on test.
- D-026 rule for trade-offs: a change may cost common species no more than 0.1 pt of regional-val top-1 unless the plan says otherwise.
- New design decisions go at the bottom of DECISIONS.md as D-027, D-028, and so on.

**Web**
- Use the user agent from `utils/http.py`, at most 1 request per second per host, cache raw pages under `$TRANSFERNER_DATA/raw/`, and respect robots.txt and each site's terms. If a site disallows automated access, skip it and note that.
- Never send the user's name or email, or any credentials, anywhere.

**Git:** commit to `setup/data-pipeline` only, then `git push`, which uses the deploy key. Do not open PRs, because there is no `gh` here.

**Time:** about 60 minutes of work per round. Leave the queue non-empty when more GPU work is planned, so the GPU doesn't sit idle until the next round.

## Parallelism (evaluated 2026-11-19)
- **GPU work is a single sequential queue.** There's one RX 7900 XTX, and ViT-H embedding is compute-bound at about 190 img/s. Two concurrent jobs would just take turns on the GPU, and each needs 6–10 GB, which risks running out of VRAM.
- **CPU and network work overlaps the GPU queue.** That covers the trait table, analysis of cached embeddings, writing and reviewing code, and reports. Rounds start every 90 minutes even while the GPU is busy so this overlap happens.
- **Only one agent round runs at a time.** Two agents committing to one branch would conflict. Inside a round, a subagent is fine for clearly separate research, for example trait-source discovery, if it saves time.

## Tools
| Tool | Use |
|---|---|
| `python -m transferner.eval.efficiency --name N [--res --tome-r --drop --w-bits --w-group --a-bits --a-mode static --smooth --a-pct --fp16-from CSV --fp16-k K --retrain --val-only]` | Evaluates one variant and appends a row to `results/tables/efficiency.csv`. Embeddings go to `artifacts/efficiency/N/`. Agreement and McNemar are computed against the `fp16` variant. |
| `python -m transferner.quant.sensitivity --smooth A [--a-pct P --tag T]` | Per-layer static W8A8 damage, written to `results/tables/quant_sensitivity[_T].csv`. |
| `python -m transferner.quant.partial_search --smooth A --sens FILE [--ks ...] [--target 0.98] [--prefix P]` | Finds the smallest k of FP16 layers that meets the target on val, then runs that k on test. |
| `src/transferner/models/efficient.py` | All the knobs: token merging (ToMe), block dropping, resolution, and fake quantization (weights per-channel or per-group; activations dynamic per-token or static per-tensor with SmoothQuant; `fp16_layers`). |

Notes on the token-merging grid:
- ViT-H/14 has 32 blocks and 257 tokens, so r=8 leaves only 5 tokens at the end and the useful range is r ≤ 6.
- ARCHITECTURE.md's r=16 is impossible on this model.
- Token merging gains only about 14% GPU throughput on ROCm for 26% fewer FLOPs, because a masked SDPA uses a slower kernel. Judge it by FLOPs. On-device speed needs the phone.

## Plan
### Phase A: efficiency (priority)
- [x] **A1 (done in round 4):**
  - Done: `fp16` 84.95, `w8a16` 84.94, `w8a8dyn` 84.96, `w4a16_g128` 84.37 (−0.58 pt, 315 MB), `w4a16_g32` 84.52 (−0.43 pt, 343 MB).
  - `w4a8dyn_g32` (8da4w), val-only: regional val 89.66% (`w4a16_g32` 89.63%), val_ext 48.05% (48.34%). Dynamic INT8 activations add nothing on top of W4, so no test run (D-032).
  - References and weight-only / dynamic quantization:
  - `fp16`
  - `w8a16`
  - `w8a8dyn` (8da8w, XNNPACK-style)
  - `w4a16_g128`, `w4a16_g32`
  - `w4a8dyn_g32` (8da4w)
- [x] **A2: partial static W8A8 for the NPU.** Done in round 2, recorded as **D-028**.
  - k = 0 passed: all 128 linears are static W8A8 with SmoothQuant 0.5. Test 83.52% against a threshold of 83.25% (FP16: 84.95%).
  - The margin is thin and contact DMR rises from 5.46% to 6.20%.
  - Queued, val-only: `w8a8p_s0.5_k1` and `w8a8p_s0.5_k4`, which keep `b1.fc1` (or the top 4 layers) in FP16. `b1.fc1` is 3.3× more sensitive than any other layer. This shows how much one fallback layer buys. Run k1 on test only if it recovers ≥ 0.5 pt on val and the GPU has time.
  - The calibration sampler is seeded from commit 9650bac on, so earlier static runs carry about ±0.15 pt of calibration noise.
  - **Round 3: min/max activation ranges remove the loss on val** (`w8a8_s0.5_p100`: regional val 90.10% vs FP16 90.03% and k0 88.75–88.90%; cosine 0.9955 vs 0.944). The real PT2E graph agrees: `MinMaxObserver` gives cosine 0.995 against 0.975 for the default `HistogramObserver` (`export_check.csv`, `--pt2e-only`). **Test: 84.82% (−0.12 pt, rel 0.9985), DMR 5.56% / 4.42%** (D-028: 83.52%, 6.20% / 4.68%). Recorded as **D-031**, which replaces D-028's ranges.
- [x] **A3: structural variants.** *(Done in round 4. Test (Δ, GFLOPs): `res210` 84.15 (−0.80 vs FP16, 293). Against `base_subset` (83.22): `drop2` 80.30 (−2.92, 314) and `drop4` 77.39 (−5.84, 293). Block dropping is dominated; see D-032.)*
  - Test top-1 (Δ vs FP16, GFLOPs): `tome_r2` 84.01 (−0.94, 292), `tome_r4` 82.24 (−2.71, 249), `tome_r6` 78.96 (−5.99, 207), `res196` 83.13 (−1.81, 255, 247 img/s vs 188), `res168` 78.49 (−6.46, 186).
  - Contact DMR rises faster than top-1 falls: `res196` 5.46 → 6.40%, `tome_r2` → 5.72%, `tome_r4` → 7.24%.
  - `tome_r2`, `tome_r4`, `tome_r6`
  - `res196`, `res168`
  - `base_subset` (retrained head on 40 images per species), then `drop2` and `drop4`. A dropped-block variant must be compared with `base_subset`, not with the production head.
- [x] **A4: combinations along the Pareto front.** *(Done in round 4: two val-only screens, no test picks. Additive on regional val, but static W8A8 adds a significant val_ext loss on top of a structure: −0.76 pt on `res196`, paired p = 0.004; −0.97 pt on `tome_r2`, p = 0.001. Recorded in D-032.)* At most about 6 jobs, chosen from A1–A3. Example: the accepted partial W8A8 with the best token-merging r or resolution, or W4A16 with token merging. For a static combination, recompute sensitivity only if the structure changes a lot.
  - **GPU time is the constraint (round 2):** the queue runs until about 09:10, and the runner starts nothing after 10:00. Screen combos with `--val-only` (about 5 min each, 27K images), then run test (about 20–36 min) only for the 1–2 picks.
  - **Insert combos ahead of `base_subset`/`drop2`/`drop4`.** Those are the lowest-value jobs (−6%/−12% FLOPs, need a retrained head), so `drop4` is the first job to cut if time runs out.
  - Weight-only and dynamic INT8 are lossless (w8a16 −0.01 pt, w8a8dyn +0.01 pt), so a `w8a16 + tome_rX` combo will land within noise of `tome_rX`. Prefer combos whose interaction is uncertain: static W8A8 + tome/res, and W4 + tome/res.
  - **Round 3: screens done, both additive (round 3 journal); no combo test runs.** Original note: `w8a8_s0.5_p100_res196` and `w8a8_s0.5_p100_tome2`. Compare each with its float structure's val (`res196` 88.25, `tome_r2` 89.23). If the static penalty on top of the structure is ≤ 0.2 pt val, the combo is additive and needs no test run; the report can cite the screen. Run test only for a combo that is a real candidate (see the round 3 journal for the time rule).
- [x] **A5: report.**
  - [x] Add `src/transferner/eval/efficiency_report.py`, producing `results/EFFICIENCY.md` (a table of every variant: top-1/top-5 on test, Δ vs FP16, val, weight MB, GFLOPs) and `results/figures/efficiency_pareto.png` (test top-1 against GFLOPs and against weight MB). Load the `dataviz` skill before writing the chart code. *(Round 1: written and checked on a synthetic table. It runs on the CPU and also reports DMR per variant from the saved embeddings. Rerun it whenever rows land: `python -m transferner.eval.efficiency_report`.)*
  - [x] Add a decision entry recommending variants per runtime (LiteRT GPU, NPU, XNNPACK CPU), and state plainly what needs the phone: latency, memory and NPU compile. *(Round 3: **D-032**. GPU: W8A16, with W4A16 g32 as the memory fallback. NPU: static W8A8 per D-031. CPU: 8da8w. No structural cut by default. Round 4 added `res210`, `drop2`/`drop4`, the 8da4w val number and the static + structure val_ext caveat.)*
- [x] **A6 (optional, after A5): export feasibility on x86.** *(LiteRT part done in round 2, recorded as D-029. ExecuTorch was not tried: it pins torch 2.15 and would need its own venv.)*
  - Set up `.venv-export` (CPU torch) with ai-edge-torch and/or executorch.
  - Export the FP16 and W8 variants and check outputs against PyTorch on 64 images.
  - Record x86 latency only as a proxy.
  - Skip if the toolchain fights back for more than one round, and write down why.
  - **Round 3:** `export/litert.py --act-observer minmax` and `--pt2e-only` (checks the PT2E graph without the 14–17 GB conversion; 8 GB RSS). Linear-only static INT8 + SmoothQuant 0.5, 128 calibration and 128 check images: min/max cosine 0.9951 (min 0.984), 127/128 agreement; histogram 0.9746 (min 0.931), 125/128.
  - **Round 2 (started early, since the CPU was idle):** `.venv-export` has torch 2.14.1+cpu, litert-torch 0.10.0, ai-edge-litert 2.3.0 and open_clip 4.2.2. ExecuTorch 1.6 pins torch 2.15, which conflicts with litert-torch (< 2.15), so it would need its own venv (`.venv-et`).
  - `export/litert.py --recipe {fp32,fp16,w8,w8dyn}` converts the image tower, checks it against PyTorch FP32 on 64 regional val photos (cosine, top-1 agreement under the production head) and times it on x86. Results go to `results/tables/export_check.csv`.
  - Run it with `cd /tmp && PYTHONPATH=~/Transferner/src ~/Transferner/.venv-export/bin/python -m transferner.export.litert --recipe w8`. It's CPU only, with a memory guard at 16 GB free.

### Phase B: mosses (when A2 is done; mostly CPU on cached embeddings)
- **Baseline (test, current probe):** Bryophyta 68.5% top-1 across 140 species; Marchantiophyta 67.8%; vascular plants 85.4%.
- **Diagnose:**
  - Are errors within the genus or across it?
  - What share of moss training images are herbarium specimens rather than field photos?
  - How do training counts per species compare?
  - How many ToL moss images are sitting in split `unused` (the per-species cap)? Those are not embedded yet; embedding them is a GPU job.
  - How good is the text init (the zero-shot share)?
- **Fix only what val supports:** select on the bryophyte slices of val and `val_ext`, under the D-026 rule.
- [x] **Diagnosis (round 1, CPU, `python -m transferner.eval.moss_diagnosis` → `results/tables/moss_diagnosis*.csv`):**
  - **Errors are coarse, not congeners.** Of the probe's test errors, only 42% (Bryophyta) and 40% (liverworts) are within the genus, against 64% for vascular plants. 41% and 31% go to another family in the same phylum (vascular: 22%), and 11% and 19% go to another phylum (vascular: 0.5%). Genus top-1 is 81.7% and 80.7%, against 94.8%.
  - **Not just image counts.** Species with ≥ 400 training images reach 73.2% (Bryophyta) against 87.8% (vascular). Liverworts with ≥ 400 reach 85.3%. Median training images per species are 168 (Bryophyta), 108 (liverworts) and 415 (vascular).
  - **The training mix is almost all ToL:** 99.1% and 98.7%, with only 0.9% and 1.3% regional field photos (vascular: 7.6%). Specimens make up 14.3% and 16.3% of training (vascular: 2.7%).
  - **Unused ToL images:** 10,309 Bryophyta images (109 species) and 4,294 liverwort images (42 species) sit in `unused`. These are species already at the cap.
  - **Zero-shot share:** the probe adds only +3.3 pt (Bryophyta) and +0.6 pt (liverworts) over zero-shot on test. Liverworts with < 20 training images fall from 34.2% zero-shot to 10.1% with the probe (the D-026 effect).
  - **Selection slices:** regional val has only 149 Bryophyta and 93 liverwort photos (17 species). `val_ext` has 1,178 and 787, so `val_ext_bryo` is the main selection slice.
- [x] **Experiments, round 2, recorded as D-030:**
  - **Adopted: field-photo loss weight w = 8** (`configs/bioclip.yaml: w_field`). Test: overall 84.96 → 85.32, mosses 68.5 → 69.3, liverworts 67.8 → 69.4.
  - Bryophyte weights and the unused ToL moss images did not pass.
  - Production retrain, calibrate and evaluate were queued at the front. The old head is kept as `artifacts/probe_d026.pt`.
  - **Next moss lever:** recent field photos for all bryophytes (see below).
  - Original plan text follows.
- **Experiments (queued after A1):** `linear_probe --w-bryo {2,4}`, `--w-field 3`, `--w-field 3 --w-bryo 2`, plus a `phaseB_base` re-run for the bryophyte-slice numbers. Each writes to `artifacts/phaseB/*.pt` (never `--out` the production `artifacts/probe.pt`), and the rows land in `results/tables/bioclip_probe.csv` (columns `probe_val_ext_bryo_top1`, `probe_val_regional_bryo_top1`, `probe_val_regional_top1`). Accept a weight only if `val_ext_bryo` improves and regional-val top-1 drops ≤ 0.1 pt (D-026 rule). Next candidates:
  - Embed the 14.6K unused bryophyte ToL images. This needs a phylum filter in `train/embed_images.py`, because `--splits unused` would embed all 383K images (about 35 GPU-minutes).
  - Field-photo augmentation.
  - **Round 2:** the 7 Phase B jobs were moved to the front of the queue. They're linear probes of about 40 s each, so the whole set takes about 10 min.
  - **Production head and the efficiency sweep (solved in round 2):** the sweep's production-head rows now always use the frozen D-026 head in `artifacts/efficiency/_head/` (`efficiency.prod_head_dir`). Replacing `artifacts/probe.pt` no longer touches the sweep.
  - **Next candidate (not tonight):** recent out-of-region field photos for **all** bryophytes, in the style of D-026 but train-only. This needs a `train_only` flag in `splits.py` so `val_ext` stays fixed, and it rewrites the manifests, so do it only when no efficiency job is reading them.

### Phase C: trait table (CPU and network; run it during long GPU jobs)
- **What to build:** a per-(species, part) table of edible, utility, toxic and conditional, with contact- and ingestion-toxicity flags and a source citation (URL) on every row (D-004/D-005).
- **Sources to try:**
  - USDA PLANTS (bulk download)
  - FDA Poisonous Plant Database
  - NC State Extension Gardener Plant Toolbox
  - Clemson HGIC
  - other state extension services
- **Hard rule:** never assert toxicity or edibility from model knowledge. Only cited sources count, and on a conflict the more toxic label wins.
- **Where it goes:** check `configs/data.yaml` for the path `species.traits` expects.
- **Order of work:** must-include, look-alike and `is_toxic` species first, then by `n_regional`.
- **Afterwards:**
  - [x] Write a coverage report: `results/tables/trait_coverage.csv`, written by `traits_build`.
  - [x] Add DMR (dangerous misidentification rate) to `eval/metrics.py` and `eval/evaluate.py`. It is also in `efficiency_report`.
  - [x] Add a decision entry: D-027.
- **Round 1 status:**
  - [x] NC State toolbox: species pages plus a genus-page fallback. 2,714 rows over 872 species, with source pages covering 68% of test images. Re-run: `python -m transferner.data.traits_ncsu fetch && ... parse && python -m transferner.data.traits_build`. Fetching resumes from the cache.
  - [ ] Second source: USDA PLANTS (504 all night, retry it) or Clemson HGIC (robots allows it; factsheets are unstructured). Implement the more-toxic-wins merge when it lands.
    - **Round 2 (04:50), all dead ends:**
      - USDA PLANTS still returns 504 for the whole site, robots.txt and the API included.
      - The Canadian Poisonous Plants Information System (cbif.gc.ca) fails TLS hostname verification. It was skipped, not bypassed.
      - California Poison Control has dropped its coded plant list; only a handful of narrative entries remain.
      - Clemson HGIC search turns up no toxicity factsheets.
    - **Decided:** don't scrape narrative pages into a safety table. Retry USDA once per round, and stop there.
  - [ ] Three look-alike *Vitis* (*cinerea*, *palmata*, *rupestris*) have no page at all. Bryophytes have essentially no trait coverage (2 of 521).

### Phase D: live detector (GPU-heavy; only after Phase A's GPU work is finished)
- **Design (D-002/D-012):** SSDLite on MNv4-Conv-Medium, with boxes pseudo-labelled by OWLv2. The stubs are `src/transferner/data/pseudo_boxes.py` and `train/train_detector.py`.
- **Steps:**
  1. Pseudo-label a manageable subset of non-specimen images (specimens are excluded by D-020).
  2. Train.
  3. Report detection quality on a held-out subset.
- **Licences:** verify them. OWLv2 and timm's MNv4 weights should be Apache-2.0.
- **Round 1 assessment:** the queue (A1, A2, B, A3) runs until about 09:00, and A4 comes after it, so Phase D's GPU work will not fit before the 10:00 stop tonight. A later round with spare CPU time may write `pseudo_boxes.py` and test it on a few images on the CPU. Both stubs are still `NotImplementedError`. `transformers` 5.23 and `timm` 1.0.31 are already in `.venv`.
- **Packages:** install missing ones such as `transformers` and `timm` into `.venv` with `uv`.
- [x] **Round 3: licences checked** (HF API, cached under `raw/hf_cards/`): `google/owlv2-base-patch16-ensemble` and `timm/mobilenetv4_conv_medium.e500_r256_in1k` are both Apache-2.0.
- [x] **Round 3: `data/pseudo_boxes.py` written and CPU-tested.** OWLv2 over at most `--per-genus` (default 20) non-specimen images per genus from train/val/test. Boxes keep their best prompt, go through class-agnostic NMS at IoU 0.5 and are capped at 10 per image, with a full-frame fallback. OWLv2's padded-square boxes are mapped back to the original image, and the geometry was checked visually on portrait and landscape photos. Output is resumable in 2,048-image parts. Run it on the GPU through the queue, `python -m transferner.data.pseudo_boxes`, once Phase A's GPU work is done. CPU runs at 0.2 img/s. **Round 4: the GPU runs at 29–33 img/s** (after a device fix), so the full 27,017-image selection takes about 15 min. `drop_nested()` is the training-time cleanup.
- **Out of reach here:** the field video test set (D-018) needs the user's walks.
- [x] **2026-11-20 (interactive): labelling done, recipe fixed (D-033).** `manifests/boxes.parquet`: 132,302 photos (100 per genus), 199,073 boxes. `train/train_detector.py` (shared genus head, image aux loss, backbone lr ×0.03) and `export/detector_litert.py` (LiteRT fp32 / full INT8 PT2E, smoke-tested).
- [ ] **Full run `mnv4_shared`** started 15:51 (60 epochs, about 4.3 h; log `$TRANSFERNER_DATA/ops/logs/detector-mnv4_shared-1120-1551.log`, checkpoints `artifacts/detector/mnv4_shared/`, row in `results/tables/detector.csv`).
- [ ] **Then (D-011):** PTQ check on the trained model, `cd /tmp && PYTHONPATH=~/Transferner/src ~/Transferner/.venv-export/bin/python -m transferner.export.detector_litert --name mnv4_shared --recipe int8` (also `--act-observer minmax`, and `--recipe fp32`), 512 val photos. QAT (`quant.qat_epochs: 5`) is not written yet: PT2E `prepare_qat_pt2e` with the same quantizer, fine-tuned on the GPU, then the same check.

## Journal
STATUS: DONE
(Newest last. Each round: what was seen, decided and queued, and what the next round should do first.)

- **2026-11-19, interactive session (pre-launch):**
  - **Built:** `models/efficient.py`, `eval/efficiency.py`, `quant/sensitivity.py`, `quant/partial_search.py`, `scripts/q.sh`, `scripts/overnight.sh`.
  - **Smoke-tested** on 400 images per split: every variant type runs, and the custom forward matches stock open_clip at cosine 1.000000.
  - **Queued:** A1, A2 and A3 in that order, about 6.5 h of GPU time.
  - **Next round should:** check A1/A2 results as they land, adjust A2 if the partial search misses, and start Phase C in parallel.
- **2026-11-19, round 1 (02:35–03:15):**
  - **Found:**
    - `fp16` finished (rc 0, 19 min): test top-1 84.95%, top-5 96.45%, regional val 90.03%, val_ext 48.89%, 188 img/s, 1,206 MB, 334.6 GFLOPs.
    - **A2 targets:** regional val ≥ 0.8823 (the partial search's own goal); test ≥ 0.8325 to accept.
    - `w8a16` finished (rc 0): test 84.94% (Δ −0.01 pt, McNemar p 0.74), weights 606 MB. Weight-only INT8 is free.
    - No failed jobs.
  - **Phase C (built this round, D-027):**
    - The trait table came from the NC State toolbox. FDA PLANTOX is gone; USDA PLANTS returned 504 all round.
    - 2,714 cited rows over 872 species. Source pages cover 68% of test images.
    - DMR is now in `metrics.dmr`, `evaluate.py` and `efficiency_report`. Baseline (probe, no abstention): 5.45% contact, 4.38% ingestion. Zero-shot: 6.44% / 4.63%.
    - **Decided:** genus pages back species without their own page. Toxic rows apply as stated; edible rows become conditional.
  - **A5:** `efficiency_report.py` writes `results/EFFICIENCY.md` and `results/figures/efficiency_pareto.png`. It runs on the CPU in about 10 s, so rerun it each round.
  - **Phase B:** the diagnosis is in the plan section. Briefly: moss errors are cross-family, the training mix is 99% ToL, and the bryophyte selection slice is `val_ext_bryo`.
    - Added `linear_probe --w-field/--w-bryo/--extra-train` and `embed_images --phyla`. All CPU smoke-tested.
  - **Queue changes:**
    - `eval.evaluate` at the front: official `eval.csv` rows with DMR, about 2 min.
    - **Decided:** 7 Phase B jobs go after A1 and before A3: 5 probes, the unused-bryophyte embed (14.6K images) and its probe, about 30 min. This brings B's results in by about 05:00 instead of after A3, and delays A3 by 30 min.
    - The queue now runs until about 09:30, so Phase D will not run tonight (noted in the plan).
  - **Next round should first:**
    1. Check `done.tsv` for `quant.sensitivity` and `quant.partial_search`.
       - If the partial search missed, queue the next A2 step at the front: `quant.sensitivity --smooth 0.8 --tag s08`, then `quant.partial_search --smooth 0.8 --sens quant_sensitivity_s08.csv --prefix w8a8p`.
       - If it hit on test, write the A2 decision (the next free D number) and tick A2.
    2. Rerun `efficiency_report`.
    3. Read the `phaseB_*` rows in `results/tables/bioclip_probe.csv` and apply the D-026 rule on `probe_val_ext_bryo_top1` and `probe_val_regional_top1`. Only if a variant passes, retrain the production probe with it and record a decision.
    4. Plan A4 once A3 results land.
  - **Careful:** `linear_probe` defaults to `--out artifacts/probe.pt`, which is production. Every experiment needs its own `--out`.
- **2026-11-19, round 2 (04:44–05:30):**
  - **Found:**
    - All jobs since round 1 finished with rc 0: sensitivity, partial search, `w8a8dyn`, `w4a16_g128` and the Phase B probes.
    - **A2 passed at k = 0** (D-028): full static W8A8 with SmoothQuant 0.5. Test 83.52% against a threshold of 83.25%. The margin is thin, and contact DMR rises from 5.46% to 6.20%.
    - `w8a8dyn` (84.96%) and `w8a16` (84.94%) are lossless. `w4a16_g128` scores 84.37% at 315 MB.
    - Calibration was unseeded, so the same variant moved ±0.15 pt between runs. It's seeded now (commit 9650bac).
    - **k = 1** (`b1.fc1` kept in FP16) scores 88.79% on val, which buys nothing, so k = 4 was dropped.
  - **Phase B** (D-030): the field-photo weight helps; bryophyte weights and unused ToL mosses don't.
    - **Adopted `w_field: 8`.** Test: 84.96 → 85.32, mosses 68.5 → 69.3, liverworts 67.8 → 69.4.
    - Selection used val only: max regional val with ToL `val_rare` loss ≤ 0.1 pt. The disclosure is in D-030: test was printed for all nine runs.
    - The production retrain, `train.calibrate` and `eval.evaluate` are queued at the front. The old head is kept as `artifacts/probe_d026.pt`.
  - **Efficiency sweep head:** decoupled from production. Every production-head efficiency row is scored with the frozen D-026 head in `artifacts/efficiency/_head/`.
  - **A6 export** (D-029), CPU only, in `.venv-export` (litert-torch 0.10, torch 2.14.1 CPU, jax < 0.12.1):
    - FP16, W8 and W8 + dynamic INT8 convert and match PyTorch (cosine ≥ 0.998).
    - **The NPU graph must quantize only the linears.** The stock PT2E full-INT8 recipe quantizes residual adds and breaks the model (cosine 0.25). Linear-only static INT8 with SmoothQuant 0.5 gives cosine 0.975 and 64/64 agreement.
    - Peak conversion RSS is 14–17 GB, so never run two conversions at once.
  - **Phase C:** no second source exists yet.
    - USDA still returns 504.
    - cbif.gc.ca fails TLS verification; it was skipped, not bypassed.
    - CalPoison and Clemson have no structured lists.
    - **Decided:** don't scrape narrative pages into a safety table. Retry USDA once per round.
  - **Queue changes:**
    - Phase B went first, followed by the field-weight sweep (2–32) and `field3+unused`.
    - k1 ran; k4 was replaced by `w8a8_s0.5_p99.9` and `_p99.5` (val-only).
    - **Correction (05:26):** p99.9 collapsed to 59.9% regional val because outlier channels matter. p99.5 was replaced with `w8a8_s0.5_p100` (min/max ranges). The PT2E histogram observer reaches cosine 0.974 against FP32, where the simulation's 99.99-percentile ranges reach 0.944, so tighter clipping may recover the margin.
    - A3 was reordered: ToMe and resolution first, then `w4a16_g32` and `w4a8dyn_g32`, then `base_subset`, `drop2`, `drop4`.
    - The queue should run until about 09:20.
  - **Next round should first:**
    1. *(Already verified in round 2: the production retrain, `calibrate` and `evaluate` succeeded. Test 85.32%, ECE 0.0053, DMR 5.40% / 4.12%.)*
    2. **Read `w8a8_s0.5_p100`** (val-only; p99.9 already failed at 59.9%).
       - If either beats k0's regional val (88.75–88.90%) by ≥ 0.3 pt, queue it on test **ahead of the drop jobs**: `eval.efficiency --name w8a8_s0.5_pXX --w-bits 8 --a-bits 8 --a-mode static --smooth 0.5 --a-pct XX` without `--val-only`. This takes about 36 min. Reuse the same name: calibration is seeded, and the full row supersedes the val-only row in the report.
       - Then update D-028's recommended calibration.
       - If p100 is not better either, the next step is code: an MSE-optimal per-tensor range search in `QState.finish_range_calibration`, like PT2E's HistogramObserver, which reached cosine 0.974 on export. Test it val-only, and don't spend test runs on calibration variants that don't beat k0 on val.
    3. Rerun `efficiency_report` as the ToMe and resolution rows land, then plan A4 (val-only screens ahead of `base_subset`, `drop2`, `drop4`; cut `drop4` first).
    4. CPU work while the GPU is busy:
       - Retry USDA once.
       - Optionally, an x86 check of the deployable INT8 graph on more images: `export/litert.py --recipe int8static --smooth 0.5 --linear-only --images 256` takes about 6 min plus about 1 s per image.
       - Or start Phase D's `pseudo_boxes.py`, CPU-tested on a few images.
- **2026-11-19, round 3 (06:58–08:00):**
  - **Found:**
    - All jobs since round 2 finished with rc 0: `w8a8_s0.5_p100` (val-only), `tome_r2/r4/r6`, `res196`, `res168` and `w4a16_g32`.
    - **Min/max activation ranges fix static W8A8** (`w8a8_s0.5_p100`, val-only): regional val 90.10%, against FP16's 90.03% and k0's 88.75–88.90%. Cosine to FP16 is 0.9955 (k0: 0.944). Clipping, not rounding, was the loss: ViT-H's outlier channels carry information.
    - **Structure (A3, test, Δ vs FP16):** `tome_r2` −0.94 pt at −13% GFLOPs, `tome_r4` −2.71 (−26%), `tome_r6` −5.99 (−38%), `res196` −1.81 (−24%, and +31% GPU img/s, where ToMe gains little), `res168` −6.46 (−44%). Contact DMR rises faster than top-1 falls (`res196`: 5.46 → 6.40%).
    - `w4a16_g32`: 84.52% (−0.43 pt) at 343 MB, against g128's 84.37% at 315 MB.
  - **Export (A6), CPU:** `export/litert.py` gained `--act-observer minmax` and `--pt2e-only`. The real PT2E linear-only graph with SmoothQuant 0.5 (128 calibration and 128 check images) reaches cosine 0.9951 and 127/128 agreement with min/max, against 0.9746 and 125/128 with the default histogram observer. The simulation's gain carries over to the deployable graph.
  - **Test of `w8a8_s0.5_p100` (queued at the front this round): 84.82%** (−0.12 pt, rel 0.9985; D-028: 83.52%). Contact and ingestion DMR are 5.56% and 4.42% (FP16 5.46% / 4.41%). Recorded as **D-031**, with a pointer in D-028. The NPU recipe is now full W8A8 over the linears with min/max ranges, nothing in FP16, at a 1.57 pt margin.
  - **A5 decision written (D-032):**
    - LiteRT GPU: W8A16, with W4A16 g32 when memory is tight.
    - NPU: D-031.
    - XNNPACK CPU: 8da8w.
    - No structural cut by default, because a still has a 10 s budget. If one is ever needed, lower the resolution before using ToMe.
    - D-032 lists what needs the phone.
  - **Phase D (CPU):** wrote `data/pseudo_boxes.py`; licences for OWLv2 and MNv4 are Apache-2.0 (checked). On 64 images (one per genus), 25% fell back to full frame (mostly frame-filling mosses and dense herbs), images with boxes average 1.6, and the median box covers 29% of the image. The geometry is right. The weak spots are spurious boxes on dead leaves and single-leaflet boxes, and both carry the genus label. Before training, consider dropping boxes nested inside a larger box of the same image, or keeping only the largest box per prompt.
  - **Phase C:** USDA PLANTS now redirects to `plants.sc.egov.usda.gov`, which still returns 504. That was the one retry for this round.
  - **Queue changes:**
    - At the front: the `p100` test run, then the val-only A4 screens `w8a8_s0.5_p100_res196` and `w8a8_s0.5_p100_tome2`, then `res210` (a new A3 Pareto point at about −12% GFLOPs).
    - **Decided:** `w4a8dyn_g32` runs val-only. Dynamic INT8 activations are lossless on test (`w8a8dyn` vs `w8a16`: +0.02 pt), so 8da4w's test result follows from `w4a16_g32` plus the val check. The ~29 min saved lets `drop2` start before 10:00.
    - **Round timing:** `overnight.sh` measures its 90-minute round interval from the end of the previous round, so round 4 starts at about 09:30. Until 10:00 it can only influence `drop4`. That's why this round waited for the A4 screens and made the test-pick call itself.
    - **A4 screens (val-only)**, under a rule fixed before they landed (additive if regional val is within 0.2 pt of the float structure; then no test run):
      - `w8a8_s0.5_p100_res196`: 88.25% vs `res196` 88.25%.
      - `w8a8_s0.5_p100_tome2`: 89.23% vs `tome_r2` 89.23%.
      - Both are additive, so **no combo test runs**: test ≈ the structure's test row − 0.1 pt. The GPU time goes to `base_subset`, `drop2` and `drop4` instead, which are the real unknowns.
    - **Open signal:** both combos lose 0.75–1.0 pt of val_ext on top of their structure (46.33 vs 47.08; 46.73 vs 47.69), while static alone loses 0.03 pt. That's about 1 SE on 6.7K images. Test can't settle it, because test is regional, but `p100`'s rare-species test top-1 also dips 0.44 pt. If a structural fallback is ever shipped on the NPU, check val_ext with a different calibration seed first.
    - **Expected queue timeline:** `res210` to about 08:10, then `w4a8dyn_g32` val-only to about 08:20, then `base_subset` to about 08:56, then `drop2` to about 09:30, then `drop4` (starting about 09:30) to about 10:00.
  - **Next round should first:**
    1. Read `res210`, `w4a8dyn_g32` (val), `base_subset` and `drop2` (from about 09:25), and rerun `efficiency_report`.
    2. Add to D-032's structural part:
       - `res210` (test)
       - `drop2`/`drop4` (Δ against `base_subset`, not FP16)
       - the 8da4w val number.
       Then tick A1 and A3, and A4 (screens done, no test picks, see above).
    3. Don't add GPU jobs that can't start before 10:00. If the queue empties early, the only planned GPU work left is Phase D: `python -m transferner.data.pseudo_boxes`, about 30K images at 20 per genus. Throughput is unmeasured, so queue it only if it would start before about 09:30.
    4. After 10:00 everything left is phone work (D-032's list), Phase B's bryophyte field photos (needs the manifests free), Phase C's second source (USDA still 504), and Phase D training. Unless a job failed, write `STATUS: DONE` under "## Journal" with that summary.
- **2026-11-19, round 4 (09:25–10:02):**
  - **Found:**
    - All jobs since round 3 finished with rc 0: `res210`, `w4a8dyn_g32` (val-only), `base_subset`, `drop2` and `drop4`.
    - **`res210`:** test 84.15% (−0.80 pt) at −12% GFLOPs and +7% GPU img/s. DMR 5.68% / 4.67%. At `tome_r2`'s FLOPs it scores 0.14 pt more (paired McNemar p = 0.019) and keeps a plain static graph, so it is the first structural fallback in D-032.
    - **Block dropping is dominated** (Δ against `base_subset`, 83.22%):
      - `drop2`: −2.92 pt for −6% GFLOPs; contact DMR 6.16% → 8.63%.
      - `drop4`: −5.84 pt (77.39%) for −12.5% GFLOPs, the same saving as `res210` at seven times the loss; contact DMR 10.40%.
      - `res210` saves twice `drop2`'s FLOPs at about a quarter of the loss. Block dropping is off D-032's fallback list.
    - **8da4w** (`w4a8dyn_g32`, val-only): regional val 89.66% against `w4a16_g32`'s 89.63%. No test run (D-032).
    - **Round 3's open val_ext signal is real** (paired McNemar on the saved val_ext embeddings, CPU):
      - Static W8A8 on top of `res196`: −0.76 pt, p = 0.004.
      - On top of `tome_r2`: −0.97 pt, p = 0.001.
      - Static W8A8 alone (vs FP16): −0.03 pt, p = 0.95.
      - Regional val is identical to the structure's in both combos.
      - Calibration already runs through the structural model at the variant's resolution, and min/max ranges don't depend on the sampling seed. So this is an interaction, not calibration noise, and round 3's "different calibration seed" check would not separate it.
      - **Decided:** caveat in D-032 only. No structural cut is shipped by default, and no GPU time is left tonight.
  - **Code (CPU):**
    - `efficiency_report`: agreement and McNemar are now recomputed from saved predictions against each row's own reference. Before this, `drop2`'s "agree ref" was against `fp16` even though the row says `base_subset`. The `fp16`-referenced rows reproduce the CSV values exactly.
    - The Pareto figure now labels retrained heads and keeps crowded labels apart.
    - `pseudo_boxes.drop_nested()` is the training-time cleanup proposed in round 3: drop boxes ≥ 90% inside a larger box of the same image. On smoke64 it removes 19 of 77 boxes (13 are "a leaf"), boxes per image go from 1.60 to 1.21 and the median area from 0.29 to 0.60. `boxes.parquet` stays raw, and `main()` is unchanged.
  - **Phase C:** USDA PLANTS still returns 504 (`plants.sc.egov.usda.gov`), as of 09:31. That was this round's one retry.
  - **Side signal for Phase B (not acted on):**
    - `base_subset`'s balanced head (40 images per species, `w_field` 8) reaches val_ext 57.77%, against the D-030 production head's 51.91%. `val_ext_bryo` is 36.39% against 32.47%.
    - It costs 2.68 pt of regional val (88.82 vs 91.50), so it fails D-026 as it stands.
    - **Not a new lever.** It's the known trade-off between regional val and val_ext that D-025/D-026 already set. Logit adjustment is in the recipe (τ = 0.5), τ = 1 cost 0.8 pt of regional val, and rare-species shrinkage `shrink_b100_k30` reached val_ext 55.2% at regional val 89.35%. Under the D-026 rule (≤ 0.1 pt), none of these pass. Nothing queued.
    - The efficiency run's `val_rare` is a different, smaller set (zero-shot 31% vs 60.5%), so ignore that column when comparing.
  - **Queue:** no new long jobs. The only addition was a bounded OWLv2 throughput test, `pseudo_boxes --limit 1024` into `artifacts/pseudo_boxes_speed/`, after `drop4`. The plan said to measure it, and the full run was not to start after about 09:30.
    - Its first attempt (09:56) **crashed on the GPU path**: the DataLoader's image sizes were on the CPU and OWLv2's boxes on CUDA. Round 3's CPU smoke test couldn't see that.
    - Fixed in 347f8b1 (`boxes_for` moves the sizes to the logits' device) and requeued. It started at 09:58:17 and finished with rc 0.
    - **Throughput: 29–33 img/s.** The full default selection (20 per genus, train/val/test) is 27,017 images over 1,362 genera, **about 15 min of GPU**.
    - On these 1,024 images: fallback 28.8%, 1.80 boxes per image, median area 0.19. `drop_nested()` removes 323 of 1,310 boxes (211 are "a leaf"), giving 1.35 boxes per image and a median area of 0.35.
  - **Status:** Phase A is finished (A1–A6 ticked). B, C and D are blocked on things this loop can't do before the 10:00 stop:
    - B: the bryophyte field-photo pull rewrites the manifests and takes hours of download.
    - C: USDA is down, and there is no other structured source.
    - D: detector training is GPU-heavy, and the field video needs the user's walks.
    - D-032's phone list needs the Pixel.
    Hence `STATUS: DONE`.
  - **Next session should first:**
    1. Read D-031/D-032 and `results/EFFICIENCY.md`. Those are the efficiency answer. The phone work is D-032's "Needs the phone" list.
    2. Phase D: run `python -m transferner.data.pseudo_boxes` through the queue (about 15 min, 27K images), then write `train/train_detector.py` with `drop_nested()` at load.
    3. Phase B: the bryophyte field-photo pull (needs a `train_only` flag in `splits.py`), the one moss lever left untried.
    4. If an NPU build ever needs a structural cut: screen that exact static combo on val_ext first, and try calibrating on a pool that includes `train_ext` field photos.

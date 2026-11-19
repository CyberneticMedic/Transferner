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
- [ ] **A1 (queued):** references and weight-only / dynamic quantization:
  - `fp16`
  - `w8a16`
  - `w8a8dyn` (8da8w, XNNPACK-style)
  - `w4a16_g128`, `w4a16_g32`
  - `w4a8dyn_g32` (8da4w)
- [ ] **A2 (queued): partial static W8A8 for the NPU.**
  - Queued: `quant.sensitivity --smooth 0.5`, then `quant.partial_search --smooth 0.5`.
  - If no k meets the target on val, try these in order: smooth 0.8, then `--a-pct` 99.9 or 100, then larger k (up to all 128 layers is just W8A16).
  - **Gate:** once a partial W8A8 meets the target on test, quantization is done.
  - Then add D-027 recording the k, which layers stay FP16, and the test numbers.
- [ ] **A3 (queued): structural variants.**
  - `tome_r2`, `tome_r4`, `tome_r6`
  - `res196`, `res168`
  - `base_subset` (retrained head on 40 images per species), then `drop2` and `drop4`. A dropped-block variant must be compared with `base_subset`, not with the production head.
- [ ] **A4: combinations along the Pareto front.** At most about 6 jobs, chosen from A1–A3. Example: the accepted partial W8A8 with the best token-merging r or resolution, or W4A16 with token merging. For a static combination, recompute sensitivity only if the structure changes a lot.
- [ ] **A5: report.**
  - Add `src/transferner/eval/efficiency_report.py`, producing `results/EFFICIENCY.md` (a table of every variant: top-1/top-5 on test, Δ vs FP16, val, weight MB, GFLOPs) and `results/figures/efficiency_pareto.png` (test top-1 against GFLOPs and against weight MB). Load the `dataviz` skill before writing the chart code.
  - Add a decision entry recommending variants per runtime (LiteRT GPU, NPU, XNNPACK CPU), and state plainly what needs the phone: latency, memory and NPU compile.
- [ ] **A6 (optional, after A5): export feasibility on x86.**
  - Set up `.venv-export` (CPU torch) with ai-edge-torch and/or executorch.
  - Export the FP16 and W8 variants and check outputs against PyTorch on 64 images.
  - Record x86 latency only as a proxy.
  - Skip if the toolchain fights back for more than one round, and write down why.

### Phase B: mosses (when A2 is done; mostly CPU on cached embeddings)
- **Baseline (test, current probe):** Bryophyta 68.5% top-1 across 140 species; Marchantiophyta 67.8%; vascular plants 85.4%.
- **Diagnose:**
  - Are errors within the genus or across it?
  - What share of moss training images are herbarium specimens rather than field photos?
  - How do training counts per species compare?
  - How many ToL moss images are sitting in split `unused` (the per-species cap)? Those are not embedded yet; embedding them is a GPU job.
  - How good is the text init (the zero-shot share)?
- **Fix only what val supports:** select on the bryophyte slices of val and `val_ext`, under the D-026 rule.

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
  - Write a coverage report.
  - Add DMR (dangerous misidentification rate) to `eval/metrics.py` and `eval/evaluate.py`.
  - Add a decision entry.

### Phase D: live detector (GPU-heavy; only after Phase A's GPU work is finished)
- **Design (D-002/D-012):** SSDLite on MNv4-Conv-Medium, with boxes pseudo-labelled by OWLv2. The stubs are `src/transferner/data/pseudo_boxes.py` and `train/train_detector.py`.
- **Steps:**
  1. Pseudo-label a manageable subset of non-specimen images (specimens are excluded by D-020).
  2. Train.
  3. Report detection quality on a held-out subset.
- **Licences:** verify them. OWLv2 and timm's MNv4 weights should be Apache-2.0.
- **Packages:** install missing ones such as `transformers` and `timm` into `.venv` with `uv`.
- **Out of reach here:** the field video test set (D-018) needs the user's walks.

## Journal
(Newest last. Each round: what was seen, decided and queued, and what the next round should do first.)

- **2026-11-19, interactive session (pre-launch):**
  - **Built:** `models/efficient.py`, `eval/efficiency.py`, `quant/sensitivity.py`, `quant/partial_search.py`, `scripts/q.sh`, `scripts/overnight.sh`.
  - **Smoke-tested** on 400 images per split: every variant type runs, and the custom forward matches stock open_clip at cosine 1.000000.
  - **Queued:** A1, A2 and A3 in that order, about 6.5 h of GPU time.
  - **Next round should:** check A1/A2 results as they land, adjust A2 if the partial search misses, and start Phase C in parallel.

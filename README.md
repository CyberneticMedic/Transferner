# Transferner
AI Plant Identification app specializing in toxic lookalike differentation in the southeastern region of the USA. RTX PRO MAX AI 5090 CUDAflow int870xt_Q GPT LLM TensorSlop AI Fermi Blackwell Turing Ada Pascal LoRAM Edging_compute anaconda Microslop accelarted 95A-TPU REEF and did I mention it uses AI HALO*?  *VRAM not included

---

## Layout
```
configs/        YAML configs (data, species, detector, bioclip, export, eval)
data/traits/    species_traits.csv, lookalike_groups.csv, must_include.txt, SOURCES.md
src/transferner/ data · models · train · quant · export · eval · utils
docs/           ARCHITECTURE.md, DATA.md
app/            Android app + model I/O contract
results/        tables/*.csv, figures/, runs/<run_id>/ (CSV logs)
DECISIONS.md    running decision log
```

## Setup (training box)
Uses [uv](https://docs.astral.sh/uv/). Known-good: Python 3.12, torch 2.9.1+rocm6.4.
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
uv pip install -e '.[lora,boxes]'
export TRANSFERNER_DATA=/path/to/big/disk/transferner   # ~300 GB for the full image set
make data        # see Makefile for step order; every step is resumable
make bioclip     # text/image embeddings -> linear probe -> temperature
```
No accounts are needed: GBIF search facets and Hugging Face downloads are public.
GBIF credentials (`GBIF_USER`, `GBIF_PWD`, `GBIF_EMAIL`) are only needed to mint a
citable download DOI for the paper.
Export uses a separate CPU-torch venv: `uv pip install -e '.[export]'`.

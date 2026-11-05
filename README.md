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
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm<your ROCm version>
pip install -e '.[lora,boxes]'
export TRANSFERNER_DATA=/path/to/big/disk/transferner
make data        # see Makefile for step order
```
Export uses a separate CPU-torch venv: `pip install -e '.[export]'`.

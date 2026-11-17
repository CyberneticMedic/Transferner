"""BioCLIP 2.5 Huge wrapper.

- load(): open_clip.create_model_and_transforms(hf_hub) -> (model, preprocess_val, tokenizer).
- embed_text(species_df) -> [N, D] normalized, template-averaged (configs/bioclip.yaml).
- ImageClassifier(encoder, W, b, logit_scale): W initialised from text embeddings (zero-shot)
  or a trained linear probe; the same graph ships on-device either way.
- Token merging (ToMe) and block dropping hooks for the efficiency sweep: not yet implemented."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.config import load as load_cfg


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible to torch. Run `python ~/ml-tools/ml_doctor.py`.")
    return torch.device("cuda:0")


def load(dev: torch.device | None = None):
    import open_clip
    hub = load_cfg("bioclip")["model"]["hf_hub"]
    model, _, preprocess = open_clip.create_model_and_transforms(hub)
    tokenizer = open_clip.get_tokenizer(hub)
    model = model.eval().to(dev or device())
    return model, preprocess, tokenizer


def prompts(row) -> dict[str, str]:
    taxo = " ".join(str(x) for x in ["Plantae", row.phylum, row["class"], row.order, row.family,
                                      row.scientific_name] if isinstance(x, str) and x)
    common = row.common_name if isinstance(row.common_name, str) and row.common_name else row.scientific_name
    return {"taxonomic": taxo, "scientific": row.scientific_name, "common": common}


@torch.no_grad()
def embed_text(species: pd.DataFrame, model, tokenizer, batch: int = 256) -> np.ndarray:
    """[N, D] float32, L2-normalized: mean of normalized per-template embeddings, renormalized."""
    cfg = load_cfg("bioclip")["text"]
    dev = next(model.parameters()).device
    out = []
    rows = [prompts(r) for _, r in species.iterrows()]
    for tmpl in cfg["templates"]:
        texts = [tmpl.format(**p) for p in rows]
        embs = []
        for i in range(0, len(texts), batch):
            tok = tokenizer(texts[i:i + batch]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = model.encode_text(tok)
            embs.append(F.normalize(e.float(), dim=-1).cpu())
        out.append(torch.cat(embs))
    e = torch.stack(out).mean(0)
    return F.normalize(e, dim=-1).numpy()


class ImageClassifier(nn.Module):
    """image -> normalized embedding -> logits = scale * (e @ W.T) + b. W rows are per-class."""

    def __init__(self, visual: nn.Module, W: torch.Tensor, b: torch.Tensor | None, logit_scale: float):
        super().__init__()
        self.visual = visual
        self.W = nn.Parameter(W.clone(), requires_grad=False)
        self.b = nn.Parameter(b.clone() if b is not None else torch.zeros(W.shape[0]), requires_grad=False)
        self.scale = float(logit_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = F.normalize(self.visual(x), dim=-1)
        return self.scale * e @ self.W.T + self.b

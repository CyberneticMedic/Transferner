"""Fit temperature T on val logits; save T alongside each model artifact.

Uses artifacts/probe.pt on cached val embeddings; reports NLL and 15-bin ECE before/after.
Writes artifacts/temperature.txt."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..utils.config import data_root


def ece(logits: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    p = logits.softmax(-1)
    conf, pred = p.max(-1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1, device=logits.device)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return float(tot)


def main() -> None:
    root = data_root()
    pr = torch.load(root / "artifacts/probe.pt")
    W, b, s = pr["W"].cuda(), pr["b"].cuda(), pr["scale"]
    e = torch.from_numpy(np.load(root / "artifacts/img_emb/val.npy")).cuda().float()
    y = torch.from_numpy(pd.read_parquet(root / "artifacts/img_emb/val.parquet").class_idx.to_numpy()).cuda()
    with torch.no_grad():
        logits = s * e @ W.T + b
    logT = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = logT.exp().item()
    print(f"T = {T:.4f} | NLL {F.cross_entropy(logits, y):.4f} -> {F.cross_entropy(logits / T, y):.4f} "
          f"| ECE {ece(logits, y):.4f} -> {ece(logits / T, y):.4f}")
    (root / "artifacts/temperature.txt").write_text(f"{T:.6f}\n")


if __name__ == "__main__":
    main()

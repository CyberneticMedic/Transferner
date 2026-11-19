"""Fit temperature T on val logits; save T alongside each model artifact.

Uses artifacts/probe.pt on cached embeddings. T is fitted on the regional slice of val by
default (the deployment distribution; ToL val images were in BioCLIP's pretraining and are
over-confidently right). Reports NLL, accuracy and 15-bin ECE before/after on regional val,
all of val, and test (T never sees test). Logits are built in chunks: full val is
~280k x 4411 classes, which with autograd buffers does not fit in 24 GiB at once.
Writes artifacts/temperature.txt."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..utils.config import data_root

CHUNK = 32768


def load(emb, split, source=None):
    meta = pd.read_parquet(emb / f"{split}.parquet")
    e = torch.from_numpy(np.load(emb / f"{split}.npy"))
    y = torch.tensor(meta.class_idx.to_numpy())
    if source:
        m = torch.tensor((meta.source == source).to_numpy())
        e, y = e[m], y[m]
    return e.cuda(), y.cuda()


@torch.no_grad()
def stats(logits_fn, e, y, T=1.0, bins=15):
    """NLL, top-1 accuracy and ECE of softmax(logits / T), computed chunkwise."""
    nll, confs, hits = 0.0, [], []
    for i in range(0, len(e), CHUNK):
        lg = logits_fn(e[i:i + CHUNK]) / T
        nll += F.cross_entropy(lg, y[i:i + CHUNK], reduction="sum").item()
        c, p = lg.softmax(-1).max(-1)
        confs.append(c)
        hits.append((p == y[i:i + CHUNK]).float())
    conf, acc = torch.cat(confs), torch.cat(hits)
    edges = torch.linspace(0, 1, bins + 1, device=conf.device)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * (conf[m].mean() - acc[m].mean()).abs().item()
    return {"nll": nll / len(e), "acc": acc.mean().item(), "ece": ece}


def fit_temperature(logits_fn, e, y) -> float:
    logT = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        tot = 0.0
        for i in range(0, len(e), CHUNK):           # accumulate grad chunk by chunk
            with torch.no_grad():
                lg = logits_fn(e[i:i + CHUNK])
            loss = F.cross_entropy(lg / logT.exp(), y[i:i + CHUNK], reduction="sum") / len(e)
            loss.backward()
            tot += loss.item()
        return torch.tensor(tot)

    opt.step(closure)
    return logT.exp().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="artifacts/img_emb")
    ap.add_argument("--probe", default="artifacts/probe.pt")
    ap.add_argument("--fit-on", choices=["val_regional", "val"], default="val_regional")
    ap.add_argument("--out", default="artifacts/temperature.txt")
    args = ap.parse_args()
    root = data_root()
    emb = root / args.emb
    pr = torch.load(root / args.probe)
    W, b, s = pr["W"].cuda(), pr["b"].cuda(), pr["scale"]
    logits_fn = lambda x: s * x.float() @ W.T + b                          # noqa: E731

    sets = {"val_regional": load(emb, "val", "gbif_regional"), "val": load(emb, "val"),
            "test": load(emb, "test")}
    if (emb / "val_ext.npy").exists():                  # rare species, out-of-region (D-026); report only
        sets["val_ext"] = load(emb, "val_ext")
    T = fit_temperature(logits_fn, *sets[args.fit_on])
    print(f"T = {T:.4f} (fitted on {args.fit_on}, n={len(sets[args.fit_on][1]):,})")
    for name, (e, y) in sets.items():
        a, c = stats(logits_fn, e, y), stats(logits_fn, e, y, T)
        print(f"  {name:13s} n={len(y):>7,} acc {a['acc']:.4f} | NLL {a['nll']:.4f} -> {c['nll']:.4f} "
              f"| ECE {a['ece']:.4f} -> {c['ece']:.4f}")
    (root / args.out).write_text(f"{T:.6f}\n")


if __name__ == "__main__":
    main()

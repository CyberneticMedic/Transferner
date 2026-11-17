"""Train the D->N linear head on cached embeddings, initialised from text embeddings.

Logits = s * e @ W.T + b with W initialised to the text embeddings and s to BioCLIP's
logit scale, so epoch 0 *is* the zero-shot classifier. Classes flagged zero_shot_only
keep their text row frozen (no image gradient). Reports zero-shot and probe accuracy on
val and test; saves artifacts/probe.pt (W, b, scale) and logs via utils.csvlog."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..utils.config import load, data_root
from ..utils.csvlog import RunLog


def load_split(root, s):
    e = torch.from_numpy(np.load(root / f"artifacts/img_emb/{s}.npy"))
    y = torch.from_numpy(pd.read_parquet(root / f"artifacts/img_emb/{s}.parquet").class_idx.to_numpy())
    return e, y


@torch.no_grad()
def topk(logits_fn, e, y, ks=(1, 5), bs=65536):
    hits = {k: 0 for k in ks}
    for i in range(0, len(e), bs):
        lg = logits_fn(e[i:i + bs].cuda().float())
        top = lg.topk(max(ks), dim=-1).indices.cpu()
        for k in ks:
            hits[k] += (top[:, :k] == y[i:i + bs, None]).any(1).sum().item()
    return {f"top{k}": hits[k] / max(len(e), 1) for k in ks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--anchor", type=float, default=1e-2, help="L2 pull of W back to the text init")
    args = ap.parse_args()

    cfg = load("bioclip")["heads"]["linear_probe"]
    root = data_root()
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx")
    T = torch.from_numpy(np.load(root / "artifacts/text_emb.npy")).cuda()
    scale = float((root / "artifacts/logit_scale.txt").read_text())
    frozen = torch.from_numpy(sp.zero_shot_only.to_numpy()).cuda()

    tr_e, tr_y = load_split(root, "train")
    va_e, va_y = load_split(root, "val")
    te_e, te_y = load_split(root, "test")
    log = RunLog("linear_probe")

    zs = lambda e: scale * e @ T.T                                           # noqa: E731
    res = {"zs_val": topk(zs, va_e, va_y), "zs_test": topk(zs, te_e, te_y)}
    print("zero-shot", res)

    W = torch.nn.Parameter(T.clone() if cfg["init_from_text"] else torch.randn_like(T) * 0.01)
    b = torch.nn.Parameter(torch.zeros(len(T), device="cuda"))
    opt = torch.optim.AdamW([W, b], lr=cfg["lr"], weight_decay=args.wd)
    steps = cfg["epochs"] * ((len(tr_e) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.05)
    tr_e_gpu, tr_y_gpu = tr_e.cuda(), tr_y.cuda()
    probe = lambda e: scale * e @ F.normalize(W, dim=-1).T + b              # noqa: E731

    best, best_state = -1.0, None
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(tr_e_gpu), device="cuda")
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = F.cross_entropy(probe(tr_e_gpu[idx].float()), tr_y_gpu[idx])
            loss = loss + args.anchor * (W - T).pow(2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            W.grad[frozen] = 0
            opt.step()
            sched.step()
            with torch.no_grad():
                W.data[frozen] = T[frozen]
            tot += loss.item() * len(idx)
        v = topk(probe, va_e, va_y)
        log.log(ep, "train", loss=tot / len(tr_e))
        log.log(ep, "val", **v)
        print(f"epoch {ep:2d} loss {tot / len(tr_e):.4f} val {v}")
        if v["top1"] > best:
            best, best_state = v["top1"], (W.detach().clone(), b.detach().clone())

    W.data, b.data = best_state
    res["probe_val"], res["probe_test"] = topk(probe, va_e, va_y), topk(probe, te_e, te_y)
    print("probe", res)
    torch.save({"W": F.normalize(W.detach(), dim=-1).cpu(), "b": b.detach().cpu(), "scale": scale},
               root / "artifacts/probe.pt")
    log.summary("bioclip_probe", n_classes=len(T), n_train=len(tr_e),
                **{f"{k}_{m}": v for k, d in res.items() for m, v in d.items()})
    log.close()


if __name__ == "__main__":
    main()

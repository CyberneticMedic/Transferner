"""Train the D->N linear head on cached embeddings, initialised from text embeddings.

Logits = s * e @ W.T + b with W initialised to the text embeddings and s to BioCLIP's
logit scale, so epoch 0 *is* the zero-shot classifier. Classes flagged zero_shot_only
keep their text row frozen (no image gradient). Reports zero-shot and probe accuracy on
val, regional val and test; saves artifacts/probe.pt (W, b, scale) and logs via utils.csvlog.

Checkpoint selection defaults to the *regional* slice of val (recent iNat/Pl@ntNet photos,
the deployment distribution). Most of val is TreeOfLife, which BioCLIP was pretrained on,
so whole-val accuracy rewards memorising ToL and picks overfit checkpoints. Test is never
used for selection."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..utils.config import load, data_root
from ..utils.csvlog import RunLog

REGIONAL = "gbif_regional"


def load_split(emb, s):
    meta = pd.read_parquet(emb / f"{s}.parquet")
    e = torch.from_numpy(np.load(emb / f"{s}.npy"))
    y = torch.tensor(meta.class_idx.to_numpy())
    return e, y, torch.tensor((meta.source == REGIONAL).to_numpy())


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
    ap.add_argument("--lr", type=float, default=None, help="override configs/bioclip.yaml")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--anchor", type=float, default=None, help="L2 pull of W back to the text init")
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override configs/bioclip.yaml")
    ap.add_argument("--select", choices=["val", "val_regional"], default="val_regional")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb", default="artifacts/img_emb", help="embedding dir, relative to TRANSFERNER_DATA")
    ap.add_argument("--out", default="artifacts/probe.pt")
    ap.add_argument("--tag", default="", help="free-text label stored in the summary row")
    args = ap.parse_args()

    cfg = dict(load("bioclip")["heads"]["linear_probe"])
    for k in ("epochs", "lr", "anchor", "label_smoothing"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg.setdefault("anchor", 1e-2)
    cfg.setdefault("label_smoothing", 0.0)
    torch.manual_seed(args.seed)
    root = data_root()
    emb = root / args.emb
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx")
    T = torch.from_numpy(np.load(root / "artifacts/text_emb.npy")).cuda()
    scale = float((root / "artifacts/logit_scale.txt").read_text())
    frozen = torch.tensor(sp.zero_shot_only.to_numpy()).cuda()

    tr_e, tr_y, _ = load_split(emb, "train")
    va_e, va_y, va_reg = load_split(emb, "val")
    te_e, te_y, _ = load_split(emb, "test")
    vr_e, vr_y = va_e[va_reg], va_y[va_reg]
    log = RunLog("linear_probe")

    def evaluate(fn, prefix, test=False):
        out = {f"{prefix}_val": topk(fn, va_e, va_y), f"{prefix}_val_regional": topk(fn, vr_e, vr_y)}
        if test:
            out[f"{prefix}_test"] = topk(fn, te_e, te_y)
        return out

    zs = lambda e: scale * e @ T.T                                           # noqa: E731
    res = evaluate(zs, "zs", test=True)
    print("zero-shot", res)

    W = torch.nn.Parameter(T.clone() if cfg["init_from_text"] else torch.randn_like(T) * 0.01)
    b = torch.nn.Parameter(torch.zeros(len(T), device="cuda"))
    opt = torch.optim.AdamW([W, b], lr=cfg["lr"], weight_decay=args.wd)
    steps = cfg["epochs"] * ((len(tr_e) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.05)
    tr_e_gpu, tr_y_gpu = tr_e.cuda(), tr_y.cuda()
    probe = lambda e: scale * e @ F.normalize(W, dim=-1).T + b              # noqa: E731

    best, best_ep, best_state = -1.0, -1, None
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(len(tr_e_gpu), device="cuda")
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = F.cross_entropy(probe(tr_e_gpu[idx].float()), tr_y_gpu[idx],
                                   label_smoothing=cfg["label_smoothing"])
            loss = loss + cfg["anchor"] * (W - T).pow(2).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            W.grad[frozen] = 0
            opt.step()
            sched.step()
            with torch.no_grad():
                W.data[frozen] = T[frozen]
            tot += loss.item() * len(idx)
        v = evaluate(probe, "p")
        log.log(ep, "train", loss=tot / len(tr_e))
        for k, d in v.items():
            log.log(ep, k[2:], **d)
        score = v[f"p_{args.select}"]["top1"]
        print(f"epoch {ep:2d} loss {tot / len(tr_e):.4f} val {v['p_val']['top1']:.4f} "
              f"val_regional {v['p_val_regional']['top1']:.4f}", flush=True)
        if score > best:
            best, best_ep, best_state = score, ep, (W.detach().clone(), b.detach().clone())

    W.data, b.data = best_state
    res.update(evaluate(probe, "probe", test=True))
    print(f"best epoch {best_ep} (by {args.select})", res)
    torch.save({"W": F.normalize(W.detach(), dim=-1).cpu(), "b": b.detach().cpu(), "scale": scale},
               root / args.out)
    log.summary("bioclip_probe", n_classes=len(T), n_train=len(tr_e),
                **{f"{k}_{m}": v for k, d in res.items() for m, v in d.items()},
                tag=args.tag, lr=cfg["lr"], anchor=cfg["anchor"], wd=args.wd,
                label_smoothing=cfg["label_smoothing"], epochs=cfg["epochs"], best_epoch=best_ep,
                select=args.select, out=args.out)
    log.close()


if __name__ == "__main__":
    main()

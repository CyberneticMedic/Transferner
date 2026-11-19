"""Run a model artifact over the test split and append to results/tables/eval.csv.

BioCLIP heads run on cached embeddings (artifacts/img_emb): zero-shot text classifier
(uncalibrated) and the linear probe with its fitted temperature. Metrics per species group
(phylum, training-image count, zero-shot-only, look-alike, toxic, must-include); top-1 CIs
resample whole observations. McNemar compares probe vs zero-shot on the same images."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from ..utils.config import data_root
from ..utils.csvlog import RunLog
from . import metrics as M

CHUNK = 32768
TRAIN_BINS = [(0, 20, "train<20"), (20, 100, "train 20-99"), (100, 400, "train 100-399"),
              (400, 10**9, "train>=400")]


@torch.no_grad()
def predict(logits_fn, e: torch.Tensor, T: float = 1.0, k: int = 5):
    tops, confs = [], []
    for i in range(0, len(e), CHUNK):
        lg = logits_fn(e[i:i + CHUNK].cuda().float()) / T
        p = lg.softmax(-1)
        tops.append(p.topk(k, dim=-1).indices.cpu())
        confs.append(p.max(-1).values.cpu())
    return torch.cat(tops).numpy(), torch.cat(confs).numpy()


def group_masks(sp: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    s = sp.set_index("class_idx").loc[y]
    g = {"all": np.ones(len(y), bool)}
    for ph in sp.phylum.unique():
        g[ph] = (s.phylum == ph).to_numpy()
    for lo, hi, name in TRAIN_BINS:
        g[name] = ((s.n_train >= lo) & (s.n_train < hi)).to_numpy()
    for col, name in [("zero_shot_only", "zero_shot_only"), ("is_lookalike", "lookalike"),
                      ("is_toxic", "toxic"), ("must_include", "must_include")]:
        g[name] = s[col].to_numpy().astype(bool)
    return g


def score(top, conf, y, obs, genus, family, mask) -> dict:
    t, c, yy = top[mask], conf[mask], y[mask]
    hit1 = t[:, 0] == yy
    rec, f1 = M.per_class_f1(t[:, 0], yy)
    lo, hi = M.cluster_bootstrap_ci(hit1, obs[mask])
    return {"n_img": int(mask.sum()), "n_species": int(len(np.unique(yy))),
            "top1": hit1.mean(), "top1_lo": lo, "top1_hi": hi,
            "top5": (t == yy[:, None]).any(1).mean(), "macro_recall": rec, "macro_f1": f1,
            "genus_top1": (genus[t[:, 0]] == genus[yy]).mean(),
            "family_top1": (family[t[:, 0]] == family[yy]).mean(),
            "ece": M.ece(c, hit1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="artifacts/img_emb")
    ap.add_argument("--split", default="test")
    ap.add_argument("--probe", default="artifacts/probe.pt")
    ap.add_argument("--temperature", default="artifacts/temperature.txt")
    ap.add_argument("--name", default="probe", help="label for the probe model in eval.csv")
    args = ap.parse_args()
    root = data_root()
    emb = root / args.emb

    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx").reset_index(drop=True)
    meta = pd.read_parquet(emb / f"{args.split}.parquet")
    obs = meta[["uuid"]].merge(pd.read_parquet(root / "manifests/splits.parquet", columns=["uuid", "observation_id"]),
                               on="uuid", how="left").observation_id.fillna(meta.uuid).astype(str).to_numpy()
    e = torch.from_numpy(np.load(emb / f"{args.split}.npy"))
    y = meta.class_idx.to_numpy()
    genus, family = sp.genus.to_numpy(), sp.family.to_numpy()
    lookalike = {int(r.class_idx): set(r.lookalike_groups) for r in sp[sp.is_lookalike].itertuples()}

    text = torch.from_numpy(np.load(root / "artifacts/text_emb.npy")).cuda()
    scale = float((root / "artifacts/logit_scale.txt").read_text())
    pr = torch.load(root / args.probe)
    W, b = pr["W"].cuda(), pr["b"].cuda()
    T = float((root / args.temperature).read_text())
    models = {"zero_shot": predict(lambda x: scale * x @ text.T, e),
              args.name: predict(lambda x: pr["scale"] * x @ W.T + b, e, T)}

    masks = group_masks(sp, y)
    log = RunLog("evaluate")
    rows = []
    for name, (top, conf) in models.items():
        for g, m in masks.items():
            if m.any():
                r = {"model": name, "split": args.split, "group": g,
                     **score(top, conf, y, obs, genus, family, m)}
                rows.append(r)
                log.summary("eval", **r, probe=args.probe if name != "zero_shot" else "",
                            temperature=T if name != "zero_shot" else 1.0)
        la = M.lookalike_confusion(top[:, 0], y, lookalike)
        print(f"{name}: look-alike species n={la['n']:,} err {la['err']:.3f}, "
              f"{la['share_of_errors_within_group']:.1%} of errors stay inside the look-alike group")
    log.close()

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.pivot(index="group", columns="model", values="top1").join(
        df[df.model == "zero_shot"].set_index("group")[["n_img", "n_species"]])
        .loc[[g for g in masks if g in set(df.group)]].round(4).to_string())
    print(df[df.group == "all"].set_index("model")[
        ["top1", "top1_lo", "top1_hi", "top5", "macro_recall", "macro_f1", "genus_top1", "family_top1", "ece"]]
        .round(4).to_string())
    a = models["zero_shot"][0][:, 0] == y
    b_ = models[args.name][0][:, 0] == y
    mc = M.mcnemar(a, b_)
    print(f"McNemar zero_shot vs {args.name}: only zero-shot right {mc['a_only']:,}, "
          f"only {args.name} right {mc['b_only']:,}, p = {mc['p']:.2e}")


if __name__ == "__main__":
    main()

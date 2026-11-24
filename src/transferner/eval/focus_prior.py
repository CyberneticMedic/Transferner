"""Live "focus" chip (D-037): the user says what kind of plant they are looking at, and the live genus readout
is re-weighted toward genera that can take that form. One model, no specialists (the measured case against
specialist live models is in D-037).

Categories per genus come from USDA PLANTS growth habits (data/traits/growth_habit.csv, cited) and taxonomy:
  tree (Tree), shrub (Shrub, Subshrub), vine (Vine), herb (Forb/herb, not a fern or moss), grass-like
  (Graminoid), fern (Polypodiopsida, Lycopodiopsida), moss (Bryophyta, Marchantiophyta, Anthocerotophyta).
A genus is in a category if ANY of its label-set species can take that form (poison ivy is herb, shrub,
subshrub and vine). A genus with no habit data is never down-weighted.
Prior: logits + log(eps) for genera outside the chosen category (soft: eps > 0 keeps a strongly seen
out-of-category genus reachable, so a look-alike never vanishes); eps = 0 would be a hard mask.

Evaluated on cached whole-frame logits (eval/detector_prior.py writes them; --logits picks the model):
  right focus   the user picks a category the photo's genus can take (each such category, averaged)
  wrong focus   the user picks a category it cannot take (each, averaged): the harm side
Selection on the regional slice of val, test reported. Writes focus.json for the app bundle (the chosen eps).
  python -m transferner.eval.focus_prior [--name mnv4_live]"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..utils.config import data_root

CATS = [("tree", "Tree"), ("shrub", "Shrub"), ("vine", "Vine"), ("herb", "Herb"), ("grass", "Grass-like"),
        ("fern", "Fern"), ("moss", "Moss")]
FERN = {"Polypodiopsida", "Lycopodiopsida"}
MOSS = {"Bryophyta", "Marchantiophyta", "Anthocerotophyta"}


def genus_table(root, repo, genera: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """-> (len(genera), len(CATS)) membership, and a per-genus 'has habit data' flag."""
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["scientific_name", "genus", "phylum", "class"])
    gh = pd.read_csv(repo / "data/traits/growth_habit.csv")
    hab = gh.set_index("scientific_name").growth_habits.fillna("").astype(str).to_dict()
    ci = {c: i for i, (c, _) in enumerate(CATS)}
    gi = {g: i for i, g in enumerate(genera)}
    M = np.zeros((len(genera), len(CATS)), bool)
    known = np.zeros(len(genera), bool)
    for r in sp.itertuples(index=False):
        if r.genus not in gi:
            continue
        g = gi[r.genus]
        if r.phylum in MOSS:
            M[g, ci["moss"]] = known[g] = True
            continue
        if r[3] in FERN:                                                # r[3] = class (a keyword, so positional)
            M[g, ci["fern"]] = known[g] = True
            continue
        h = set(filter(None, hab.get(r.scientific_name, "").split(";")))
        if not h:
            continue
        known[g] = True
        if "Tree" in h: M[g, ci["tree"]] = True
        if h & {"Shrub", "Subshrub"}: M[g, ci["shrub"]] = True
        if "Vine" in h: M[g, ci["vine"]] = True
        if "Forb/herb" in h: M[g, ci["herb"]] = True
        if "Graminoid" in h: M[g, ci["grass"]] = True
        if "Nonvascular" in h: M[g, ci["moss"]] = True
    return M, known


def topk_acc(lg, y, k=1):
    top = np.argpartition(-lg, k, 1)[:, :k] if k > 1 else lg.argmax(1)[:, None]
    return (top == y[:, None]).any(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_live")
    args = ap.parse_args()
    root = data_root()
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    d = root / "artifacts/detector" / args.name
    genera = json.loads((d / "labels.json").read_text())["genera"]
    z = np.load(d / "image_logits.npz", allow_pickle=True)
    M, known = genus_table(root, repo, genera)
    print(f"{len(genera)} genera; habit data for {known.sum()}; per category: " +
          ", ".join(f"{c} {M[:, i].sum()}" for i, (c, _) in enumerate(CATS)), flush=True)
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["path", "source"]).set_index("path").source
    rows = []
    for split in ("val", "test"):
        lg, y, paths = z[f"{split}_logits"].astype(np.float32), z[f"{split}_y"], z[f"{split}_path"]
        reg = (im.reindex(paths) == "gbif_regional").to_numpy()
        lg, y = lg[reg], y[reg]
        lg = lg - lg.max(1, keepdims=True)
        base = topk_acc(lg, y).mean()
        for eps in (1.0, 0.3, 0.1, 0.03, 0.01, 0.0):
            right, wrong, n_r, n_w = 0.0, 0.0, 0, 0
            r5 = 0.0
            for c in range(len(CATS)):
                inside = M[:, c] | ~known
                bias = np.where(inside, 0.0, np.log(eps) if eps > 0 else -1e9).astype(np.float32)
                adj = lg + bias
                hit1 = topk_acc(adj, y)
                can = M[y, c]                                            # photos whose genus can take this form
                cannot = known[y] & ~M[y, c]
                right += hit1[can].sum(); n_r += can.sum()
                r5 += topk_acc(adj[can], y[can], 5).sum()
                wrong += hit1[cannot].sum(); n_w += cannot.sum()
            rows.append({"split": split, "eps": eps, "n": len(y), "no_focus_top1": round(float(base), 4),
                         "right_focus_top1": round(right / max(n_r, 1), 4), "right_focus_top5": round(r5 / max(n_r, 1), 4),
                         "wrong_focus_top1": round(wrong / max(n_w, 1), 4)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # choose eps on val: the largest right-focus gain whose wrong-focus top-1 keeps at least half the no-focus top-1
    v = df[df.split == "val"]
    ok = v[v.wrong_focus_top1 >= 0.5 * v.no_focus_top1]
    eps = float(ok.sort_values("right_focus_top1").iloc[-1].eps) if len(ok) else 1.0
    out = {"eps": eps, "source": "USDA PLANTS growth habit (plants.usda.gov), data/traits/growth_habit.csv; taxonomy for ferns and mosses",
           "rule": "genus in category if any label-set species can take the form; genera without habit data are never down-weighted",
           "categories": [{"id": c, "label": lab, "genera": [int(g) for g in np.where(M[:, i] | ~known)[0]]} for i, (c, lab) in enumerate(CATS)],
           "check": df.to_dict(orient="records")}
    (d / "focus.json").write_text(json.dumps(out))
    print(f"chosen eps {eps} (val); wrote {d / 'focus.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Paired photo bootstrap of class-agnostic box AP50 between two live checkpoints (D-049).

Reads the detections that `train_live_head --eval-only <ckpt> --boxes-only --ignore-boxes ... --dump-dets <npz>`
saves for the real-photo and scene test sets, with and without the box check's rejects as ignore regions, raw and
under the app's box rules. Matching at IoU 0.5 is per photo (coco_map's greedy rule), so it is done once and a
resample only reweights photos; the full-sample AP is checked against coco_map. Both checkpoints are resampled
with the same photo weights, so the interval is for B - A on this test set (not for run-to-run training noise).

Also splits each model's false boxes by where they fall: inside a target box (a part of the plant), straddling
one, or away from every target.
  python -m transferner.eval.box_bootstrap ops/dets_d046.npz ops/dets_d046b.npz"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..utils.config import data_root
from .metrics import box_iou_np, coco_map

REC = np.linspace(0, 1, 101)


def load(path: str) -> dict:
    p = Path(path)
    z = np.load(p if p.is_absolute() else data_root() / p)
    out = {}
    for f in z.files:
        key, field = f.split("__")
        out.setdefault(key, {})[field] = z[f]
    return out


def match50(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per detection: matched a target at IoU >= 0.5 (coco_map's rule: per photo, by score, the unmatched target
    with the highest IoU), and the largest fraction of it inside any target of its photo."""
    di, ds, db, gi, gb = d["img"], d["score"], d["box"], d["gt_img"], d["gt_box"]
    tp, ioa = np.zeros(len(di), bool), np.zeros(len(di))
    order = np.lexsort((-ds, di))
    d_imgs, d_start = np.unique(di[order], return_index=True)
    d_end = np.append(d_start[1:], len(order))
    g_ord = np.argsort(gi, kind="stable")
    g_imgs, g_start = np.unique(gi[g_ord], return_index=True)
    g_pos = dict(zip(g_imgs.tolist(), zip(g_start.tolist(), np.append(g_start[1:], len(g_ord)).tolist())))
    for im, s, e in zip(d_imgs.tolist(), d_start.tolist(), d_end.tolist()):
        if im not in g_pos:
            continue
        k_, g = order[s:e], g_ord[slice(*g_pos[im])]
        b, t = db[k_], gb[g]
        iou = box_iou_np(b, t)
        ix = np.clip(np.minimum(b[:, None, 2], t[:, 2]) - np.maximum(b[:, None, 0], t[:, 0]), 0, None)
        iy = np.clip(np.minimum(b[:, None, 3], t[:, 3]) - np.maximum(b[:, None, 1], t[:, 1]), 0, None)
        ioa[k_] = (ix * iy / np.clip((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6, None)[:, None]).max(1)
        used = np.zeros(len(g), bool)
        for k in np.flatnonzero(iou.max(1) >= 0.5):
            row = np.where(used, -1.0, iou[k])
            j = int(row.argmax())
            if row[j] >= 0.5:
                used[j] = True
                tp[k_[k]] = True
    return tp, ioa


def ap50(img: np.ndarray, tp: np.ndarray, gt_img: np.ndarray, w: np.ndarray) -> float:
    """AP50 of score-sorted detections with photo weights w (a bootstrap resample's photo counts)."""
    dw = w[img]
    tps, fps = np.cumsum(dw * tp), np.cumsum(dw * ~tp)
    rc, pr = tps / w[gt_img].sum(), tps / (tps + fps + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    pi = np.searchsorted(rc, REC, side="left")
    return float(np.where(pi < len(pr), pr[np.minimum(pi, len(pr) - 1)], 0.0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="--dump-dets npz of checkpoint A (the baseline)")
    ap.add_argument("b", help="--dump-dets npz of checkpoint B")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--where", type=float, nargs="*", default=[0.25, 0.5], help="objectness thresholds for the false-box split")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    rng = np.random.default_rng(0)
    for key in sorted(set(A) & set(B)):
        a, b = A[key], B[key]
        assert np.array_equal(a["gt_img"], b["gt_img"]) and np.array_equal(a["gt_box"], b["gt_box"]), f"{key}: different targets"
        n_img = int(max(a["img"].max(initial=0), b["img"].max(initial=0), a["gt_img"].max()) + 1)
        prep = []
        for d in (a, b):
            tp, ioa = match50(d)
            ref = coco_map(d["img"], np.zeros(len(d["img"]), int), d["score"], d["box"], d["gt_img"],
                           np.zeros(len(d["gt_img"]), int), d["gt_box"])["ap50"]
            srt = np.argsort(-d["score"], kind="stable")
            full = ap50(d["img"][srt], tp[srt], d["gt_img"], np.ones(n_img))
            assert abs(full - ref) < 1e-6, f"{key}: weighted AP50 {full:.6f} != coco_map {ref:.6f}"
            prep.append((d["img"][srt], tp[srt], full, tp, ioa))
        (ia, ta, fa, _, _), (ib, tb, fb, _, _) = prep
        diff = np.empty(args.n_boot)
        for r in range(args.n_boot):
            w = rng.multinomial(n_img, np.full(n_img, 1 / n_img)).astype(float)
            diff[r] = ap50(ib, tb, b["gt_img"], w) - ap50(ia, ta, a["gt_img"], w)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        print(f"{key:26s} photos {n_img:5,d} targets {len(a['gt_img']):6,d}  AP50 A {fa:.4f} B {fb:.4f}  B-A {fb - fa:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(B>A) {np.mean(diff > 0):.3f}")
        for name, d, (_, _, _, tp, ioa) in (("A", a, prep[0]), ("B", b, prep[1])):
            for thr in args.where:
                m = d["score"] >= thr
                fp = m & ~tp
                print(f"    {name} objectness >= {thr}: hits {int((m & tp).sum()):,}, false boxes {int(fp.sum()):,} "
                      f"({fp.sum() / n_img:.3f}/photo): inside a target {int((fp & (ioa >= 0.5)).sum()):,}, straddling "
                      f"{int((fp & (ioa > 0) & (ioa < 0.5)).sum()):,}, away {int((fp & (ioa == 0)).sum()):,}")


if __name__ == "__main__":
    main()

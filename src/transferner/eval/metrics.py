"""top-k, macro-F1, ECE, look-alike confusion, per-level (species/genus/family). Bootstrap CIs + McNemar.
DMR (dangerous miss rate, proposal §Evaluation) over the toxic sets of the trait table (D-004/D-005)."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of top-1 confidence, equal-width bins on (0, 1]."""
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def per_class_f1(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(macro recall, macro F1) over the classes present in y; predicted-only classes count as FP."""
    classes = np.unique(y)
    n = max(pred.max(), y.max()) + 1
    tp = np.bincount(y[pred == y], minlength=n).astype(float)
    support = np.bincount(y, minlength=n).astype(float)
    predicted = np.bincount(pred, minlength=n).astype(float)
    rec = tp[classes] / support[classes]
    prec = np.divide(tp[classes], predicted[classes], out=np.zeros(len(classes)), where=predicted[classes] > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(classes)), where=(prec + rec) > 0)
    return float(rec.mean()), float(f1.mean())


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 1000,
                         alpha: float = 0.05, seed: int = 0, weights: np.ndarray | None = None) -> tuple[float, float]:
    """CI of mean(values), resampling whole clusters (observations: their images are correlated). With weights,
    of the weighted mean: encounter-weighted rates (eval/encounter.py; audit evaluation-power-and-weighting)."""
    _, inv = np.unique(clusters, return_inverse=True)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    sums = np.bincount(inv, weights=values.astype(float) * w)
    cnts = np.bincount(inv, weights=w)
    rng = np.random.default_rng(seed)
    k = len(sums)
    stats = np.empty(n_boot)
    for i in range(0, n_boot, 50):
        j = min(50, n_boot - i)
        w = rng.multinomial(k, np.full(k, 1 / k), size=j)        # cluster resample counts
        stats[i:i + j] = (w @ sums) / (w @ cnts)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test on paired correctness. b_only = cases only B gets right."""
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    p = binomtest(b_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_only": a_only, "b_only": b_only, "p": float(p)}


def paired_agreement(ref_logits: np.ndarray, test_logits: np.ndarray, y: np.ndarray) -> dict:
    """A reference model (float) against a variant of it (INT8) on the same photos: McNemar discordant counts
    (ref_only: right only in the reference; test_only: right only in the variant), the top-1 change with its
    paired 95% CI, top-1 agreement and mean KL(ref || test) of the softmax distributions.

    Why (audit live_int8_precision-loss-overstated, D-043): D-041 read -3.7 pt INT8 loss from 512 photos (26 vs 7
    discordant); paired on 12,358 regional test photos it was -1.08 +- 0.38 pt (361 vs 228). A 1-3 pt delta on a
    few hundred photos is noise, while agreement and KL are stable from ~1,300 photos and rank PTQ recipes.
    Ties (INT8 logits with a coarse scale tie exactly on ~4.6% of frames) go to the lowest index, as argmax."""
    a = np.asarray(ref_logits, np.float64)
    b = np.asarray(test_logits, np.float64)
    y = np.asarray(y)
    la = a - a.max(1, keepdims=True)
    la -= np.log(np.exp(la).sum(1, keepdims=True))
    lb = b - b.max(1, keepdims=True)
    lb -= np.log(np.exp(lb).sum(1, keepdims=True))
    pa, pb = la.argmax(1), lb.argmax(1)
    mc = mcnemar(pa == y, pb == y)
    n = len(y)
    d = (mc["b_only"] - mc["a_only"]) / n
    se = np.sqrt(max(mc["a_only"] + mc["b_only"] - n * d * d, 0.0)) / n     # SD of the per-photo difference / sqrt(n)
    kl = (np.exp(la) * (la - lb)).sum(1)
    return {"n": n, "ref_only": mc["a_only"], "test_only": mc["b_only"], "mcnemar_p": mc["p"],
            "delta_top1": float(d), "delta_ci95": float(1.96 * se), "top1_agree": float((pa == pb).mean()),
            "kl_mean": float(kl.mean())}


def lookalike_confusion(pred: np.ndarray, y: np.ndarray, groups: dict[int, set[str]]) -> dict:
    """Among images of look-alike species: error rate, and the share of errors that land on a
    species from the same look-alike group (the confusions the groups exist to catch)."""
    m = np.array([c in groups for c in y])
    wrong = m & (pred != y)
    same = sum(bool(groups.get(p, set()) & groups[t]) for p, t in zip(pred[wrong], y[wrong]))
    return {"n": int(m.sum()), "err": float(wrong.sum() / max(m.sum(), 1)),
            "err_within_group": float(same / max(m.sum(), 1)),
            "share_of_errors_within_group": float(same / max(wrong.sum(), 1))}


def dmr(pred: np.ndarray, y: np.ndarray, toxic: np.ndarray, conf: np.ndarray | None = None,
        tau: float | None = None) -> dict:
    """Dangerous miss rate: of the images whose TRUE species is toxic (toxic[class] is True),
    the share predicted as a non-toxic species without abstaining (abstain = conf < tau).
    Predicting another toxic species is not a miss (the warning still fires). Species with no
    trait rows count as non-toxic predictions, exactly as the app would treat them."""
    m = toxic[y]
    if not m.any():
        return {"n": 0, "dmr": float("nan"), "coverage": float("nan")}
    sure = np.ones(len(y), bool) if tau is None else conf >= tau
    miss = ~toxic[pred] & sure
    return {"n": int(m.sum()), "dmr": float(miss[m].mean()), "coverage": float(sure[m].mean())}


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of xyxy boxes a (n, 4) and b (m, 4)."""
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def coco_map(det_img, det_cls, det_score, det_box, gt_img, gt_cls, gt_box,
             thrs=np.linspace(0.5, 0.95, 10)) -> dict:
    """COCO bbox AP without pycocotools' per-(image, class) Python loops, which take minutes at
    1,362 classes. Same rules as COCOeval (area 'all', no crowd, maxDets not binding at <= 100 per
    image): per image and class, detections in score order take the unmatched GT of the same class
    with the highest IoU >= t; per class, precision is made monotone and read at 101 recall points;
    classes without GT are skipped. Returns {'map': mean over t, 'ap50', 'ap75', 'per_class50'}."""
    det_img, det_cls, gt_img, gt_cls = map(np.asarray, (det_img, det_cls, gt_img, gt_cls))
    det_score, det_box, gt_box = np.asarray(det_score, float), np.asarray(det_box, float), np.asarray(gt_box, float)
    T = len(thrs)
    tp = np.zeros((T, len(det_img)), bool)
    d_ord = np.lexsort((-det_score, det_img))                     # by image, then score (desc)
    d_imgs, d_start = np.unique(det_img[d_ord], return_index=True)
    d_end = np.append(d_start[1:], len(d_ord))
    g_ord = np.argsort(gt_img, kind="stable")
    g_imgs, g_start = np.unique(gt_img[g_ord], return_index=True)
    g_end = np.append(g_start[1:], len(g_ord))
    g_pos = dict(zip(g_imgs.tolist(), zip(g_start.tolist(), g_end.tolist())))
    for im, s, e in zip(d_imgs.tolist(), d_start.tolist(), d_end.tolist()):
        if im not in g_pos:
            continue
        d = d_ord[s:e]
        g = g_ord[slice(*g_pos[im])]
        iou = box_iou_np(det_box[d], gt_box[g])
        iou[det_cls[d][:, None] != gt_cls[g][None, :]] = -1.0
        cand = np.flatnonzero(iou.max(1) >= thrs[0])              # only these can ever be a TP
        if len(cand) == 0:
            continue
        for t, thr in enumerate(thrs):
            used = np.zeros(len(g), bool)
            for k in cand:
                row = np.where(used, -1.0, iou[k])
                j = int(row.argmax())
                if row[j] >= min(thr, 1 - 1e-10):
                    used[j] = True
                    tp[t, d[k]] = True
    rec_thrs = np.linspace(0, 1, 101)
    n_gt = np.bincount(gt_cls, minlength=max(gt_cls.max(), det_cls.max() if len(det_cls) else 0) + 1)
    classes = np.flatnonzero(n_gt)
    c_ord = np.lexsort((-det_score, det_cls))
    c_vals, c_start = np.unique(det_cls[c_ord], return_index=True)
    c_end = np.append(c_start[1:], len(c_ord))
    c_pos = dict(zip(c_vals.tolist(), zip(c_start.tolist(), c_end.tolist())))
    ap = np.zeros((T, len(classes)))
    for ci, c in enumerate(classes.tolist()):
        if c not in c_pos:
            continue                                               # GT but no detections: AP 0
        idx = c_ord[slice(*c_pos[c])]
        for t in range(T):
            tps = np.cumsum(tp[t, idx])
            fps = np.cumsum(~tp[t, idx])
            rc = tps / n_gt[c]
            pr = tps / (tps + fps + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            pi = np.searchsorted(rc, rec_thrs, side="left")
            ap[t, ci] = np.where(pi < len(pr), pr[np.minimum(pi, len(pr) - 1)], 0.0).mean()
    i50 = int(np.argmin(np.abs(thrs - 0.5)))
    i75 = int(np.argmin(np.abs(thrs - 0.75)))
    return {"map": float(ap.mean()), "ap50": float(ap[i50].mean()), "ap75": float(ap[i75].mean()),
            "per_class50": dict(zip(classes.tolist(), ap[i50].tolist()))}

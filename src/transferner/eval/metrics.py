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
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI of mean(values), resampling whole clusters (observations: their images are correlated)."""
    _, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(float))
    cnts = np.bincount(inv).astype(float)
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

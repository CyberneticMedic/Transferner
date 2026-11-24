"""The audit's yardstick fixes on synthetic data (run: python tests/test_eval_yardsticks.py; CPU, no data needed):
paired float-vs-INT8 agreement (export/detector_app check), the fixed-false-alert threshold of eval/edible_alert,
encounter weights, the weighted cluster bootstrap and the app's name-twin merge."""
import numpy as np
import pandas as pd

from transferner.eval.edible_alert import threshold_at_fa
from transferner.eval.encounter import n_eff, photo_weights, wmean
from transferner.eval.metrics import cluster_bootstrap_ci, paired_agreement
from transferner.eval.warning_flags import merge_twins, name_twins


def test_paired_agreement_reproduces_audit():
    """361 photos right only in float and 228 only in INT8 out of 12,358 is -1.08 +- 0.38 pt
    (audit live_int8_precision-loss-overstated)."""
    n, G = 12358, 20
    r = np.random.default_rng(0)
    y = r.integers(G, size=n)
    ref = r.normal(0, 1, (n, G)); ref[np.arange(n), y] += 10            # float right everywhere...
    test = ref.copy()
    wrong = (y + 1) % G
    a, b = np.arange(228), np.arange(228, 589)
    ref[a, y[a]] = -10; ref[a, wrong[a]] = 20                           # ...but on 228 only the variant is right
    test[b, y[b]] = -10; test[b, wrong[b]] = 20                         # and on 361 only float is right
    out = paired_agreement(ref, test, y)
    assert (out["ref_only"], out["test_only"], out["n"]) == (361, 228, n), out
    assert abs(100 * out["delta_top1"] - (-1.076)) < 0.01, out
    assert abs(100 * out["delta_ci95"] - 0.38) < 0.01, out
    assert out["mcnemar_p"] < 1e-6
    assert abs(out["top1_agree"] - (1 - 589 / n)) < 1e-9
    assert out["kl_mean"] > 0
    same = paired_agreement(ref, ref, y)
    assert same["ref_only"] == same["test_only"] == 0 and same["top1_agree"] == 1 and abs(same["kl_mean"]) < 1e-12


def test_threshold_at_fa_never_exceeds_target():
    r = np.random.default_rng(1)
    for _ in range(200):
        s = np.round(r.random(r.integers(5, 300)), r.integers(1, 3))      # coarse scores: many ties
        w = r.random(len(s)) if r.random() < 0.5 else np.ones(len(s))
        for rate in (0.0, 0.01, 0.05, 0.3, 1.0):
            t = threshold_at_fa(s, w, rate)
            fa = w[s >= t].sum() / w.sum()
            assert fa <= rate + 1e-12, (rate, fa)
            # and it is the lowest such threshold: the next score down would break the budget
            below = s[s < t]
            if len(below) and rate < 1.0:
                assert w[s >= below.max()].sum() / w.sum() > rate - 1e-12


def test_photo_weights_carry_encounter_share():
    cls = np.array([0, 0, 0, 1, 2, 2])
    counts = np.array([10.0, 30.0, 60.0])
    w = photo_weights(cls, counts)
    assert abs(w.sum() - 1) < 1e-12
    assert np.allclose([w[cls == k].sum() for k in range(3)], counts / counts.sum())
    genus_of = np.array([0, 0, 1])
    wg = photo_weights(cls, counts, genus_of, by="genus")
    assert np.allclose([wg[genus_of[cls] == g].sum() for g in range(2)], [0.4, 0.6])
    assert wmean(np.array([1.0, 0, 0, 0, 0, 0]), w) == w[0] and n_eff(np.ones(7)) == 7


def test_weighted_bootstrap_matches_unweighted_with_unit_weights():
    r = np.random.default_rng(2)
    v = (r.random(500) < 0.6).astype(float)
    c = r.integers(400, size=500)
    assert cluster_bootstrap_ci(v, c) == cluster_bootstrap_ci(v, c, weights=np.ones(500))
    lo, hi = cluster_bootstrap_ci(v, c, weights=r.random(500) + 0.1)
    assert 0.4 < lo < hi < 0.8


def test_name_twins_merge_like_the_app():
    sp = pd.DataFrame({"class_idx": range(5), "scientific_name": ["a", "b", "a", "c", "b"]})
    canon, twin = name_twins(sp)
    assert list(canon) == [0, 1, 0, 3, 1] and list(twin) == [2, 4]
    p = np.array([[0.1, 0.2, 0.3, 0.15, 0.25]])
    assert np.allclose(merge_twins(p, canon, twin), [[0.4, 0.45, 0, 0.15, 0]])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)

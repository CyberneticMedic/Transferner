"""Live edible recall at a fixed false-alert rate: the goal metric nothing measured (audit evaluation-no-goal-metric,
edible_targeting_app-01/-03, recommended_sequence step 1). Selection so far used genus top-1 and box AP alone.

What the app does (0.6; core/Live.kt ForageScorer, ui/AppViewModel.kt): each frame's unfocused whole-frame genus
distribution p gets a forage score S = sum over target genera of p_g * e_g, where e_g is the share of the genus'
regional sightings that are species with a cited edible or conditional row (bundle forage.json). The alert names
argmax_g p_g * e_g. Walk mode alerts when the smoothed S stays >= alert_enter (0.5) for 0.75 s and BioCLIP then
confirms the frame (genus probability >= verify_genus_p, or BioCLIP's top species is in that genus); a chip
shows from chip_min (0.3). Here one cached regional photo stands for one steady frame (no EMA, no hold).

Photo classes, per target list:
  target      species with a cited edible or conditional row, in a target genus (what an alert should find)
  non-target  genus not in the list: the false-alert set
  congener    a species without such a row in a target genus. A genus-level score cannot tell these from the
              targets (audit evaluation-no-goal-metric (f)), so they are reported apart, not hidden in either rate;
              "toxic congener" = a congener with a cited toxic row.
Lists: "alert" = the default alert tier of forage.json (47 genera); "all" = every forage.json genus (all 192 genera
with a cited edible or conditional species, as if the user put them all in the alert tier).

Operating points: the threshold giving a false-alert rate <= 1%, 2%, 5% of non-target photos, found on the same
split and weighting (a point on the curve, like TPR at fixed FPR; not a tuned threshold), and the app's own
S >= alert_enter and S >= chip_min. Columns at each point: recall (fires AND names the true genus), recall_any
(fires), fa, congener, toxic_congener; *_checked add the BioCLIP confirmation (cached capture embeddings of the
same photos), which only removes alerts. recall_lo/hi: 95% observation-cluster bootstrap of recall at that
threshold. Rates per photo and encounter-weighted (eval/encounter.py; species weights unless --weighting genus);
n_eff_pos is the Kish effective size of the weighted targets (val has ~160 target photos: read test).

An alert means "worth a closer look", never "edible": the toxic-congener column is what a genus-level alert
costs, and absence of a cited toxic row never means safe.
Writes results/tables/edible_alert.csv and edible_alert_genera.csv (per alert-tier genus, per photo, at the
per-photo thresholds), replacing earlier rows of the same model and logits. CPU only.
  python -m transferner.eval.edible_alert --name mnv4_live448 --logits image_logits_step12000.npz"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils.config import data_root
from .encounter import n_eff, photo_weights, species_counts, wmean
from .metrics import cluster_bootstrap_ci
from .warning_flags import merge_twins, name_twins

FA_TARGETS = (0.01, 0.02, 0.05)


def threshold_at_fa(s_neg: np.ndarray, w_neg: np.ndarray, rate: float) -> float:
    """Lowest threshold t whose (weighted) share of non-target photos with S >= t is <= rate. Ties at the
    boundary are excluded, so the achieved rate never exceeds the target."""
    o = np.argsort(-s_neg, kind="stable")
    s, c = s_neg[o], np.cumsum(w_neg[o]) / w_neg.sum()
    j = int(np.searchsorted(c, rate, side="right"))          # first photo that would push the rate over
    return float(s[-1]) if j >= len(s) else float(np.nextafter(s[j], np.inf))


def bioclip_check(root: Path, split: str, uuids: np.ndarray, sp: pd.DataFrame, gi: dict, chunk: int = 4096):
    """Per photo: BioCLIP's probability summed per detector genus (n, G), and the detector genus of BioCLIP's top
    species (-1 when outside the detector's genera), on probabilities merged over name twins as in the app."""
    m = pd.read_parquet(root / f"artifacts/img_emb/{split}.parquet", columns=["uuid"])
    row = pd.Series(np.arange(len(m)), index=m.uuid.to_numpy()).reindex(uuids)
    assert row.notna().all(), f"{int(row.isna().sum())} photos lack a BioCLIP embedding"
    row = row.to_numpy().astype(np.int64)
    e = np.load(root / f"artifacts/img_emb/{split}.npy", mmap_mode="r")
    pr = torch.load(root / "artifacts/probe.pt", map_location="cpu")
    T = float((root / "artifacts/temperature.txt").read_text())
    canon, twin = name_twins(sp)
    sg = np.array([gi.get(g, -1) for g in sp.genus])
    valid = np.where(sg >= 0)[0]
    gmass = np.zeros((len(row), len(gi)), np.float32)
    top_g = np.empty(len(row), np.int64)
    for s in range(0, len(row), chunk):
        x = torch.from_numpy(np.asarray(e[row[s:s + chunk]], dtype=np.float32))
        p = merge_twins(((pr["scale"] * x @ pr["W"].float().T + pr["b"].float()) / T).softmax(-1).numpy(), canon, twin)
        gmass[s:s + len(p)] = torch.zeros(len(p), len(gi)).index_add_(
            1, torch.from_numpy(sg[valid]), torch.from_numpy(p[:, valid])).numpy()
        top_g[s:s + len(p)] = sg[p.argmax(1)]
    return gmass, top_g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_live448", help="detector whose cached whole-frame logits to use")
    ap.add_argument("--logits", default="image_logits_step12000.npz", help="cache file in the detector's directory")
    ap.add_argument("--forage", default="bundle/0.6/forage.json", help="tiers and e_g, relative to $TRANSFERNER_DATA")
    ap.add_argument("--tuning", default="bundle/0.6/tuning.json")
    ap.add_argument("--weighting", choices=["species", "genus"], default="species")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = data_root()
    repo = Path(__file__).resolve().parents[3]
    tun = json.loads((root / args.tuning).read_text())
    forage = json.loads((root / args.forage).read_text())["genera"]
    d = root / "artifacts/detector" / args.name
    genera = json.loads((d / "labels.json").read_text())["genera"]
    gi = {g: i for i, g in enumerate(genera)}
    assert all(genera[f["g"]] == f["genus"] for f in forage), "forage.json was built for another genus list"
    e_g = np.zeros(len(genera))
    for f in forage:
        e_g[f["g"]] = f["weight"]
    lists = {"alert": np.array(sorted(f["g"] for f in forage if f["tier"] == "alert")),
             "all": np.array(sorted(f["g"] for f in forage))}
    label = {f["g"]: f.get("label", "") for f in forage}

    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx").reset_index(drop=True)
    tr = pd.read_csv(repo / "data/traits/species_traits.csv")
    ed = sp.scientific_name.isin(set(tr[tr.category.isin(["edible", "conditional"])].scientific_name)).to_numpy()
    toxic = sp.scientific_name.isin(set(tr[tr.category == "toxic"].scientific_name)).to_numpy()
    sp_g = np.array([gi.get(g, -1) for g in sp.genus])
    _, genus_of = np.unique(sp.genus.to_numpy(), return_inverse=True)
    counts = species_counts(root)
    enc = f"encounter_{args.weighting}"
    z = np.load(d / args.logits, allow_pickle=True)
    im = pd.read_parquet(root / "manifests/images.parquet",
                         columns=["uuid", "path", "source", "class_idx", "observation_id"]).set_index("path")
    rows, grows = [], []
    for split in ("val", "test"):
        meta = im.reindex(z[f"{split}_path"])
        reg = (meta.source == "gbif_regional").to_numpy()
        meta = meta[reg]
        lg = z[f"{split}_logits"][reg].astype(np.float64)
        cls = meta.class_idx.to_numpy().astype(int)
        gy = sp_g[cls]
        assert (gy == z[f"{split}_y"][reg]).all(), "cached labels disagree with the species' genus"
        obs = meta.observation_id.astype(str).where(meta.observation_id.notna(), meta.uuid).to_numpy()
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        gmass, top_g = bioclip_check(root, split, meta.uuid.to_numpy(), sp, gi)
        weights = {"photo": None, enc: photo_weights(cls, counts, genus_of, args.weighting)}
        for lname, L in lists.items():
            wl = np.zeros(len(genera)); wl[L] = e_g[L]
            contrib = p * wl
            S, a = contrib.sum(1), contrib.argmax(1)
            in_l = np.isin(gy, L)
            pos, cong, neg = in_l & ed[cls], in_l & ~ed[cls], ~in_l
            tcong = cong & toxic[cls]
            named = a == gy
            checked = (gmass[np.arange(len(a)), a] >= tun["verify_genus_p"]) | (top_g == a)
            for wname, w in weights.items():
                wn = np.ones(neg.sum()) if w is None else w[neg]
                sub = lambda m: None if w is None else w[m]
                points = [(f"fa<={int(100 * r)}%", threshold_at_fa(S[neg], wn, r)) for r in FA_TARGETS]
                points += [(f"app alert S>={tun['alert_enter']}", tun["alert_enter"]), (f"app chip S>={tun['chip_min']}", tun["chip_min"])]
                for pname, t in points:
                    fire = S >= t
                    fc = fire & checked
                    hit = (fire & named)[pos]
                    lo, hi = cluster_bootstrap_ci(hit.astype(float), obs[pos], args.boot, weights=sub(pos)) if pos.any() else (np.nan, np.nan)
                    rows.append({"model": args.name, "logits": args.logits, "split": split, "list": lname, "weighting": wname,
                                 "point": pname, "threshold": round(t, 5), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
                                 "n_cong": int(cong.sum()), "n_toxic_cong": int(tcong.sum()),
                                 "n_eff_pos": round(n_eff(np.ones(pos.sum()) if w is None else w[pos]), 1),
                                 "recall": round(wmean(hit, sub(pos)), 4), "recall_lo": round(lo, 4), "recall_hi": round(hi, 4),
                                 "recall_any": round(wmean(fire[pos], sub(pos)), 4), "fa": round(wmean(fire[neg], sub(neg)), 4),
                                 "congener": round(wmean(fire[cong], sub(cong)), 4),
                                 "toxic_congener": round(wmean(fire[tcong], sub(tcong)), 4),
                                 "recall_checked": round(wmean((fc & named)[pos], sub(pos)), 4),
                                 "fa_checked": round(wmean(fc[neg], sub(neg)), 4),
                                 "congener_checked": round(wmean(fc[cong], sub(cong)), 4),
                                 "toxic_congener_checked": round(wmean(fc[tcong], sub(tcong)), 4)})
                    if lname == "alert" and w is None:
                        for g in L:
                            pg, cg = pos & (gy == g), cong & (gy == g)
                            grows.append({"model": args.name, "logits": args.logits, "split": split, "genus": genera[g],
                                          "label": label.get(int(g), ""), "e_g": e_g[g], "point": pname, "threshold": round(t, 5),
                                          "n_pos": int(pg.sum()), "recall": round(wmean((fire & named)[pg], None), 4),
                                          "recall_checked": round(wmean((fc & named)[pg], None), 4),
                                          "n_cong": int(cg.sum()), "congener": round(wmean(fire[cg], None), 4)})
    df, gdf = pd.DataFrame(rows), pd.DataFrame(grows)
    for name, new in (("edible_alert", df), ("edible_alert_genera", gdf)):
        out = repo / f"results/tables/{name}.csv"
        if out.exists():                          # replace this model's rows; other weightings' rows stay
            old = pd.read_csv(out)
            same = (old.model == args.name) & (old.logits == args.logits)
            if "weighting" in old:
                same &= old.weighting.isin(new.weighting.unique())
            new = pd.concat([old[~same], new], ignore_index=True)
        new.to_csv(out, index=False)
        print(f"wrote {out}")
    pd.set_option("display.width", 250)
    cols = ["split", "list", "weighting", "point", "threshold", "recall", "recall_lo", "recall_hi", "recall_any", "fa",
            "congener", "toxic_congener", "recall_checked", "fa_checked", "n_pos", "n_eff_pos"]
    print(df[cols].to_string(index=False))
    never = [f"{genera[g]} {e_g[g]:.2f}" for g in lists["alert"] if e_g[g] < tun["alert_enter"]]
    print(f"alert-tier genera whose e_g < alert_enter ({tun['alert_enter']}): S cannot reach the alert on their own, "
          f"even at p_g = 1: {len(never)} of {len(lists['alert'])}: {', '.join(never)}")


if __name__ == "__main__":
    main()

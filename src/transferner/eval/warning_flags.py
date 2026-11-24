"""Warning-flag rules, false positives preferred over false negatives (user, 2026-11-24; D-038).

Ground truth: a photo is "toxic" when its true species has a cited toxic row (data/traits/species_traits.csv);
"serious" when one of those rows has moderate/high severity or is contact-toxic. A rule MISSES a toxic photo
when it raises no flag, and FALSE-ALARMS on a non-toxic photo when it raises one. Regional photos only (val
selects, test reports). A miss rate here is never a safety claim: absence of a cited toxic row is not "safe".

Capture (BioCLIP probe, cached embeddings) - a flag is a caution banner:
  top5          a cited toxic species in the top 5 (app 0.4)
  top5+rel      ... or a candidate without trait rows whose genus has a cited toxic species ("relative")
  p>=t (+rel)   any cited toxic (or relative) species with probability >= t anywhere in the 4,411
  app0.5 ...    D-038 as shipped in 0.4.1-0.5: any toxic, conditional or relative species in the top 5 at any
                probability, or a toxic or relative species at p >= 0.005 anywhere
  app0.6 ...    D-044 as shipped (core/Capture.kt CaptureRules.caution, bundle tuning.json), on probabilities
                merged over the 28 name twins as the app does: toxic + relative probability summed over all
                4,411 >= banner_sum, or a conditional-only species in the top 5 at >= conditional_min;
                "strong" = the sum >= banner_strong
Live (detector whole-frame genus, cached float logits) - a flag is the live warning row:
  any toxic / serious genus p>=t   a genus with probability >= t holding a cited toxic (serious) species
  app0.6 live ...  ml/Models.kt LiveDetector.run: a toxic genus among the top 8 at p >= 0.02 (LIVE_MIN_P); on a
                   single photo the warning EMA equals the frame. "+presence" also applies the plant-presence
                   gate (top-5 mass of the frame >= presence_warn), which hides the row on frames that look like
                   no known plant. "walk haptic": the walk-mode long buzz (AppViewModel), mass of the genera the
                   app counts as serious >= toxic_haptic on a frame that passes the gate.
The audit (errors_found "Warning evaluation does not measure the app's rules..."; capture_lookalike_resolution-4,
live_int8_precision-warnings-objectness) found the old rows differ from the app on both paths; the app rows
are the ones to quote.

Every rule is reported per photo (weighting "photo", the old rows unchanged) and encounter-weighted
(eval/encounter.py: each species, or with --weighting genus each genus, carries its share of regional GBIF
sightings; audit evaluation-power-and-weighting). The live val slice is 1,304 photos (about 235 toxic), so its
rates are +-3 pt; read test.
  python -m transferner.eval.warning_flags [--live mnv4_shared] [--logits image_logits.npz] [--weighting species]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..utils.config import data_root
from .encounter import photo_weights, species_counts, wmean

LIVE_MIN_P = 0.02            # Traits.kt WarningIndex.LIVE_MIN_P: not in tuning.json
CAPTURE_MIN_P_05 = 0.005     # Traits.kt WarningIndex.CAPTURE_MIN_P: the 0.4.1-0.5 banner (D-038)


def topk_idx(p: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest entries per row, unordered (the rules only ask "is it among the top k")."""
    return np.argpartition(-p, k, axis=1)[:, :k]


def name_twins(sp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Capture.kt NameAliases (audit capture_model-1): per class index, the first index of its scientific name,
    and the twin indices whose probability the app folds onto it. sp is species.parquet sorted by class_idx."""
    canon = sp.groupby("scientific_name").class_idx.transform("min").to_numpy()
    twin = np.where(canon != np.arange(len(sp)))[0]
    assert len(set(canon[twin])) == len(twin), "a name with more than two classes"
    return canon, twin


def merge_twins(p: np.ndarray, canon: np.ndarray, twin: np.ndarray) -> np.ndarray:
    """Species probabilities as the app 0.6 sees them: each twin's mass summed onto its first index."""
    pm = p.copy()
    pm[:, canon[twin]] += p[:, twin]
    pm[:, twin] = 0
    return pm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="mnv4_shared", help="detector whose cached whole-frame logits to use")
    ap.add_argument("--logits", default="image_logits.npz", help="cache file in the detector's directory (eval/detector_prior --ckpt)")
    ap.add_argument("--tuning", default="bundle/0.6/tuning.json", help="the shipped thresholds, relative to $TRANSFERNER_DATA")
    ap.add_argument("--weighting", choices=["species", "genus"], default="species",
                    help="encounter weights per species (toxicity is a species fact) or per genus (D-040's encounter_top1)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = data_root()
    repo = Path(__file__).resolve().parents[3]
    tun = json.loads((root / args.tuning).read_text())
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx").reset_index(drop=True)
    tr = pd.read_csv(repo / "data/traits/species_traits.csv")
    tox_rows = tr[tr.category == "toxic"]
    # every toxic row has ingestion_toxic or contact_toxic set, so this is the app's TraitSummary.toxic
    toxic = sp.scientific_name.isin(set(tox_rows.scientific_name)).to_numpy()
    serious = sp.scientific_name.isin(set(tox_rows[tox_rows.ingestion_severity != "low"].scientific_name)
                                      | set(tox_rows[tox_rows.contact_toxic == 1].scientific_name)).to_numpy()
    # the app's TraitSummary.serious (live genusWarn == 2, the walk haptic): highest severity not "low";
    # contact toxicity alone does not count there, unlike D-038's text and the ground truth above
    serious_app = sp.scientific_name.isin(set(tox_rows[tox_rows.ingestion_severity.isin(["moderate", "high"])].scientific_name)).to_numpy()
    has_rows = sp.scientific_name.isin(set(tr.scientific_name)).to_numpy()
    tox_genera = set(sp.genus[toxic])
    rel = (~has_rows) & sp.genus.isin(tox_genera).to_numpy()
    cond = sp.scientific_name.isin(set(tr[tr.category == "conditional"].scientific_name)).to_numpy()
    cond_only = cond & ~toxic & ~rel             # Capture.kt: a conditional candidate that is not toxic or a relative
    tox_rel = toxic | rel
    needs_caution = toxic | cond | rel           # TraitSummary.needsCaution (the 0.4-0.5 top-5 term)
    canon, twin = name_twins(sp)
    _, genus_of = np.unique(sp.genus.to_numpy(), return_inverse=True)
    counts = species_counts(root)
    enc = f"encounter_{args.weighting}"
    pr = torch.load(root / "artifacts/probe.pt", map_location="cpu")
    T = float((root / "artifacts/temperature.txt").read_text())
    rows = []

    def add(kind, split, rule, flag, y_toxic, y_serious, w, weighting):
        rows.append({"kind": kind, "split": split, "rule": rule, "n": len(flag),
                     "miss_toxic": round(wmean(~flag[y_toxic], None if w is None else w[y_toxic]), 4),
                     "miss_serious": round(wmean(~flag[y_serious], None if w is None else w[y_serious]), 4),
                     "false_alarm": round(wmean(flag[~y_toxic], None if w is None else w[~y_toxic]), 4),
                     "weighting": weighting})

    def add_all(kind, split, flags, cls):
        yt, ys = toxic[cls], serious[cls]
        w = photo_weights(cls, counts, genus_of, args.weighting)
        for rule, f in flags.items():
            add(kind, split, rule, f, yt, ys, None, "photo")
        for rule, f in flags.items():
            add(kind, split, rule, f, yt, ys, w, enc)

    bs, bst, cmin = tun["banner_sum"], tun["banner_strong"], tun["conditional_min"]
    for split in ("val", "test"):
        m = pd.read_parquet(root / f"artifacts/img_emb/{split}.parquet")
        e = np.load(root / f"artifacts/img_emb/{split}.npy", mmap_mode="r")
        keep = np.where((m.source == "gbif_regional").to_numpy())[0]
        y = m.class_idx.to_numpy()[keep]
        flags: dict[str, list] = {}
        for s in range(0, len(keep), 20000):     # chunks: test is 176,590 x 4,411 probabilities
            p = ((pr["scale"] * torch.from_numpy(np.asarray(e[keep[s:s + 20000]], dtype=np.float32))
                  @ pr["W"].float().T + pr["b"].float()) / T).softmax(-1).numpy()
            top5 = topk_idx(p, 5)
            f = {"top5": toxic[top5].any(1), "top5+rel": tox_rel[top5].any(1)}
            for t in (0.02, 0.01, 0.005):
                f[f"p>={t}"] = (p[:, toxic] >= t).any(1) | toxic[top5].any(1)
                f[f"p>={t}+rel"] = (p[:, tox_rel] >= t).any(1) | tox_rel[top5].any(1)
            f["app0.5 banner: top5 toxic/cond/rel or toxic/rel p>=0.005"] = \
                needs_caution[top5].any(1) | (p[:, tox_rel] >= CAPTURE_MIN_P_05).any(1)
            pm = merge_twins(p, canon, twin)      # app 0.6 merges name twins before every rule
            ssum = pm[:, tox_rel].astype(np.float64).sum(1)
            top5m = topk_idx(pm, 5)
            cterm = (cond_only[top5m] & (np.take_along_axis(pm, top5m, 1) >= cmin)).any(1)
            f[f"toxic+rel sum>={bs}"] = ssum >= bs
            f[f"app0.6 banner: toxic+rel sum>={bs} or cond-only top5 p>={cmin}"] = (ssum >= bs) | cterm
            f[f"app0.6 strong: toxic+rel sum>={bst}"] = ssum >= bst
            for k, v in f.items():
                flags.setdefault(k, []).append(v)
        add_all("capture", split, {k: np.concatenate(v) for k, v in flags.items()}, y)

    d = root / "artifacts/detector" / args.live
    genera = json.loads((d / "labels.json").read_text())["genera"]
    g_tox = np.array([g in tox_genera for g in genera])
    g_ser = np.array([g in set(sp.genus[serious]) for g in genera])
    g_ser_app = np.array([g in set(sp.genus[serious_app]) for g in genera])
    z = np.load(d / args.logits, allow_pickle=True)
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path", "source", "class_idx"]).set_index("path")
    pw, th = tun["presence_warn"], tun["toxic_haptic"]
    for split in ("val", "test"):
        lg, paths = z[f"{split}_logits"].astype(np.float64), z[f"{split}_path"]
        meta = im.reindex(paths)
        reg = (meta.source == "gbif_regional").to_numpy()
        lg, cls = lg[reg], meta.class_idx.to_numpy()[reg].astype(int)
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        f = {}
        for t in (0.2, 0.1, 0.05, 0.02):
            f[f"any toxic genus p>={t}"] = (p[:, g_tox] >= t).any(1)
            f[f"serious genus p>={t}"] = (p[:, g_ser] >= t).any(1)
        top8 = topk_idx(p, 8)
        app = (g_tox[top8] & (np.take_along_axis(p, top8, 1) >= LIVE_MIN_P)).any(1)
        presence = np.take_along_axis(p, topk_idx(p, 5), 1).sum(1)
        f[f"app0.6 live: toxic genus in top8 p>={LIVE_MIN_P}"] = app
        f[f"app0.6 live: toxic genus in top8 p>={LIVE_MIN_P} +presence>={pw}"] = app & (presence >= pw)
        f[f"app0.6 walk haptic: app-serious genus mass>={th} +presence>={pw}"] = (p[:, g_ser_app].sum(1) >= th) & (presence >= pw)
        add_all("live", split, f, cls)

    # photo rows first in the old order, then the encounter-weighted copies (each block: capture, then live)
    df = pd.DataFrame(rows)
    df = df.assign(_w=(df.weighting != "photo").astype(int), _k=(df.kind == "live").astype(int), _i=range(len(df)))
    df = df.sort_values(["_w", "_k", "_i"]).drop(columns=["_w", "_k", "_i"])
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))
    out = repo / ("results/tables/warning_flags.csv" if args.live == "mnv4_shared" else f"results/tables/warning_flags_{args.live}.csv")
    df.to_csv(out, index=False)
    print(f"relatives (no rows, genus has a cited toxic species): {rel.sum()} species; toxic genera {len(tox_genera)} "
          f"({g_tox.sum()} of the detector's {len(genera)}), serious {g_ser.sum()} (app-serious {g_ser_app.sum()}: "
          f"{sorted(g for g, a, b in zip(genera, g_ser, g_ser_app) if a and not b)[:12]}... not in the walk haptic); "
          f"{len(twin)} name twins merged for app0.6; wrote {out}")


if __name__ == "__main__":
    main()

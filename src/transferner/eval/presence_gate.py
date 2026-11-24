"""Plant-presence gate for the live screen (D-044; audit evaluation-objectness-not-plantness).

The objectness score cannot tell plants from non-plant scenes (AUROC ~0.45: an all-black frame scores 0.99),
because no training frame was ever plant-free. Until a retrain with negative frames fixes that, the app gates
boxes and the live toxic row on the whole-frame head's top-5 genus mass of the current frame, which does
separate them. This picks the two thresholds on held-out frames, scored with the app's own INT8 graph
(--tflite, CPU) or the float checkpoint (GPU):
  plants      BioCLIP's regional val photos (one per item, capped), clean and through field_degrade (walking
              blur, downscale, JPEG): the degraded set sets the thresholds, so field frames are not gated away
  toxic       the same pool's photos of genera holding a cited toxic species: the live warning rule (a toxic
              genus in the top 8 at p >= 0.02) with and without the gate
  non-plant   manifests/negatives.parquet plant_free frames of the val split (COCO: no plant pixels; Open Images:
              "Plant" verified absent)
  presence_warn   the highest threshold that keeps --warn-keep of degraded plant frames (99.5%) AND adds at most
                  --max-added-miss (0.1 pt) of toxic warning misses (false positives over negatives)
  presence_boxes  keeps --boxes-keep of degraded plant frames (97%)
Writes artifacts/detector/<name>/presence.json, which export/bundle.py copies into tuning.json.
  python -m transferner.eval.presence_gate --name mnv4_live448 --ckpt step12000.pt                    (GPU venv)
  python -m transferner.eval.presence_gate --name mnv4_live448 --tflite detector_app_int8_rg.tflite  (export venv)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..train.train_detector import BoxSet, collate, field_degrade
from ..utils.config import data_root, load


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def regional_pool(root, split, gi) -> list:
    m = pd.read_parquet(root / f"artifacts/img_emb/{split}.parquet", columns=["uuid", "class_idx", "is_specimen", "source"])
    m = m[(m.source == "gbif_regional") & ~m.is_specimen.fillna(False).astype(bool)]
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "genus"])
    m = m.merge(sp, on="class_idx").merge(pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")
    m = m[m.genus.isin(gi)]
    box = np.zeros((1, 4), np.float32)
    return [(p, box, np.array([gi[g] + 1])) for p, g in zip(m.path, m.genus)]


def torch_runner(d: Path, ckpt: str, size: int, n_genera: int):
    from ..train.train_detector import Net, build, forward_raw
    ck = torch.load(d / ckpt, map_location="cpu")
    model = build(n_genera, load("detector")["model"]["backbone"], size, pretrained=False, head="shared", image_aux=True,
                  image_head=ck["args"].get("image_head", 1280))
    model.load_state_dict(ck["model"]); model.to("cuda").eval()
    net = Net(model)
    mean = torch.tensor(model.backbone.mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(model.backbone.std, device="cuda").view(1, 3, 1, 1)

    def run(imgs):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return forward_raw(net, ((imgs.to("cuda").float() / 255) - mean) / std)["image_logits"].float().softmax(-1).cpu().numpy()
    return run, f"{d.name}/{ckpt} (float, GPU)"


def tflite_runner(d: Path, name: str, threads: int):
    """The app graph exactly as shipped: int8 input from the sidecar's quantisation, dequantised logits."""
    from ai_edge_litert.interpreter import Interpreter
    side = json.loads((d / name.replace(".tflite", ".json")).read_text())
    it = Interpreter(model_path=str(d / name), num_threads=threads)
    it.allocate_tensors()
    r = it.get_signature_runner(side["signature"])
    qi, qo = side["input"], side["outputs"]["image_logits"]
    mean = torch.tensor(side["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(side["std"]).view(1, 3, 1, 1)

    def run(imgs):
        x = (((imgs.float() / 255) - mean) / std).numpy()
        out = []
        for i in range(len(x)):
            v = np.clip(np.round(x[i:i + 1] / qi["scale"]) + qi["zero_point"], -128, 127).astype(np.int8)
            lg = r(**{qi["name"]: v})[qo["name"]].astype(np.float32)
            if qo["dtype"] == "int8":
                lg = (lg - qo["zero_point"]) * qo["scale"]
            out.append(torch.from_numpy(lg[0]).softmax(-1).numpy())
        return np.stack(out)
    return run, f"{d.name}/{name} (INT8 app graph, CPU)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_live448")
    ap.add_argument("--ckpt", default="step12000.pt")
    ap.add_argument("--tflite", default="", help="score with this app graph in the detector directory (export venv)")
    ap.add_argument("--n-plant", type=int, default=6000)
    ap.add_argument("--n-toxic", type=int, default=4000)
    ap.add_argument("--n-neg", type=int, default=6000)
    ap.add_argument("--warn-keep", type=float, default=0.995)
    ap.add_argument("--max-added-miss", type=float, default=0.001)
    ap.add_argument("--boxes-keep", type=float, default=0.97)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.manual_seed(0)
    root = data_root()
    repo = Path(__file__).resolve().parents[3]
    d = root / "artifacts/detector" / args.name
    genera = json.loads((d / "labels.json").read_text())["genera"]
    gi = {g: i for i, g in enumerate(genera)}
    if args.tflite:
        size = json.loads((d / args.tflite.replace(".tflite", ".json")).read_text())["size"]
        run, what = tflite_runner(d, args.tflite, args.threads)
    else:
        size = torch.load(d / args.ckpt, map_location="cpu")["args"]["size"]
        run, what = torch_runner(d, args.ckpt, size, len(genera))
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["scientific_name", "genus"])
    tr = pd.read_csv(repo / "data/traits/species_traits.csv")
    tox_genera = set(sp[sp.scientific_name.isin(set(tr[tr.category == "toxic"].scientific_name))].genus)
    g_tox = np.array([g in tox_genera for g in genera])
    rng = np.random.default_rng(0)
    pool = regional_pool(root, "val", gi)
    perm = rng.permutation(len(pool))
    plants = [pool[i] for i in perm[:args.n_plant]]
    toxic = [it for it in (pool[i] for i in perm) if g_tox[int(it[2][0]) - 1]][:args.n_toxic]
    neg = pd.read_parquet(root / "manifests/negatives.parquet")
    neg = neg[(neg.kind == "plant_free") & (neg.split == "val")]
    neg = neg.sample(min(args.n_neg, len(neg)), random_state=0)
    box = np.zeros((1, 4), np.float32)
    negs = [(p, box, np.array([1])) for p in neg.path]

    def probs(items, degrade=False):
        out = []
        dl = DataLoader(BoxSet(root, items, size, False), batch_size=64, num_workers=args.workers, collate_fn=collate)
        for imgs, *_ in dl:
            if degrade:
                imgs = torch.stack([field_degrade(im, size) for im in imgs])
            p = run(imgs)
            out.append(np.stack([np.sort(p, 1)[:, -5:].sum(1),                                   # top-5 mass
                                 np.array([any(g_tox[j] and q[j] >= 0.02 for j in np.argsort(-q)[:8]) for q in p])], 1))
        return np.concatenate(out)

    pc, pdg, tx, nm = probs(plants)[:, 0], probs(plants, True)[:, 0], probs(toxic, True), probs(negs)[:, 0]
    q_warn = float(np.quantile(pdg, 1 - args.warn_keep))
    fires, tmass = tx[:, 1] > 0, tx[:, 0]
    added = lambda t: float((fires & (tmass < t)).mean())
    cands = np.sort(np.unique(np.round(np.concatenate([pdg, tmass]), 4)))
    ok = [t for t in cands if t <= q_warn and added(t) <= args.max_added_miss]
    warn = float(max(ok)) if ok else 0.0
    boxes = float(np.quantile(pdg, 1 - args.boxes_keep))
    by_src = {s: {"n": int((neg.source == s).sum()), "warn_passes": round(float((nm[(neg.source == s).to_numpy()] >= warn).mean()), 4),
                  "boxes_pass": round(float((nm[(neg.source == s).to_numpy()] >= boxes).mean()), 4)} for s in sorted(neg.source.unique())}
    check = {"scored_with": what, "size": size, "n_plant": len(plants), "n_toxic": len(toxic), "n_nonplant": len(negs),
             "auroc_clean": round(auroc(pc, nm), 4), "auroc_degraded": round(auroc(pdg, nm), 4),
             "plant_mass_p5_degraded": round(float(np.quantile(pdg, 0.05)), 4), "nonplant_mass_p50": round(float(np.median(nm)), 4),
             "warn_keep_plants_degraded": round(float((pdg >= warn).mean()), 4), "warn_keep_plants_clean": round(float((pc >= warn).mean()), 4),
             "boxes_keep_plants_degraded": round(float((pdg >= boxes).mean()), 4), "boxes_keep_plants_clean": round(float((pc >= boxes).mean()), 4),
             "toxic_warning_miss_no_gate": round(float((~fires).mean()), 4), "toxic_warning_miss_added_by_gate": round(added(warn), 4),
             "nonplant_warn_gate_passes": round(float((nm >= warn).mean()), 4), "nonplant_boxes_gate_passes": round(float((nm >= boxes).mean()), 4),
             "by_source": by_src, "warn_keep": args.warn_keep, "max_added_miss": args.max_added_miss, "boxes_keep": args.boxes_keep}
    res = {"presence_warn": round(warn, 4), "presence_boxes": round(boxes, 4), "check": check}
    (d / "presence.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

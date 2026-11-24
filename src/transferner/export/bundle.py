"""Build the app's first-launch download bundle (D-016, app/README.md), and the app's unit-test goldens.

Layout under $TRANSFERNER_DATA/bundle/ (served as is by scripts/serve_bundle.sh):
  latest.json                       {"version": V, "manifest": "V/manifest.json"}
  V/manifest.json                   every file below with bytes and sha256; the app verifies each before use
  V/detector.tflite + detector.json live detector, app graph (export/detector_app.py) and its sidecar
  V/focus.json                      live focus chip: genera per growth form + prior strength (D-037), optional
  V/species_<id>.tflite + species.json   BioCLIP 2.5 image tower, one file per variant (VARIANTS: gpu W8A16,
                                    cpu dynamic INT8, cpu-small INT4 + dynamic INT8; D-032/D-035). Manifest
                                    group "species:<id>": a phone downloads only its own variant
  V/head.bin + head.json            production probe: W as float16 [n, 1024] row-major little-endian, bias,
                                    logit scale and temperature T (logits = (scale * e.W^T + b) / T)
  V/catalog.json                    per species: names, genus, family, look-alike groups, cited trait rows,
                                    thumbnail and its attribution
  V/thumbs/<class_idx>.jpg          one CC0 / CC BY 4.0 field photo per species where one exists (D-019)
  V/text_emb.bin                    BioCLIP text embeddings of the species names, float16 [n, 1024]: the capture
                                    "not in my list" score (D-044); head.json names it (text_file, text_dim)
  V/forage.json                     walk-mode targets (D-044): every detector genus with a cited edible or
                                    conditional row, its default tier (data/forage/default_tiers.csv: alert, show
                                    or off; unlisted genera off) and weight e_g = share of the genus' regional
                                    sightings that are species with such a row
  V/tuning.json                     live and capture thresholds (D-044); presence gate from eval/presence_gate.py

Toxicity comes only from cited trait rows (D-004/D-005/D-027), never from species.parquet's is_toxic flag
(it disagrees with the trait table, e.g. Delphinium tricorne). Runs in the CPU export venv:
  cd /tmp && PYTHONPATH=~/Transferner/src ~/Transferner/.venv-export/bin/python -m transferner.export.bundle \\
      --version 0.1 [--goldens $TRANSFERNER_DATA/app_golden]   (goldens: data, so never in git)"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from ..utils.config import data_root

THUMB_LICENSES = ("cc-0-1.0", "cc-by-4.0")
LICENSE_NAMES = {"cc-0-1.0": "CC0 1.0", "cc-by-4.0": "CC BY 4.0"}
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]      # open_clip_config.json of imageomics/bioclip-2.5-vith14
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_notes(s: str) -> dict:
    """species_traits.csv notes: '; '-joined segments ('severity High', 'toxic principle: ...', 'symptoms: ...',
    'genus-level page (X), species not individually assessed', ...)."""
    out = {"toxic_principle": "", "symptoms": "", "genus_level": False, "remarks": []}
    if not isinstance(s, str) or not s:
        return out
    if "symptoms:" in s:                                 # symptoms come last and contain '; ' themselves
        s, out["symptoms"] = s.split("symptoms:", 1)
        out["symptoms"] = out["symptoms"].strip()
    if "toxic principle:" in s:                          # comes right before symptoms; may contain '; ' too
        s, tp = s.split("toxic principle:", 1)
        tp = tp.strip().rstrip(";").strip()
        out["toxic_principle"] = "" if tp in ("", ".") else tp
    for seg in (t.strip() for t in s.split("; ") if t.strip()):
        low = seg.lower()
        if low.startswith("severity"):
            continue                                                     # already in ingestion_severity
        if low.startswith("toxic principle:"):
            v = seg.split(":", 1)[1].strip()
            out["toxic_principle"] = "" if v in ("", ".") else v
        elif "genus-level page" in low:
            out["genus_level"] = True
        else:
            out["remarks"].append(seg)
    return out


def tflite_io(path: Path) -> dict:
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    key = next(iter(it.get_signature_list()))
    r = it.get_signature_runner(key)
    q = lambda d: {"dtype": np.dtype(d["dtype"]).name, "shape": [int(s) for s in d["shape"]],
                   "scale": float(d["quantization"][0]), "zero_point": int(d["quantization"][1])}
    ins, outs = r.get_input_details(), r.get_output_details()
    return {"signature": key, "inputs": {n: q(d) for n, d in ins.items()}, "outputs": {n: q(d) for n, d in outs.items()}}


def catalog(root: Path, out: Path, repo: Path, thumb_px: int) -> list[dict]:
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx").reset_index(drop=True)
    tr = pd.read_csv(repo / "data/traits/species_traits.csv")
    la = pd.read_csv(repo / "data/traits/lookalike_groups.csv")
    im = pd.read_parquet(root / "manifests/images.parquet",
                         columns=["uuid", "class_idx", "license", "rights_holder", "is_specimen", "path", "source",
                                  "occurrence_ref", "url", "pick_rank"])
    spl = pd.read_parquet(root / "manifests/splits.parquet", columns=["uuid", "split"])
    im = im.merge(spl, on="uuid")
    im = im[im.license.isin(THUMB_LICENSES) & ~im.is_specimen.fillna(False).astype(bool) & im.rights_holder.notna()
            & im.split.isin(["train", "val"])]
    # field photos first (regional, then any), then the selection's own ranking
    im = im.assign(o=(im.source != "gbif_regional").astype(int), s=(im.split != "train").astype(int)) \
        .sort_values(["class_idx", "o", "s", "pick_rank"]).groupby("class_idx").head(1).set_index("class_idx")
    groups = la.groupby("scientific_name").group_id.apply(list).to_dict()
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    traits = {n: g for n, g in tr.groupby("scientific_name")}
    entries = []
    for r in sp.itertuples():
        e = {"i": int(r.class_idx), "name": r.scientific_name,
             "common": r.common_name if isinstance(r.common_name, str) else "",
             "genus": r.genus, "family": r.family, "lookalike_groups": groups.get(r.scientific_name, []),
             "traits": [], "thumb": None}
        for t in traits.get(r.scientific_name, pd.DataFrame()).itertuples():
            n = parse_notes(t.notes)
            e["traits"].append({
                "part": t.part, "category": t.category, "contact_toxic": int(t.contact_toxic),
                "ingestion_toxic": int(t.ingestion_toxic),
                "severity": t.ingestion_severity if isinstance(t.ingestion_severity, str) else "",
                "conditions": t.conditions if isinstance(t.conditions, str) else "",
                "toxic_principle": n["toxic_principle"], "symptoms": n["symptoms"], "genus_level": n["genus_level"],
                "remarks": n["remarks"], "source": t.source_name, "url": t.source_url, "accessed": t.accessed})
        if r.class_idx in im.index:
            m = im.loc[r.class_idx]
            f = f"thumbs/{int(r.class_idx)}.jpg"
            ImageOps.fit(Image.open(root / m.path).convert("RGB"), (thumb_px, thumb_px)).save(out / f, quality=80)
            ref = m.occurrence_ref if isinstance(m.occurrence_ref, str) else (m.url if isinstance(m.url, str) else "")
            e["thumb"] = {"file": f, "by": m.rights_holder, "license": LICENSE_NAMES[m.license], "ref": ref}
        entries.append(e)
    return entries


def goldens(root: Path, gdir: Path, uuid: str, det_path: Path, det_side: dict, sp_path: Path, head: dict) -> None:
    """Reference tensors for the app's JVM unit tests on one photo: the detector's int8 input (torchvision
    Resize((320, 320), antialias) on the decoded JPEG, as in training), its LiteRT outputs and decoded top
    boxes; BioCLIP's float input (open_clip val transform: shortest side to 224 bicubic, centre crop), its
    LiteRT embedding, and the head's top 5."""
    from ai_edge_litert.interpreter import Interpreter
    from torchvision.io import read_image
    from torchvision.transforms import v2
    from torchvision.ops import nms
    from torchvision.models.detection._utils import BoxCoder
    gdir.mkdir(parents=True, exist_ok=True)
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    src = root / im[im.uuid == uuid].iloc[0].path
    shutil.copy(src, gdir / "photo.jpg")
    # detector
    s = det_side["size"]
    img = v2.Resize((s, s), antialias=True)(read_image(str(src)))                       # uint8 (3, s, s)
    x = (img.float() / 255 - torch.tensor(det_side["mean"]).view(3, 1, 1)) / torch.tensor(det_side["std"]).view(3, 1, 1)
    qi = det_side["input"]
    xq = np.clip(np.round(x.numpy() / qi["scale"]) + qi["zero_point"], -128, 127).astype(np.int8)
    (gdir / "detector_rgb320.u8").write_bytes(img.numpy().astype(np.uint8).tobytes())  # CHW
    (gdir / "detector_input.i8").write_bytes(xq.tobytes())
    it = Interpreter(model_path=str(det_path))
    r = it.get_signature_runner(det_side["signature"])
    o = r(**{qi["name"]: xq[None]})
    deq = {}
    for k, d in det_side["outputs"].items():
        a = o[d["name"]].astype(np.float32)
        deq[k] = (a - d["zero_point"]) * d["scale"] if d["dtype"] == "int8" else a
    (gdir / "detector_image_logits.f32").write_bytes(deq["image_logits"].astype("<f4").tobytes())
    (gdir / "detector_objectness.f32").write_bytes(deq["objectness"].astype("<f4").tobytes())
    (gdir / "detector_boxes.i8").write_bytes(o[det_side["outputs"]["boxes"]["name"]].astype(np.int8).tobytes())
    anchors = torch.tensor(det_side["anchors"])
    bc = BoxCoder(det_side["box_coder"]["weights"])
    dec = bc.decode_single(torch.from_numpy(deq["boxes"][0]), anchors).clamp(0, s)
    ob = torch.from_numpy(deq["objectness"][0])
    sc, a = ob.topk(det_side["postprocess"]["objectness_topk"])
    keep = nms(dec[a], sc, det_side["postprocess"]["nms_iou"])[:5]
    p = torch.from_numpy(deq["image_logits"][0]).softmax(-1)
    pv, pi = p.topk(5)
    det_g = {"genus_top5": [[det_side["genera"][int(j)], float(v)] for j, v in zip(pi, pv)],
             "boxes_top5": [[float(c) for c in dec[a[k]]] + [float(sc[k])] for k in keep]}
    # species model
    pil = Image.open(src).convert("RGB")
    tf = v2.Compose([v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), v2.CenterCrop(224)])
    crop = tf(pil)
    xs = (torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255
          - torch.tensor(CLIP_MEAN).view(3, 1, 1)) / torch.tensor(CLIP_STD).view(3, 1, 1)
    (gdir / "species_input.f32").write_bytes(xs.numpy().astype("<f4").tobytes())
    sio = tflite_io(sp_path)
    it2 = Interpreter(model_path=str(sp_path))
    r2 = it2.get_signature_runner(sio["signature"])
    (in_name,), (out_name,) = sio["inputs"], sio["outputs"]
    emb = r2(**{in_name: xs.numpy()[None]})[out_name][0].astype(np.float32)
    (gdir / "species_embedding.f32").write_bytes(emb.astype("<f4").tobytes())
    W = torch.from_numpy(np.fromfile(head["_bin"], dtype="<f2").reshape(head["n"], head["dim"]).astype(np.float32))
    logits = (head["scale"] * torch.from_numpy(emb) @ W.T + torch.tensor(head["bias"])) / head["temperature"]
    sv, si = logits.softmax(-1).topk(5)
    json.dump({"uuid": uuid, "bundle": det_path.parent.name, "detector": det_g, "species_top5": [[int(j), float(v)] for j, v in zip(si, sv)],
               "embedding_norm": float(np.linalg.norm(emb))}, open(gdir / "golden.json", "w"), indent=1)
    print("goldens:", json.load(open(gdir / "golden.json")), flush=True)


VARIANTS = {   # D-032 per-runtime choice; D-035 fallback chain gpu -> cpu -> cpu-small
    "gpu": {"id": "gpu", "label": "INT8 weights, FP16 math (GPU)", "file": "artifacts/export/bioclip_visual_w8.tflite",
            "accelerator": "GPU", "min_ram_gb": 10},
    "cpu": {"id": "cpu", "label": "Dynamic INT8 (CPU)", "file": "artifacts/export/bioclip_visual_w8dyn.tflite",
            "accelerator": "CPU", "min_ram_gb": 6},
    "cpu-small": {"id": "cpu-small", "label": "INT4 weights, dynamic INT8 (CPU, low memory)",
                  "file": "artifacts/export/bioclip_visual_w4dyn.tflite", "accelerator": "CPU", "min_ram_gb": 0},
}


def variant_check(root: Path, model: Path, head: dict, head_bin: Path, uuid: str) -> dict:
    """The variant on the golden photo through the open_clip transform: its top 5 under the head, and the
    embedding cosine to the GPU variant's (the reference the app's unit tests use)."""
    from ai_edge_litert.interpreter import Interpreter
    from torchvision.transforms import v2
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    pil = Image.open(root / im[im.uuid == uuid].iloc[0].path).convert("RGB")
    crop = v2.Compose([v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), v2.CenterCrop(224)])(pil)
    x = ((torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float() / 255 - torch.tensor(CLIP_MEAN).view(3, 1, 1))
         / torch.tensor(CLIP_STD).view(3, 1, 1)).numpy()[None]
    io = tflite_io(model)
    r = Interpreter(model_path=str(model), num_threads=4).get_signature_runner(io["signature"])
    (i,), (o,) = io["inputs"], io["outputs"]
    t = time.perf_counter()
    e = r(**{i: x})[o][0].astype(np.float32)
    ms = 1000 * (time.perf_counter() - t)
    W = np.fromfile(head_bin, dtype="<f2").reshape(head["n"], head["dim"]).astype(np.float32)
    lg = (head["scale"] * W @ e + np.array(head["bias"], np.float32)) / head["temperature"]
    p = np.exp(lg - lg.max()); p /= p.sum()
    top = np.argsort(-p)[:5]
    ref = root / "app_golden/species_embedding.f32"
    cos = float(e @ np.fromfile(ref, dtype="<f4")) if ref.exists() else None
    return {"top5": [[int(k), round(float(p[k]), 4)] for k in top], "cos_to_gpu_variant": cos, "x86_ms_4_threads": round(ms)}


def forage(root: Path, repo: Path, genera: list[str]) -> dict:
    """Walk-mode targets (D-044; audit edible_targeting_app-01/03): each detector genus holding a species with a
    cited edible or conditional row, weighted by that species group's share of the genus' regional sightings."""
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["gbif_taxon_key", "scientific_name", "genus", "common_name"])
    tr = pd.read_csv(repo / "data/traits/species_traits.csv")
    ed = set(tr[tr.category.isin(["edible", "conditional"])].scientific_name)
    pr = pd.read_parquet(root / "manifests/priors_state_month.parquet").groupby("gbif_taxon_key")["count"].sum()
    sp["n"] = sp.gbif_taxon_key.map(pr).fillna(0) + 1
    sp["ed"] = sp.scientific_name.isin(ed)
    tiers = pd.read_csv(repo / "data/forage/default_tiers.csv", keep_default_na=False).set_index("genus")
    gi = {g: i for i, g in enumerate(genera)}
    out = []
    for g, d in sp.groupby("genus"):
        if g not in gi or not d.ed.any():
            continue
        row = tiers.loc[g] if g in tiers.index else None
        common = d[d.ed].sort_values("n", ascending=False).common_name.dropna()
        # genera the curated list leaves out default to off: several are mostly toxic (Hydrangea, Trillium,
        # Lathyrus) and would otherwise read as "Maybe <name>" chips (0.6 review); the user can switch them on
        out.append({"g": gi[g], "genus": g, "tier": row.tier if row is not None else "off",
                    "weight": round(float(d.n[d.ed].sum() / d.n.sum()), 4),
                    "label": (row.label if row is not None and row.label else (common.iloc[0] if len(common) else g)).strip(),
                    "note": row.note if row is not None else ""})
    missing = sorted(set(tiers.index) - {o["genus"] for o in out})
    assert not missing, f"default_tiers.csv names genera without a cited edible row in the detector: {missing}"
    tally = pd.Series([o["tier"] for o in out]).value_counts().to_dict()
    print(f"forage: {len(out)} genera {tally}", flush=True)
    return {"genera": sorted(out, key=lambda o: o["genus"]), "source": "data/traits/species_traits.csv (NC State Extension "
            "Plant Toolbox), regional GBIF sightings, data/forage/default_tiers.csv"}


def tuning(det_dir: Path) -> dict:
    """Thresholds the app reads from tuning.json (D-044). The presence gate comes from eval/presence_gate.py for
    this detector; without it the gate is off. The rest are the read-only audit's values (app Tuning defaults)."""
    t = {"presence_boxes": 0.0, "presence_warn": 0.0, "secondary_offset": 0.15, "secondary_floor": 0.4, "containment": 0.8,
         "label_tau_s": 0.56, "warn_hold_frames": 4, "walk_fps": 4, "alert_tau_s": 1.0, "alert_enter": 0.5, "alert_hold_s": 0.75,
         "alert_exit": 0.35, "chip_min": 0.3, "refractory_s": 600, "toxic_haptic": 0.3, "toxic_hold_s": 1.0, "verify_genus_p": 0.25,
         "text_min": 0.561, "banner_sum": 0.002, "banner_strong": 0.02, "conditional_min": 0.02, "species_min": 0.8, "genus_min": 0.8}
    gate = det_dir / "presence.json"
    if gate.exists():
        g = json.loads(gate.read_text())
        t.update(presence_boxes=g["presence_boxes"], presence_warn=g["presence_warn"])
        t["presence_check"] = g.get("check", {})
    print(f"tuning: presence gate boxes {t['presence_boxes']}, warnings {t['presence_warn']}", flush=True)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.2")
    ap.add_argument("--detector", default="artifacts/detector/mnv4_shared/detector_app_int8")
    ap.add_argument("--variants", default="gpu,cpu,cpu-small", help="species-model variants to ship (VARIANTS); missing files are skipped")
    ap.add_argument("--probe", default="artifacts/probe.pt")
    ap.add_argument("--temperature", default="artifacts/temperature.txt")
    ap.add_argument("--thumb-px", type=int, default=256)
    ap.add_argument("--goldens", default="", help="write the app's unit-test goldens to this directory")
    ap.add_argument("--no-publish", action="store_true", help="build the version directory but leave latest.json (staging)")
    ap.add_argument("--golden-uuid", default="gbif-6178854992-1")
    ap.add_argument("--text-emb", default="artifacts/text_emb.npy", help="species-name embeddings for the open-set score ('' = none)")
    args = ap.parse_args()
    root = data_root()
    repo = Path(__file__).resolve().parents[3]
    base = root / "bundle"
    out = base / args.version
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    t0 = time.time()

    det_src = root / (args.detector + ".tflite")
    side = json.loads((root / (args.detector + ".json")).read_text())
    shutil.copy(det_src, out / "detector.tflite")
    side["model"] = "detector.tflite"
    (out / "detector.json").write_text(json.dumps(side))
    focus = (root / args.detector).parent / "focus.json"                 # eval/focus_prior.py (D-037), optional
    if focus.exists():
        f = json.loads(focus.read_text())
        f.pop("check", None)
        (out / "focus.json").write_text(json.dumps(f))

    pr = torch.load(root / args.probe, map_location="cpu")
    W = pr["W"].float().numpy()
    W.astype("<f2").tofile(out / "head.bin")
    head = {"file": "head.bin", "n": int(W.shape[0]), "dim": int(W.shape[1]), "dtype": "float16",
            "scale": float(pr["scale"]), "temperature": float((root / args.temperature).read_text()),
            "bias": [float(v) for v in pr["b"]], "probe": args.probe}
    if args.text_emb:
        t = np.load(root / args.text_emb).astype(np.float32)
        assert t.shape[0] == head["n"], (t.shape, head["n"])
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        t.astype("<f2").tofile(out / "text_emb.bin")
        head.update(text_file="text_emb.bin", text_dim=int(t.shape[1]), text_source=args.text_emb)
    (out / "head.json").write_text(json.dumps(head))
    (out / "forage.json").write_text(json.dumps(forage(root, repo, side["genera"])))
    (out / "tuning.json").write_text(json.dumps(tuning((root / args.detector).parent)))

    variants, groups = [], {}
    for v in (VARIANTS[k] for k in args.variants.split(",")):
        src = root / v["file"]
        if not src.exists():
            print(f"variant {v['id']}: {src} missing, skipped", flush=True)
            continue
        name = f"species_{v['id']}.tflite"
        os.link(src, out / name)                                         # 0.3-0.6 GB: hard link, not a copy
        sio = tflite_io(src)
        (in_name, ind), = sio["inputs"].items()
        (out_name, outd), = sio["outputs"].items()
        variants.append({
            "id": v["id"], "label": v["label"], "model": name, "variant": src.stem, "accelerator": v["accelerator"],
            "min_ram_gb": v["min_ram_gb"], "bytes": src.stat().st_size, "signature": sio["signature"],
            "input": {"name": in_name, **ind}, "output": {"name": out_name, **outd},
            "preprocess": {"resize": "shortest side to 224, bicubic, antialiased", "crop": "centre 224 x 224",
                           "mean": CLIP_MEAN, "std": CLIP_STD, "layout": "NCHW"},
            "output_normalised": True, "check": variant_check(root, src, head, out / "head.bin", args.golden_uuid)})
        groups[name] = f"species:{v['id']}"
        print(f"variant {v['id']}: {name} {src.stat().st_size / 2**20:.0f} MB, golden check {variants[-1]['check']}", flush=True)
    assert variants, "no species-model variant found"
    (out / "species.json").write_text(json.dumps({"variants": variants}))

    cat = catalog(root, out, repo, args.thumb_px)
    assert len(cat) == head["n"]
    (out / "catalog.json").write_text(json.dumps({"species": cat, "region": ["SC", "NC", "GA", "TN"],
                                                  "trait_rule": "cited rows only; absence never means safe (D-027)"}))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {"version": args.version, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "files": [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p),
                           **({"group": groups[p.name]} if p.name in groups else {})} for p in files]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=0))
    if not args.no_publish:
        (base / "latest.json").write_text(json.dumps({"version": args.version, "manifest": f"{args.version}/manifest.json"}))
    total = sum(f["bytes"] for f in manifest["files"])
    n_thumb = sum(1 for e in cat if e["thumb"])
    n_tr = sum(1 for e in cat if e["traits"])
    print(f"bundle {args.version}: {len(files)} files, {total / 2**20:.0f} MB; thumbnails {n_thumb}/{len(cat)}, "
          f"species with trait rows {n_tr}/{len(cat)} ({time.time() - t0:.0f} s)", flush=True)
    if args.goldens:
        goldens(root, Path(args.goldens).expanduser(), args.golden_uuid, out / "detector.tflite", side,
                out / variants[0]["model"], {**head, "_bin": out / "head.bin"})


if __name__ == "__main__":
    main()

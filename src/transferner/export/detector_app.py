"""Live detector graph for the Android app (app prototype, 2026-11-23; D-033 follow-up).

The training/export graph (detector_litert.Export) emits SSD's raw layout: (1, A, G + 1) class logits with
each location's genus logits repeated over its 6 anchors, 4.4M values per frame at 1,359 genera. The app
needs far less, so this graph emits:
  image_logits  (1, G)     whole-frame genus logits (pooled stride-32 features -> the shared classifier).
                           The live label: 56.2% test top-1 against 49.1% for the best box (D-033).
  objectness    (1, A)     1 - P(background) per anchor = sigmoid(logsumexp_g(genus logits) - bg logit),
                           identical to softmax over [bg, genera] but computed per location, not per anchor.
  boxes         (1, A, 4)  SSD box regression; the app decodes against the anchors in the sidecar
                           (torchvision BoxCoder, weights 10, 10, 5, 5) and runs one class-agnostic NMS.
  region_logits (1, 9, G)  with --regions (D-044): the whole-frame head on 9 overlapping windows of the stride-32
                           map (3 x 3, each about 57% of the frame per side), +1.4% compute. Walk mode names
                           and draws the window behind an edible alert (read-only audit, edible_targeting_app-05).
Input as in detector_litert: normalised (1, 3, 320, 320) NCHW; full INT8 has int8 input, so the app quantises
pixels with a per-channel lookup table built from the sidecar's mean, std, scale and zero point.

Writes artifacts/detector/<name>/detector_app_<tag>.tflite and detector_app_<tag>.json (the sidecar: I/O
names, dtypes and quantisation, anchors, box coder, genera/families, and this script's own check), and a row
to results/tables/export_check.csv; --out-dir writes the two files elsewhere (a trial run that must not
replace the staged graph). The check scores the float graph and the converted one on the same photos and pairs
them (eval.metrics.paired_agreement: McNemar discordant counts, paired CI, top-1 agreement, KL): D-041 read a
-3.7 pt INT8 loss from 512 photos that was -1.08 +- 0.38 pt on the full regional test (audit
live_int8_precision-loss-overstated), so the default check is that test (--split test --images 20000, D-043).
Runs in the CPU export venv:
  cd /tmp && PYTHONPATH=~/Transferner/src ~/Transferner/.venv-export/bin/python -m transferner.export.detector_app"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import nms

from ..eval.metrics import coco_map, paired_agreement
from ..train.train_detector import BoxSet, Net, anchors_for, build, collate, load_items
from ..utils.config import data_root, load
from .litert import mem_guard, peak_rss_gb, static_quantizer, write

OUTPUTS = ("image_logits", "objectness", "boxes")


def region_grid(h: int) -> tuple[int, int]:
    """Window size k and stride s (in stride-32 cells) for 3 x 3 overlapping windows on an h x h map:
    s = h // 4, k = h - 2s (10 -> 6/2 at 320 px, 14 -> 8/3 at 448 px)."""
    s = h // 4
    return h - 2 * s, s


class AppExport(nn.Module):
    def __init__(self, ssd, float_image_head: bool = False, regions: bool = False):
        super().__init__()
        self.backbone, self.head = ssd.backbone, ssd.head
        self.regions = regions
        if regions:
            import copy
            self.region_cls = copy.deepcopy(ssd.head.image_cls[1:])
        self.float_image_head = False
        ic = getattr(self.head, "image_cls", None)
        if float_image_head and ic is not None:
            # The dedicated whole-frame head (pool -> 1x1 conv + BN -> ReLU -> 1x1 conv) rewritten as mean + matmul,
            # which the LiteRT PT2E quantizer does not annotate (its static ops are conv/linear/add/mul/cat/pools):
            # the head runs in float on the dequantized stride-32 features (~3M MACs per frame).
            conv1, bn, conv2 = ic[1], ic[2], ic[5]
            k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            self.register_buffer("w1", (conv1.weight.flatten(1) * k[:, None]).detach().clone())
            self.register_buffer("b1", (bn.bias - bn.running_mean * k).detach().clone())
            self.register_buffer("w2", conv2.weight.flatten(1).detach().clone())
            self.register_buffer("b2", conv2.bias.detach().clone())
            self.float_image_head = True
        assert getattr(self.head, "image_embed", None) is not None or getattr(self.head, "image_cls", None) is not None, \
            "needs a shared head with a whole-frame branch (--aux-image or train_live_head)"

    def forward(self, x):
        feats = list(self.backbone(x).values())
        h = self.head
        obj = []
        for f, bg, em in zip(feats, h.bg, h.embed):
            b = bg(f)                                                    # (B, a, H, W) background logits
            g = h.genus(em(f))                                           # (B, G, H, W) genus logits per location
            o = torch.sigmoid(torch.logsumexp(g, dim=1, keepdim=True) - b)
            obj.append(o.permute(0, 2, 3, 1).reshape(x.shape[0], -1))   # anchor order = SSD's (location, shape)
        if self.float_image_head:
            x = feats[1].mean(dim=(2, 3))
            image_logits = torch.matmul(torch.relu(torch.matmul(x, self.w1.t()) + self.b1), self.w2.t()) + self.b2
        else:
            image_logits = h.image_logits(feats)
        if not self.regions:
            return image_logits, torch.cat(obj, 1), h.regression_head(feats)
        # The windows go through their own copy of the head (self.region_cls): a second call of the same module
        # kept a float copy of its weights (14 -> 25 MB), and one shared call gave the whole-frame logits the
        # windows' coarser int8 scale (0.27 -> 0.60). A copy is quantised on its own (+3 MB).
        k, s = region_grid(feats[1].shape[-1])
        r = self.region_cls(F.avg_pool2d(feats[1], kernel_size=k, stride=s))  # (B, G, 3, 3)
        regions = r.flatten(2).permute(0, 2, 1)                                 # (B, 9, G), row-major windows
        return image_logits, torch.cat(obj, 1), h.regression_head(feats), regions


def score(run, loader, box_coder, anchors0, size, family_of, mean, std, keep=100, iou=0.55, sink=None) -> dict:
    """run(x) -> (image_logits, objectness, boxes) for a normalised batch: whole-frame genus/family top-k
    and class-agnostic box AP, the two things the app shows. A dict as sink collects the whole-frame logits
    and genus labels, so two runs over the same loader can be paired (float vs INT8)."""
    fam_t = torch.as_tensor(family_of)
    n_fam = int(family_of.max()) + 1
    top1 = top5 = fam1 = n = 0
    det = {k: [] for k in ("img", "score", "box")}
    gt = {k: [] for k in ("img", "box")}
    for imgs, boxes, labels, genus in loader:
        x = ((imgs.float() / 255) - mean) / std
        il, ob, bx = run(x)
        y = genus - 1
        if sink is not None:
            sink.setdefault("logits", []).append(il.float())
            sink.setdefault("y", []).append(y)
        p = il.float().softmax(-1)
        top1 += int((p.argmax(1) == y).sum())
        top5 += int((p.topk(5, 1).indices == y[:, None]).any(1).sum())
        fam1 += int((torch.zeros(len(y), n_fam).index_add_(1, fam_t, p).argmax(1) == fam_t[y]).sum())
        B, A = ob.shape
        dec = box_coder.decode_single(bx.float().reshape(-1, 4), anchors0.repeat(B, 1)).reshape(B, A, 4).clamp(0, size)
        for i in range(B):
            s, a = ob[i].float().topk(300)
            k = nms(dec[i, a], s, iou)[:keep]
            det["img"].append(np.full(len(k), n + i)); det["score"].append(s[k].numpy()); det["box"].append(dec[i, a[k]].numpy())
            gt["img"].append(np.full(len(boxes[i]), n + i)); gt["box"].append(boxes[i].numpy())
        n += B
    det = {k: np.concatenate(v) for k, v in det.items()}
    gt = {k: np.concatenate(v) for k, v in gt.items()}
    m = coco_map(det["img"], np.zeros(len(det["img"]), int), det["score"], det["box"],
                 gt["img"], np.zeros(len(gt["img"]), int), gt["box"])
    return {"agnostic_ap50": m["ap50"], "agnostic_map": m["map"], "image_genus_top1": top1 / n,
            "image_genus_top5": top5 / n, "image_family_top1": fam1 / n, "n_images": n}


def regions_meta(model, size: int) -> dict:
    """The 9 windows of region_logits as [x0, y0, x1, y1] fractions of the squashed frame, row-major."""
    with torch.no_grad():
        h = model.backbone(torch.zeros(1, 3, size, size))
    cells = list(h.values())[1].shape[-1]
    k, s = region_grid(cells)
    win = [[round(c * s / cells, 4), round(r * s / cells, 4), round((c * s + k) / cells, 4), round((r * s + k) / cells, 4)]
           for r in range(3) for c in range(3)]
    return {"grid": [3, 3], "cells": cells, "window_cells": k, "stride_cells": s, "windows": win}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_shared")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--boxes", default="manifests/boxes.parquet")
    ap.add_argument("--recipe", choices=["fp32", "int8"], default="int8")
    ap.add_argument("--act-observer", choices=["histogram", "minmax"], default="histogram")
    ap.add_argument("--float-image-head", action="store_true",
                    help="keep the whole-frame head (head.image_cls: pool -> 1280 -> genera, ~3M MACs) in float (tag _fh); "
                         "D-043: +0.1 pt on the regional test for 22.5 MB instead of 14 MB")
    ap.add_argument("--regions", action="store_true", help="add the (1, 9, G) region_logits output (tag _rg)")
    # D-043 / audit live_int8_precision-loss-overstated: the old default (512 val photos, 93% ToL) put the INT8
    # loss at -3.7 pt where the full regional test says -1.08 +- 0.38; at most 20,000 covers the 12,358 test photos
    ap.add_argument("--images", type=int, default=20000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=None,
                    help="write the .tflite and sidecar here instead of artifacts/detector/<name> (trial runs)")
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mem_guard(args.min_free_gb)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    root, cfg = data_root(), load("detector")
    d = root / "artifacts/detector" / args.name
    labels = json.loads((d / "labels.json").read_text())
    ck = torch.load(d / args.ckpt, map_location="cpu")
    size = ck["args"].get("size") or cfg["model"]["input"]
    model = build(len(labels["genera"]), cfg["model"]["backbone"], size, pretrained=False,
                  head=ck["args"].get("head", "ssdlite"), image_aux=ck["args"].get("aux_image", 0) > 0,
                  image_head=ck["args"].get("image_head", 0))
    model.load_state_dict(ck["model"])
    model.eval()
    genera, family_of, families, splits = load_items(root, args.boxes, ck["args"].get("fallback", "keep"), 0, args.seed)
    assert genera == labels["genera"]
    rng = np.random.default_rng(args.seed)
    pick = lambda items, n: [items[i] for i in sorted(rng.permutation(len(items))[:n])]
    check, calib = pick(splits[args.split], args.images), pick(splits["train"], args.calib)
    mean = torch.tensor(model.backbone.mean).view(1, 3, 1, 1)
    std = torch.tensor(model.backbone.std).view(1, 3, 1, 1)
    loader = lambda items, bs: list(DataLoader(BoxSet(root, items, size, False), batch_size=bs, num_workers=4,
                                               collate_fn=collate))
    dl = loader(check, 16)                     # materialised before LiteRT/JAX is imported (see detector_litert)
    cx = torch.cat([((imgs.float() / 255) - mean) / std for imgs, *_ in loader(calib, 32)])
    anchors0 = anchors_for(model, size, "cpu")
    run_score = lambda run, sink=None: score(run, dl, model.box_coder, anchors0, size, family_of, mean, std, sink=sink)
    fl_sink, q_sink = {}, {}

    exp = AppExport(model, args.float_image_head, args.regions).eval()
    if args.regions:
        assert model.head.image_cls is not None, "--regions needs the whole-frame head (train_live_head)"
    with torch.no_grad():                      # the app graph must equal the training graph's postprocessing
        x = cx[:4]
        il, ob, bx = exp(x)[:3]
        ref = Net(model)(x)
        ref_obj = 1 - ref["cls_logits"].float().softmax(-1)[..., 0]
        err = float((ob - ref_obj).abs().max())
        assert err < 1e-4 and torch.equal(bx, ref["bbox_regression"]) and torch.allclose(il, ref["image_logits"], atol=1e-5), err
        print(f"app graph == training graph (objectness max err {err:.1e})", flush=True)
        t0 = time.time()
        fl = run_score(lambda x: exp(x)[:3], fl_sink)
    print("float:", fl, f"({time.time() - t0:.0f} s)", flush=True)

    tag = args.recipe + ("_minmax" if args.recipe == "int8" and args.act_observer == "minmax" else "") + \
        ("_fh" if args.float_image_head and args.recipe == "int8" else "") + ("_rg" if args.regions else "")
    row = {"recipe": f"detector_{args.name}_app_{tag}", "torch": torch.__version__, "images": args.images,
           "split": args.split, "threads": args.threads, "epoch": ck["epoch"],
           **{f"float_{k}": round(v, 4) for k, v in fl.items() if k != "n_images"}}
    sample = cx[:1]
    quantizer = None
    if args.recipe == "int8":
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        quantizer = static_quantizer(False, args.act_observer)
        m = prepare_pt2e(torch.export.export(exp, (sample,)).module(), quantizer)
        with torch.no_grad():
            for i in range(len(cx)):
                m(cx[i:i + 1])
        m = convert_pt2e(m, fold_quantize=False)
        row.update(calib=args.calib, act_observer=args.act_observer, float_image_head=args.float_image_head)
    else:
        m = exp

    import litert_torch
    from litert_torch.quantize.quant_config import QuantConfig
    out_dir = Path(args.out_dir) if args.out_dir else d
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"detector_app_{tag}.tflite"
    t = time.time()
    litert_torch.convert(m, (sample,), quant_config=QuantConfig(pt2e_quantizer=quantizer) if quantizer else None).export(str(path))
    row.update(convert_s=round(time.time() - t), file_mb=round(path.stat().st_size / 2**20, 2))

    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path), num_threads=args.threads)
    it.allocate_tensors()
    sig = it.get_signature_list()
    key = next(iter(sig))
    runner = it.get_signature_runner(key)
    ins, outs = runner.get_input_details(), runner.get_output_details()
    names = OUTPUTS + (("region_logits",) if args.regions else ())
    assert len(ins) == 1 and len(outs) == len(names), (ins.keys(), outs.keys())
    in_name, ind = next(iter(ins.items()))
    G, A = len(genera), len(anchors0)
    by_shape = {(1, G): "image_logits", (1, A): "objectness", (1, A, 4): "boxes", (1, 9, G): "region_logits"}
    out_names = {by_shape[tuple(int(s) for s in o["shape"])]: n for n, o in outs.items()}
    assert set(out_names) == set(names), out_names
    q = lambda det: {"dtype": np.dtype(det["dtype"]).name, "shape": [int(s) for s in det["shape"]],
                     "scale": float(det["quantization"][0]), "zero_point": int(det["quantization"][1])}
    times = []

    def tfl(x):
        res = {k: [] for k in names}
        for i in range(len(x)):
            v = x[i:i + 1].numpy()
            if np.issubdtype(ind["dtype"], np.integer):
                s, z = ind["quantization"]
                info = np.iinfo(ind["dtype"])
                v = np.clip(np.round(v / s) + z, info.min, info.max)
            t1 = time.perf_counter()
            o = runner(**{in_name: v.astype(ind["dtype"])})
            times.append(time.perf_counter() - t1)
            for k in names:
                det = outs[out_names[k]]
                a = o[out_names[k]].astype(np.float32)
                if np.issubdtype(det["dtype"], np.integer):
                    a = (a - det["quantization"][1]) * det["quantization"][0]
                res[k].append(torch.from_numpy(a))
        return tuple(torch.cat(res[k]) for k in names)

    got = run_score(lambda x: tfl(x)[:3], q_sink)
    print("litert:", got, flush=True)
    yf, yq = torch.cat(fl_sink["y"]), torch.cat(q_sink["y"])
    assert torch.equal(yf, yq), "float and converted runs saw different photos"
    paired = paired_agreement(torch.cat(fl_sink["logits"]).numpy(), torch.cat(q_sink["logits"]).numpy(), yf.numpy())
    print(f"paired float vs {tag} on {paired['n']} {args.split} photos: right only in float {paired['ref_only']}, only "
          f"converted {paired['test_only']} (McNemar p {paired['mcnemar_p']:.2g}), top-1 "
          f"{100 * paired['delta_top1']:+.2f} +- {100 * paired['delta_ci95']:.2f} pt, agreement {paired['top1_agree']:.4f}, "
          f"KL(float||{tag}) {paired['kl_mean']:.4f}", flush=True)
    region_check = None
    if args.regions:                           # INT8 vs float region labels on 64 held-out frames of the check split
        hx = torch.cat([((imgs.float() / 255) - mean) / std for imgs, *_ in dl[:4]])[:64]
        with torch.no_grad():
            rf = exp(hx)[3]
        rq = tfl(hx)[3]
        region_check = {"split": args.split, "frames": len(hx), "top1_agree": round(float((rf.argmax(-1) == rq.argmax(-1)).float().mean()), 4),
                        "max_abs": round(float((rf - rq).abs().max()), 3)}
        print("regions int8 vs float:", region_check, flush=True)
    row.update(status="ok", x86_ms_median=round(1000 * float(np.median(times[1:])), 1), peak_rss_gb=round(peak_rss_gb(), 1),
               **{f"q_{k}": round(v, 4) for k, v in got.items() if k != "n_images"},
               **{f"delta_{k}": round(got[k] - fl[k], 4) for k in ("agnostic_ap50", "image_genus_top1", "image_family_top1")},
               paired_float_only=paired["ref_only"], paired_q_only=paired["test_only"],
               paired_mcnemar_p=float(f"{paired['mcnemar_p']:.3g}"), paired_delta_ci95=round(paired["delta_ci95"], 4),
               paired_top1_agree=round(paired["top1_agree"], 4), paired_kl_mean=round(paired["kl_mean"], 5),
               **({"out_dir": str(out_dir)} if args.out_dir else {}))
    write(row)

    side = {
        "format": 1, "model": path.name, "signature": key, "size": size, "layout": "NCHW",
        "mean": [float(v) for v in model.backbone.mean], "std": [float(v) for v in model.backbone.std],
        "resize": "squash to size x size, bilinear (training used torchvision Resize((s, s)), antialias)",
        "input": {"name": in_name, **q(ind)},
        "outputs": {k: {"name": out_names[k], **q(outs[out_names[k]])} for k in names},
        "box_coder": {"weights": [float(w) for w in model.box_coder.weights], "clip": math.log(1000.0 / 16),
                      "anchors": "xyxy pixels at size"},
        "anchors": [[round(float(v), 3) for v in a] for a in anchors0],
        "postprocess": {"objectness_topk": 300, "nms_iou": 0.55, "max_boxes": 100},
        "genera": genera, "families": families, "family_of": [int(v) for v in family_of],
        **({"regions": regions_meta(model, size)} if args.regions else {}),
        "check": {"split": args.split, "images": args.images, "float": fl, "litert": got, "regions": region_check,
                  "paired": paired, "checkpoint": f"{args.name}/{args.ckpt}", "epoch": int(ck["epoch"])},
    }
    (out_dir / f"detector_app_{tag}.json").write_text(json.dumps(side))
    print(f"wrote {path} and its sidecar", flush=True)


if __name__ == "__main__":
    main()

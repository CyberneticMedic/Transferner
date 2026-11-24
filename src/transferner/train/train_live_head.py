"""Live-label retrain (D-036): the detector's whole-frame genus head, distilled from BioCLIP on every photo.

App feedback (2026-11-23): fast, but the live label needs to be more accurate. The live label is the
detector's whole-frame genus readout (D-033/D-034), trained on only the 100,018 boxed photos (at most 100
per genus). Whole-frame genus needs no boxes, and 1.37M non-specimen training photos have cached BioCLIP
embeddings, so this fine-tunes the D-033 detector (mnv4_shared) with two streams per step:
  boxed     --batch-box photos with OWLv2 boxes: SSD box + class loss (keeps detection) and image CE;
  unboxed   --batch-img photos from all 1.37M: image loss only, (1 - kd) * CE(genus label) +
            kd * CE(BioCLIP genus distribution), the teacher being the production probe (D-030) on the
            cached embeddings, calibrated (T), summed to the detector's genera, top 10 kept.
Half the boxed samples are 2x2 mosaics (--mosaic) so the detector learns several smaller plants per frame, and
a mosaic val metric tracks it. The whole-frame branch becomes MobileNetV4's own classifier head (pool, 960 -> 1280, BN, ReLU, classifier),
initialised from the ImageNet checkpoint's head (--image-head). Both streams add low-light frames
(train_detector.low_light) to the walking-video augmentation. Selection: val whole-frame genus top-1 (all
of val; the regional slice is reported) or the encounter-weighted top-1 (--select), plus --box-weight x the
mosaic box AP50, so a step whose boxes are broken cannot win on the label alone (mnv4_live448 picked step 0,
the 320 px weights at 448: label 0.7648 vs 0.7643 after training, mosaic AP50 0.579 vs 0.675). Test once at
the end. Writes artifacts/detector/<name>/ (best.pt, final.pt) and a row to results/tables/detector_live.csv.
  python -m transferner.train.train_live_head --name mnv4_live
  python -m transferner.train.train_live_head --eval-only artifacts/detector/mnv4_live/best.pt --size 384"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from ..eval.metrics import coco_map
from ..utils.config import data_root, load
from ..utils.csvlog import RunLog
from .train_detector import (BoxSet, Net, anchors_for, build, collate, detect, evaluate, field_degrade, forward_raw,
                             load_items, photometric, python_headers_for_triton, ssd_loss_batched, use_shift_add_depthwise,
                             white_balance)


def teacher(root, out_path, genera, args) -> pd.DataFrame:
    """BioCLIP's calibrated species distribution on every cached training embedding, summed to the
    detector's genera (mass on other genera dropped, then renormalised), top 10. Cached."""
    if out_path.exists():
        z = np.load(out_path, allow_pickle=True)
        return pd.DataFrame({"uuid": z["uuid"], "tidx": list(z["tidx"]), "tp": list(z["tp"])})
    meta = pd.read_parquet(root / args.emb / "train.parquet")
    emb = np.load(root / args.emb / "train.npy", mmap_mode="r")
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx")
    gi = {g: i for i, g in enumerate(genera)}
    G = len(genera)
    s2g = torch.tensor([gi.get(g, G) for g in sp.genus], device="cuda")
    pr = torch.load(root / args.probe, map_location="cuda")
    W, b, scale = pr["W"].float(), pr["b"].float(), float(pr["scale"])
    T = float((root / args.temperature).read_text())
    idx, prob = [], []
    with torch.no_grad():
        for s in range(0, len(meta), 65536):
            e = torch.from_numpy(np.asarray(emb[s:s + 65536], dtype=np.float32)).cuda()
            p = ((scale * e @ W.T + b) / T).softmax(-1)
            g = torch.zeros(len(e), G + 1, device="cuda").index_add_(1, s2g, p)[:, :G]
            g = g / g.sum(1, keepdim=True).clamp(min=1e-8)
            v, k = g.topk(10, 1)
            idx.append(k.short().cpu().numpy()); prob.append((v / v.sum(1, keepdim=True)).half().cpu().numpy())
    idx, prob = np.concatenate(idx), np.concatenate(prob)
    np.savez(out_path, uuid=meta.uuid.to_numpy(), tidx=idx, tp=prob)
    print(f"teacher: {len(meta):,} photos, top-10 genus mass kept", flush=True)
    return pd.DataFrame({"uuid": meta.uuid.to_numpy(), "tidx": list(idx), "tp": list(prob)})


class ImgSet(Dataset):
    """Unboxed photos for the whole-frame loss: random resized crop (walking framing), flip, photometric
    distortion, then the field artefacts incl. low light."""

    def __init__(self, root, paths, labels, tidx, tp, size, low_light_p, photo="ssd", wb_p=0.0):
        self.root, self.paths, self.labels, self.tidx, self.tp = root, paths, labels, tidx, tp
        self.size, self.low_light_p, self.wb_p = size, low_light_p, wb_p
        self.tf = v2.Compose([v2.RandomResizedCrop(size, scale=(0.2, 1.0), antialias=True), v2.RandomHorizontalFlip(),
                              photometric(photo)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(decode_image(str(self.root / self.paths[i]), mode=ImageReadMode.RGB))
        if self.wb_p and float(torch.rand(())) < self.wb_p:
            img = white_balance(img)
        img = field_degrade(img, self.size, self.low_light_p)
        return img, int(self.labels[i]), torch.from_numpy(self.tidx[i].astype(np.int64)), torch.from_numpy(self.tp[i].astype(np.float32))


class MosaicBoxSet(Dataset):
    """Boxed photos, and with probability p a 2x2 mosaic of four of them at half size (YOLO-style): the
    observation photos are close-ups whose box is mostly the whole frame (regional test: median top box 93% of
    the frame), so the live view drew no useful box. Mosaics teach several smaller plants per frame. A mosaic
    has no single genus, so its image label is 0 (ignored by the image loss). With train=False and p=1 it
    builds deterministic mosaics of consecutive photos for the mosaic validation metric."""

    def __init__(self, root, items, size, train, p, low_light_p=0.0, seed=0):
        self.full = BoxSet(root, items, size, train, low_light_p)
        self.tile = BoxSet(root, items, size // 2, train, low_light_p)
        self.size, self.p, self.train = size, p, train
        self.n = len(items) if train else len(items) // 4

    def __len__(self):
        return len(self.full) if self.train else self.n

    def __getitem__(self, i):
        if self.train and float(torch.rand(())) >= self.p:
            return self.full[i]
        idx = [i] + torch.randint(0, len(self.full), (3,)).tolist() if self.train else [4 * i + k for k in range(4)]
        h = self.size // 2
        img = torch.zeros(3, self.size, self.size, dtype=torch.uint8)
        boxes, labels = [], []
        for k, j in enumerate(idx):
            t, b, lab, _ = self.tile[j]
            oy, ox = (k // 2) * h, (k % 2) * h
            img[:, oy:oy + h, ox:ox + h] = t
            boxes.append(b + torch.tensor([ox, oy, ox, oy], dtype=b.dtype))
            labels.append(lab)
        return img, torch.cat(boxes), torch.cat(labels), 0


K_SOFT = 4


def paste(bg: torch.Tensor, crop: torch.Tensor, x0: int, y0: int, feather: int) -> torch.Tensor:
    """Alpha-blend crop (3, h, w) into bg at (x0, y0) with a linear ramp of `feather` px at the crop's edges,
    so there is no hard seam to learn (audit: hard-seam mosaics were 91% 'detectable' vs 40% feathered)."""
    _, h, w = crop.shape
    f = max(1, min(feather, h // 4, w // 4))
    ry = torch.clamp(torch.minimum(torch.arange(h) + 1, torch.arange(h).flip(0) + 1).float() / f, max=1.0)
    rx = torch.clamp(torch.minimum(torch.arange(w) + 1, torch.arange(w).flip(0) + 1).float() / f, max=1.0)
    a = (ry[:, None] * rx[None, :])[None]
    region = bg[:, y0:y0 + h, x0:x0 + w].float()
    bg[:, y0:y0 + h, x0:x0 + w] = (a * crop.float() + (1 - a) * region).round().clamp(0, 255).byte()
    return bg


class LiveBoxSet(Dataset):
    """The boxed stream of the D-044 recipe. Each sample is one of:
      a plant-free frame (p_neg; manifests/negatives.parquet): no boxes, so with the SSD loss's negative floor
        the detector learns "no plant here", and a uniform whole-frame target (the presence gate's signal);
      a seam-free composite (p_comp): 1-3 OWLv2-boxed plants cut out with a 10% margin, scaled to 15-50% of the
        frame, feathered and pasted onto a real vegetation scene; boxes where they landed, and an area-weighted
        whole-frame target over the pasted genera (mosaics had none);
      a boxed photo as before, except that vascular full-frame fallback boxes (often wrong; audit
        detector_training-02) contribute no box loss when fallback_off holds the photo's path.
    Returns (img, boxes, labels, soft) with soft = (genus idx[K_SOFT], weight[K_SOFT], uniform, box_loss)."""

    def __init__(self, root, items, size, low_light_p, photo, wb_p, p_comp, p_neg, negs, vegs, fallback_off, comp_items):
        self.root, self.items, self.size = root, items, size
        self.full = BoxSet(root, items, size, True, low_light_p, photo, zoom_out=p_comp == 0, wb_p=wb_p)
        self.p_comp, self.p_neg, self.negs, self.vegs = p_comp, p_neg, negs, vegs
        self.fallback_off, self.comp_items = fallback_off, comp_items
        self.low_light_p, self.wb_p = low_light_p, wb_p
        self.bg_tf = v2.Compose([v2.RandomResizedCrop(size, scale=(0.35, 1.0), antialias=True), v2.RandomHorizontalFlip(),
                                 photometric(photo)])

    def __len__(self):
        return len(self.items)

    def _soft(self, genera, weights, uniform=False, box_loss=True):
        idx = torch.full((K_SOFT,), 0, dtype=torch.long)
        w = torch.zeros(K_SOFT)
        for k, (g, v) in enumerate(list(zip(genera, weights))[:K_SOFT]):
            idx[k], w[k] = g, v
        if w.sum() > 0:
            w = w / w.sum()
        return idx, w, torch.tensor(uniform), torch.tensor(box_loss)

    def _finish(self, img):
        if self.wb_p and float(torch.rand(())) < self.wb_p:
            img = white_balance(img)
        return field_degrade(img, self.size, self.low_light_p)

    def _bg(self, paths):
        p = paths[int(torch.randint(0, len(paths), ()))]
        return self.bg_tf(decode_image(str(self.root / p), mode=ImageReadMode.RGB))

    def composite(self):
        S = self.size
        img = self._bg(self.vegs).clone()
        boxes, labels, areas = [], [], []
        for _ in range(int(torch.randint(1, 4, ()))):
            path, frac, lab = self.comp_items[int(torch.randint(0, len(self.comp_items), ()))]
            src = decode_image(str(self.root / path), mode=ImageReadMode.RGB)
            h, w = src.shape[-2:]
            x0, y0, x1, y1 = (np.asarray(frac[0], np.float32) * np.array([w, h, w, h], np.float32)).tolist()
            mx, my = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            cx0, cy0, cx1, cy1 = max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx) + 1), min(h, int(y1 + my) + 1)
            crop = src[:, cy0:cy1, cx0:cx1]
            if crop.shape[1] < 16 or crop.shape[2] < 16:
                continue
            scale = float(torch.empty(()).uniform_(0.15, 0.5)) * S / max(crop.shape[1:])
            nh, nw = max(8, int(crop.shape[1] * scale)), max(8, int(crop.shape[2] * scale))
            crop = v2.functional.resize(crop, [nh, nw], antialias=True)
            if float(torch.rand(())) < 0.5:
                crop = crop.flip(-1)
            for _try in range(8):                                        # avoid covering earlier pastes
                px, py = int(torch.randint(0, S - nw + 1, ())), int(torch.randint(0, S - nh + 1, ()))
                b = torch.tensor([px + (x0 - cx0) * scale, py + (y0 - cy0) * scale, px + (x1 - cx0) * scale, py + (y1 - cy0) * scale])
                if not boxes or float(box_iou_1(b, torch.stack(boxes)).max()) < 0.2:
                    break
            paste(img, crop, px, py, int(torch.randint(4, 17, ())))
            boxes.append(b.clamp(0, S)); labels.append(int(lab[0])); areas.append(float((b[2] - b[0]) * (b[3] - b[1])))
        img = self._finish(img)
        if not boxes:
            return img, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), self._soft([], [], uniform=False, box_loss=False)
        return img, torch.stack(boxes).float(), torch.tensor(labels), self._soft([l - 1 for l in labels], areas)

    def negative(self):
        img = self._finish(self._bg(self.negs))
        return img, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), self._soft([], [], uniform=True)

    def __getitem__(self, i):
        r = float(torch.rand(()))
        if r < self.p_neg:
            return self.negative()
        if r < self.p_neg + self.p_comp:
            return self.composite()
        img, b, lab, g = self.full[i]
        off = self.items[i][0] in self.fallback_off
        if off:
            b, lab = torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
        return img, b, lab, self._soft([g - 1], [1.0], box_loss=not off)


def box_iou_1(b: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    ix = (torch.minimum(b[2], others[:, 2]) - torch.maximum(b[0], others[:, 0])).clamp(min=0)
    iy = (torch.minimum(b[3], others[:, 3]) - torch.maximum(b[1], others[:, 1])).clamp(min=0)
    inter = ix * iy
    area = (b[2] - b[0]) * (b[3] - b[1]) + (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]) - inter
    return inter / area.clamp(min=1e-6)


def collate_live(batch):
    imgs, boxes, labels, soft = zip(*batch)
    idx, w, uni, bl = zip(*soft)
    return torch.stack(imgs), list(boxes), list(labels), (torch.stack(idx), torch.stack(w), torch.stack(uni), torch.stack(bl))


def mosaic_eval(model, net, items, size, anchors0, dev, mean, std, workers, n=2000) -> dict:
    """Class-agnostic box AP on deterministic 2x2 mosaics of the first n val photos: several plants per
    frame, each about a quarter of it, the live walking view the close-up photos lack."""
    model.eval()
    dets, gts = {k: [] for k in ("img", "score", "box")}, {k: [] for k in ("img", "box")}
    k0 = 0
    dl = DataLoader(MosaicBoxSet(data_root(), items[:n], size, False, 1.0), batch_size=64, num_workers=workers, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, boxes, _, _ in dl:
            out = forward_raw(net, ((imgs.to(dev).float() / 255) - mean) / std)
            res, _ = detect(model.box_coder, {k: v.float() for k, v in out.items()}, anchors0, size)
            for i, r in enumerate(res):
                dets["img"].append(np.full(len(r["ascore"]), k0 + i)); dets["score"].append(r["ascore"]); dets["box"].append(r["abox"])
                gts["img"].append(np.full(len(boxes[i]), k0 + i)); gts["box"].append(boxes[i].numpy())
            k0 += len(res)
    model.train()
    d = {k: np.concatenate(v) for k, v in dets.items()}
    g = {k: np.concatenate(v) for k, v in gts.items()}
    m = coco_map(d["img"], np.zeros(len(d["img"]), int), d["score"], d["box"], g["img"], np.zeros(len(g["img"]), int), g["box"])
    return {"mosaic_agnostic_ap50": m["ap50"], "mosaic_agnostic_map": m["map"]}


class SeededComposites(Dataset):
    """Deterministic composites (index-seeded) for the composite metric: the walking view of several
    plants on real vegetation without the hard seams the 2x2 mosaic metric rewarded."""

    def __init__(self, base: "LiveBoxSet", n: int):
        self.base, self.n = base, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        torch.manual_seed(100_003 + i)
        img, b, lab, _ = self.base.composite()
        return img, b, lab, 0


class SeededNegatives(Dataset):
    def __init__(self, root, paths, size):
        self.root, self.paths, self.size = root, paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = v2.Resize((self.size, self.size), antialias=True)(decode_image(str(self.root / self.paths[i]), mode=ImageReadMode.RGB))
        return img, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), 0


def composite_eval(model, net, base, size, anchors0, dev, mean, std, workers, n=1500) -> dict:
    """Class-agnostic box AP50 on n seeded composites (see LiveBoxSet.composite)."""
    model.eval()
    dets, gts = {k: [] for k in ("img", "score", "box")}, {k: [] for k in ("img", "box")}
    k0 = 0
    dl = DataLoader(SeededComposites(base, n), batch_size=64, num_workers=workers, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, boxes, _, _ in dl:
            out = forward_raw(net, ((imgs.to(dev).float() / 255) - mean) / std)
            res, _ = detect(model.box_coder, {k: v.float() for k, v in out.items()}, anchors0, size)
            for i, r in enumerate(res):
                dets["img"].append(np.full(len(r["ascore"]), k0 + i)); dets["score"].append(r["ascore"]); dets["box"].append(r["abox"])
                gts["img"].append(np.full(len(boxes[i]), k0 + i)); gts["box"].append(boxes[i].numpy())
            k0 += len(res)
    model.train()
    d = {k: np.concatenate(v) for k, v in dets.items()}
    g = {k: np.concatenate(v) for k, v in gts.items()}
    m = coco_map(d["img"], np.zeros(len(d["img"]), int), d["score"], d["box"], g["img"], np.zeros(len(g["img"]), int), g["box"])
    return {"composite_ap50": m["ap50"], "composite_map": m["map"]}


def negative_eval(model, net, root, paths, size, dev, mean, std, workers) -> dict:
    """Plant-free val frames: share with any box at the app's default plant score 0.25, and the median
    top-5 whole-frame mass (the presence gate's signal; lower is better here)."""
    model.eval()
    boxed, masses = [], []
    dl = DataLoader(SeededNegatives(root, paths, size), batch_size=64, num_workers=workers, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, _, _, _ in dl:
            out = forward_raw(net, ((imgs.to(dev).float() / 255) - mean) / std)
            obj = 1.0 - out["cls_logits"].float().softmax(-1)[..., 0]
            boxed.append((obj.amax(1) >= 0.25).cpu().numpy())
            masses.append(out["image_logits"].float().softmax(-1).topk(5, 1).values.sum(1).cpu().numpy())
    model.train()
    return {"neg_box_rate": float(np.concatenate(boxed).mean()), "neg_mass_p50": float(np.median(np.concatenate(masses)))}


def trait_items(root, gi, split="val", cap=20, seed=0) -> list:
    """Non-specimen photos (all sources) of genera with cited trait rows, at most `cap` per genus: the guard
    metric for encounter weighting (eval/trait_recall.py; garden edibles and hemlocks are rare in the pool)."""
    repo = Path(__file__).resolve().parents[3]
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "scientific_name", "genus"])
    tg = set(sp[sp.scientific_name.isin(set(pd.read_csv(repo / "data/traits/species_traits.csv").scientific_name))].genus)
    m = pd.read_parquet(root / f"artifacts/img_emb/{split}.parquet", columns=["uuid", "class_idx", "is_specimen"])
    m = m[~m.is_specimen.fillna(False).astype(bool)].merge(sp[["class_idx", "genus"]], on="class_idx")
    m = m[m.genus.isin(tg) & m.genus.isin(gi)].merge(pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")
    m = m.sample(frac=1, random_state=seed).groupby("genus").head(cap)
    box = np.zeros((1, 4), np.float32)
    return [(p, box, np.array([gi[g] + 1])) for p, g in zip(m.path, m.genus)]


def trait_eval(model, net, items, size, n_genera, dev, mean, std, workers) -> dict:
    model.eval()
    hit = np.zeros(n_genera); cnt = np.zeros(n_genera)
    dl = DataLoader(BoxSet(data_root(), items, size, False), batch_size=128, num_workers=workers, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, _, _, genus in dl:
            p = forward_raw(net, ((imgs.to(dev).float() / 255) - mean) / std)["image_logits"].argmax(1).cpu().numpy()
            y = genus.numpy() - 1
            np.add.at(cnt, y, 1); np.add.at(hit, y, p == y)
    model.train()
    m = cnt > 0
    return {"trait_genus_recall": float((hit[m] / cnt[m]).mean())}


def encounter_share(root, genera) -> np.ndarray:
    """Each detector genus' share of the 2.27M regional GBIF observations (priors_state_month, all states/months)."""
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["gbif_taxon_key", "genus"])
    pr = pd.read_parquet(root / "manifests/priors_state_month.parquet").merge(sp, on="gbif_taxon_key")
    c = pr.groupby("genus")["count"].sum().reindex(genera).fillna(0).to_numpy(float) + 1.0
    return c / c.sum()


def regional_pool(root, split, gi) -> list:
    """BioCLIP's regional photos of a split (non-specimen, genus known to the detector) as BoxSet items with a
    dummy box: ~20K val photos, far more per common genus than the detector's capped boxed split."""
    m = pd.read_parquet(root / f"artifacts/img_emb/{split}.parquet", columns=["uuid", "class_idx", "is_specimen", "source"])
    m = m[(m.source == "gbif_regional") & ~m.is_specimen.fillna(False).astype(bool)]
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "genus"])
    paths = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    m = m.merge(sp, on="class_idx").merge(paths, on="uuid")
    m = m[m.genus.isin(gi)]
    box = np.zeros((1, 4), np.float32)
    return [(p, box, np.array([gi[g] + 1])) for p, g in zip(m.path, m.genus)]


def encounter_eval(model, net, items, size, enc, dev, mean, std, workers) -> dict:
    """Whole-frame genus recall per genus on regional photos, averaged with each genus weighted by how often it
    is met (D-040): the live accuracy to expect on a walk, not on a per-species-capped photo set."""
    model.eval()
    hit = np.zeros(len(enc)); cnt = np.zeros(len(enc))
    dl = DataLoader(BoxSet(data_root(), items, size, False), batch_size=128, num_workers=workers, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, _, _, genus in dl:
            p = forward_raw(net, ((imgs.to(dev).float() / 255) - mean) / std)["image_logits"].argmax(1).cpu().numpy()
            y = genus.numpy() - 1
            np.add.at(cnt, y, 1); np.add.at(hit, y, p == y)
    model.train()
    m = cnt > 0
    rec = hit[m] / cnt[m]
    top50 = np.argsort(-enc)[:50]
    m50 = np.zeros(len(enc), bool); m50[top50] = True
    return {"encounter_top1": float((rec * enc[m]).sum() / enc[m].sum()), "pool_top1": float(hit.sum() / cnt.sum()),
            "top50_recall": float((hit[m50 & m].sum()) / max(cnt[m50 & m].sum(), 1)), "pool_n": int(cnt.sum())}


def eval_only(root, args, init, genera, family_of, splits, size, gi, dev) -> None:
    """Val and test metrics of a trained live checkpoint at --size (resolution sweeps without retraining)."""
    model = build(len(genera), load("detector")["model"]["backbone"], size, pretrained=False, head="shared", image_aux=True,
                  image_head=args.image_head)
    missing, _ = model.load_state_dict(init["model"], strict=False)
    assert not missing, f"checkpoint lacks {sorted(missing)[:4]}"
    model.to(dev)
    use_shift_add_depthwise(model)
    python_headers_for_triton()
    net = torch.compile(Net(model), dynamic=False)
    mean = torch.tensor(model.backbone.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(model.backbone.std, device=dev).view(1, 3, 1, 1)
    anchors0 = anchors_for(model, size, dev)
    enc = encounter_share(root, genera)
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["path", "source"]).set_index("path").source
    w = args.workers_img
    # the D-044 yardsticks (seeded composites, plant-free frames, toxic genera), built exactly as main() builds them
    bx = pd.read_parquet(root / args.boxes, columns=["uuid", "fallback"]).merge(
        pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")
    fb_paths = set(bx[bx.fallback].path)
    negm = pd.read_parquet(root / "manifests/negatives.parquet")
    for split in ("val", "test"):
        m = image_eval(model, net, splits[split], size, family_of, dev, mean, std, w)
        r = image_eval(model, net, [x for x in splits[split] if im.get(x[0]) == "gbif_regional"], size, family_of, dev, mean, std, w)
        negs = negm[(negm.kind == "plant_free") & (negm.split == split)].sample(1500, random_state=0).path.to_numpy()
        items = [it for it in splits[split] if it[0] not in fb_paths]
        comp = LiveBoxSet(root, items, size, 0.0, "jitter", 0.0, 1.0, 0.0, negs,
                          negm[(negm.kind == "vegetation") & (negm.split == split)].path.to_numpy(), set(), items)
        row = {**{k: v for k, v in m.items() if k != "n"}, **{f"regional_{k}": v for k, v in r.items() if k != "n"},
               **mosaic_eval(model, net, splits[split], size, anchors0, dev, mean, std, w),
               **encounter_eval(model, net, regional_pool(root, split, gi), size, enc, dev, mean, std, w),
               **composite_eval(model, net, comp, size, anchors0, dev, mean, std, w),
               **negative_eval(model, net, root, negs, size, dev, mean, std, w),
               **trait_eval(model, net, trait_items(root, gi, split), size, len(genera), dev, mean, std, w)}
        print(f"eval-only {args.eval_only} @ {size} {split} " + " ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)


def image_eval(model, net, items, size, family_of, dev, mean, std, workers) -> dict:
    """Whole-frame genus top-1/top-5 and family top-1 on a list of (path, boxes, labels) items."""
    fam = torch.as_tensor(family_of, device=dev)
    n_fam = int(family_of.max()) + 1
    t1 = t5 = f1 = n = 0
    model.eval()
    dl = DataLoader(BoxSet(data_root(), items, size, False), batch_size=128, num_workers=workers, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, _, _, genus in dl:
            x = ((imgs.to(dev).float() / 255) - mean) / std
            p = forward_raw(net, x)["image_logits"].float().softmax(-1)
            y = genus.to(dev) - 1
            t1 += int((p.argmax(1) == y).sum()); t5 += int((p.topk(5, 1).indices == y[:, None]).any(1).sum())
            f1 += int((torch.zeros(len(y), n_fam, device=dev).index_add_(1, fam, p).argmax(1) == fam[y]).sum())
            n += len(y)
    model.train()
    return {"image_genus_top1": t1 / n, "image_genus_top5": t5 / n, "image_family_top1": f1 / n, "n": n}


def main() -> None:
    cfg = load("detector")
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_live")
    ap.add_argument("--init", default="artifacts/detector/mnv4_shared/best.pt")
    ap.add_argument("--boxes", default="manifests/boxes.parquet")
    ap.add_argument("--image-head", type=int, default=1280)
    ap.add_argument("--size", type=int, default=0, help="input size (default: configs/detector.yaml model.input, 320)")
    ap.add_argument("--epochs", type=int, default=3, help="passes over the unboxed photos")
    ap.add_argument("--batch-box", type=int, default=32)
    ap.add_argument("--batch-img", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3, help="new whole-frame head; old heads x0.3, backbone x --backbone-lr-mult")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--kd", type=float, default=0.5, help="weight of the BioCLIP soft target in the image loss")
    ap.add_argument("--img-weight", type=float, default=1.0)
    ap.add_argument("--low-light", type=float, default=0.2)
    ap.add_argument("--mosaic", type=float, default=0.5, help="share of boxed samples that are 2x2 mosaics")
    ap.add_argument("--sampler-power", type=float, default=0.5)
    ap.add_argument("--encounter-mix", type=float, default=0.0,
                    help="unboxed sampler: share of each genus' training attention set by how often it is met in the region "
                         "(its share of regional observations ^ --encounter-power); the rest stays n^sampler-power (D-040)")
    ap.add_argument("--encounter-power", type=float, default=1.0)
    ap.add_argument("--select", choices=["val", "encounter", "pool"], default="val",
                    help="checkpoint selection: val whole-frame genus top-1, or encounter-weighted regional val top-1")
    ap.add_argument("--box-weight", type=float, default=0.25, help="selection: + this x val mosaic (or composite) agnostic AP50")
    ap.add_argument("--preset", choices=["", "d044"], default="",
                    help="d044: composites 0.4, plant-free frames 0.15, negative floor 32, no negatives inside boxes, "
                         "vascular fallback boxes out of the box loss, colour jitter without channel shuffles + white "
                         "balance 0.3, no mosaics, encounter weighting with trait genera exempt, select on pool top-1")
    ap.add_argument("--composite", type=float, default=0.0, help="share of boxed samples that are seam-free composites")
    ap.add_argument("--negatives", type=float, default=0.0, help="share of boxed samples that are plant-free frames")
    ap.add_argument("--neg-floor", type=int, default=0, help="SSD loss: at least this many hard negatives per image")
    ap.add_argument("--inside-ignore", type=float, default=0.0, help="SSD loss: no negatives this much inside a GT box")
    ap.add_argument("--fallback-box-loss", choices=["keep", "vascular_off"], default="keep")
    ap.add_argument("--photo", choices=["ssd", "jitter"], default="ssd")
    ap.add_argument("--wb", type=float, default=0.0, help="white-balance augmentation probability")
    ap.add_argument("--neg-image-weight", type=float, default=0.5, help="weight of the uniform whole-frame target on plant-free frames")
    ap.add_argument("--neg-weight", type=float, default=0.25, help="selection: - this x plant-free frames with a box")
    ap.add_argument("--encounter-exempt-traits", action="store_true", help="genera with cited trait rows keep at least their balanced share")
    ap.add_argument("--eval-only", default="", help="checkpoint: val + test metrics at --size, no training, no table row")
    ap.add_argument("--emb", default="artifacts/img_emb", help="BioCLIP embedding set for the unboxed stream and teacher (D-045: img_emb_v2)")
    ap.add_argument("--probe", default="artifacts/probe.pt", help="teacher probe")
    ap.add_argument("--temperature", default="artifacts/temperature.txt")
    ap.add_argument("--eval-every", type=int, default=4000, help="steps")
    ap.add_argument("--workers-img", type=int, default=14)
    ap.add_argument("--workers-box", type=int, default=6)
    ap.add_argument("--limit-img", type=int, default=0, help="smoke tests: unboxed photos")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke tests: total steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.preset == "d044":
        for k, v in dict(composite=0.4, negatives=0.15, neg_floor=32, inside_ignore=0.5, fallback_box_loss="vascular_off",
                         photo="jitter", wb=0.3, mosaic=0.0, encounter_exempt_traits=True, select="pool").items():
            setattr(args, k, v)
    torch.manual_seed(args.seed)
    root, dev = data_root(), "cuda"
    size = args.size or cfg["model"]["input"]
    args.size = size
    if args.eval_only:
        args.init = args.eval_only
    init = torch.load(root / args.init, map_location="cpu")
    genera, family_of, families, splits = load_items(root, args.boxes, init["args"].get("fallback", "keep"), 0, args.seed)
    out_dir = root / "artifacts/detector" / args.name
    if not args.eval_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (out_dir / "labels.json").write_text(json.dumps({"genera": genera, "families": families, "family_of": family_of.tolist()}))
    gi = {g: i for i, g in enumerate(genera)}
    if args.eval_only:
        return eval_only(root, args, init, genera, family_of, splits, size, gi, dev)

    # unboxed stream: every non-specimen training photo whose genus the detector knows (D-020: no specimens)
    t = teacher(root, out_dir / "teacher_train.npz", genera, args)
    meta = pd.read_parquet(root / args.emb / "train.parquet", columns=["uuid", "class_idx", "is_specimen"])
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "genus"])
    paths = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    u = meta[~meta.is_specimen.fillna(False).astype(bool)].merge(sp, on="class_idx").merge(paths, on="uuid").merge(t, on="uuid")
    u = u[u.genus.isin(gi)].reset_index(drop=True)
    if args.limit_img:
        u = u.sample(args.limit_img, random_state=args.seed).reset_index(drop=True)
    u["g"] = u.genus.map(gi)
    n_g = u.g.value_counts()
    w = u.g.map(n_g).to_numpy(float) ** -args.sampler_power
    enc = encounter_share(root, genera)                                   # per detector genus, sums to 1
    if args.encounter_mix:
        bal = (n_g.reindex(range(len(genera))).fillna(0).to_numpy() ** (1 - args.sampler_power))
        bal = bal / bal.sum()
        e = enc ** args.encounter_power; e = e / e.sum()
        target = (1 - args.encounter_mix) * bal + args.encounter_mix * e   # each genus' share of the samples
        if args.encounter_exempt_traits:                                   # audit edible_targeting_app-10
            repo = Path(__file__).resolve().parents[3]
            spx = pd.read_parquet(root / "manifests/species.parquet", columns=["scientific_name", "genus"])
            tg = set(spx[spx.scientific_name.isin(set(pd.read_csv(repo / "data/traits/species_traits.csv").scientific_name))].genus)
            ex = np.array([g in tg for g in genera])
            target = np.where(ex, np.maximum(target, bal), target)
            target = target / target.sum()
            print(f"encounter weighting: {ex.sum()} trait genera keep at least their balanced share ({target[ex].sum():.3f} of samples)", flush=True)
        w = (target / np.maximum(n_g.reindex(range(len(genera))).fillna(1).to_numpy(), 1))[u.g.to_numpy()]
        top50 = np.argsort(-enc)[:50]
        print(f"encounter mix {args.encounter_mix}: top-50 most-met genera get {target[top50].sum():.3f} of samples "
              f"(balanced alone {bal[top50].sum():.3f}; their sighting share {enc[top50].sum():.2f})", flush=True)
    ds_img = ImgSet(root, u.path.to_numpy(), u.g.to_numpy(), np.stack(u.tidx.to_numpy()), np.stack(u.tp.to_numpy()), size, args.low_light,
                    args.photo, args.wb)
    dl_img = DataLoader(ds_img, batch_size=args.batch_img, num_workers=args.workers_img, pin_memory=True, drop_last=True,
                        persistent_workers=True, prefetch_factor=4,
                        sampler=WeightedRandomSampler(torch.as_tensor(w), len(u), replacement=True,
                                                      generator=torch.Generator().manual_seed(args.seed)))
    train = splits["train"]
    nb = np.bincount([lab[0] for _, _, lab in train], minlength=len(genera) + 1)
    wb = np.array([nb[lab[0]] for _, _, lab in train], float) ** -args.sampler_power
    live = bool(args.composite or args.negatives or args.fallback_box_loss != "keep" or args.photo != "ssd")
    if live:
        bx = pd.read_parquet(root / args.boxes, columns=["uuid", "genus", "fallback"]).merge(
            pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")
        fb_paths = set(bx[bx.fallback].path)
        off = set()
        if args.fallback_box_loss == "vascular_off":
            spx = pd.read_parquet(root / "manifests/species.parquet", columns=["genus", "phylum", "family"]).drop_duplicates("genus")
            nonvasc = set(spx[spx.phylum.isin(["Bryophyta", "Marchantiophyta", "Anthocerotophyta"]) |
                              spx.family.isin(["Poaceae", "Cyperaceae", "Juncaceae"])].genus)   # mosses and mats keep the frame box
            off = set(bx[bx.fallback & ~bx.genus.isin(nonvasc)].path)
        negm = pd.read_parquet(root / "manifests/negatives.parquet")
        negs_tr = negm[(negm.kind == "plant_free") & (negm.split == "train")].path.to_numpy()
        vegs_tr = negm[(negm.kind == "vegetation") & (negm.split == "train")].path.to_numpy()
        comp_items = [it for it in train if it[0] not in fb_paths]
        print(f"D-044 boxed stream: composites {args.composite}, plant-free {args.negatives} ({len(negs_tr):,} frames, "
              f"{len(vegs_tr):,} vegetation backgrounds), fallback box loss off for {sum(it[0] in off for it in train):,} of "
              f"{len(train):,} boxed photos, photometric {args.photo}, white balance {args.wb}", flush=True)
        box_ds = LiveBoxSet(root, train, size, args.low_light, args.photo, args.wb, args.composite, args.negatives,
                            negs_tr, vegs_tr, off, comp_items)
        box_collate = collate_live
        vegs_val = negm[(negm.kind == "vegetation") & (negm.split == "val")].path.to_numpy()
        negs_val = negm[(negm.kind == "plant_free") & (negm.split == "val")].sample(1500, random_state=0).path.to_numpy()
        val_comp = LiveBoxSet(root, [it for it in splits["val"] if it[0] not in fb_paths], size, 0.0, "jitter", 0.0, 1.0, 0.0,
                              negs_val, vegs_val, set(), [it for it in splits["val"] if it[0] not in fb_paths])
    else:
        box_ds, box_collate = MosaicBoxSet(root, train, size, True, args.mosaic, args.low_light), collate
    dl_box = DataLoader(box_ds, batch_size=args.batch_box,
                        num_workers=args.workers_box, collate_fn=box_collate, pin_memory=True, drop_last=True,
                        persistent_workers=True, prefetch_factor=4,
                        sampler=WeightedRandomSampler(torch.as_tensor(wb), len(train), replacement=True,
                                                      generator=torch.Generator().manual_seed(args.seed + 1)))
    print(f"unboxed {len(u):,} photos ({u.genus.nunique()} genera), boxed {len(train):,}; {len(genera)} genera", flush=True)

    model = build(len(genera), cfg["model"]["backbone"], size, pretrained=False, head="shared", image_aux=True,
                  image_head=args.image_head)
    missing, unexpected = model.load_state_dict(init["model"], strict=False)
    print(f"init from {args.init}: missing {sorted({k.split('.')[1] + '.' + k.split('.')[2] for k in missing})}, "
          f"unexpected {sorted({k.split('.')[1] + '.' + k.split('.')[2] for k in unexpected})}", flush=True)
    fresh_head = any(k.startswith("head.image_cls") for k in missing)      # else: continuing a trained head
    if fresh_head and args.image_head == 1280:   # MobileNetV4's own ImageNet head: 960 -> 1280 conv + BN
        tm = timm.create_model(cfg["model"]["backbone"], pretrained=True)
        model.head.image_cls[1].weight.data.copy_(tm.conv_head.weight.data)
        bn = model.head.image_cls[2]
        for k in ("weight", "bias", "running_mean", "running_var"):
            getattr(bn, k).data.copy_(getattr(tm.norm_head, k).data)
        del tm
    if fresh_head:
        torch.nn.init.normal_(model.head.image_cls[-1].weight, std=0.01)
        torch.nn.init.zeros_(model.head.image_cls[-1].bias)
    model.to(dev)
    use_shift_add_depthwise(model)
    python_headers_for_triton()
    net = torch.compile(Net(model), dynamic=False)
    mean = torch.tensor(model.backbone.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(model.backbone.std, device=dev).view(1, 3, 1, 1)
    groups = {}
    for n, p in model.named_parameters():
        kind = "bb" if n.startswith("backbone.body.") else "new" if n.startswith("head.image_cls") else "old"
        groups.setdefault((kind, p.ndim > 1), []).append(p)
    mult = {"bb": args.backbone_lr_mult, "new": 1.0, "old": 0.3}
    opt = torch.optim.AdamW([{"params": ps, "lr": args.lr * mult[k], "weight_decay": args.wd if dec else 0.0}
                             for (k, dec), ps in groups.items()], lr=args.lr)
    steps_per_epoch = len(dl_img)
    total = args.max_steps or args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler()
    amp = torch.autocast("cuda", dtype=torch.float16)
    anchors0 = anchors_for(model, size, dev)
    im = pd.read_parquet(root / "manifests/images.parquet", columns=["path", "source"]).set_index("path").source
    regional = lambda items: [x for x in items if im.get(x[0]) == "gbif_regional"]
    val_reg = regional(splits["val"])
    pool_val = regional_pool(root, "val", gi)
    print(f"{total:,} steps ({steps_per_epoch:,}/epoch), batch {args.batch_box}+{args.batch_img}; regional val {len(val_reg):,}", flush=True)

    log = RunLog(f"detector_{args.name}")
    model.train()
    best, best_step, step, t0, t_start = -1.0, -1, 0, time.time(), time.time()
    it_box = iter(dl_box)

    last_row = [None]

    trait_val = trait_items(root, gi) if live else []

    def evaluate_now():
        nonlocal best, best_step
        m = image_eval(model, net, splits["val"], size, family_of, dev, mean, std, args.workers_img)
        r = image_eval(model, net, val_reg, size, family_of, dev, mean, std, args.workers_img)
        mo = mosaic_eval(model, net, splits["val"], size, anchors0, dev, mean, std, args.workers_img)
        en = encounter_eval(model, net, pool_val, size, enc, dev, mean, std, args.workers_img)
        extra = {}
        if live:
            extra.update(composite_eval(model, net, val_comp, size, anchors0, dev, mean, std, args.workers_img))
            extra.update(negative_eval(model, net, root, negs_val, size, dev, mean, std, args.workers_img))
            extra.update(trait_eval(model, net, trait_val, size, len(genera), dev, mean, std, args.workers_img))
        row = {**{k: v for k, v in m.items() if k != "n"}, **{f"regional_{k}": v for k, v in r.items() if k != "n"}, **mo, **en, **extra}
        log.log(step, "val", **row)
        print(f"step {step} val " + " ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
        base = {"encounter": en["encounter_top1"], "pool": en["pool_top1"], "val": m["image_genus_top1"]}[args.select]
        boxes_term = extra["composite_ap50"] if live else mo["mosaic_agnostic_ap50"]
        key = base + args.box_weight * boxes_term - (args.neg_weight * extra["neg_box_rate"] if live else 0.0)
        if key > best:
            best, best_step = key, step
            torch.save(ckpt(row), out_dir / "best.pt")
        last_row[0] = row
        return row

    def ckpt(row):
        return {"model": model.state_dict(), "epoch": step, "args": {**init["args"], **vars(args), "head": "shared",
                "aux_image": 1.0}, "val": row}

    evaluate_now()                                # a new head: a floor; a continued one: the baseline
    done = False
    for ep in range(args.epochs):
        for ximg, y, tidx, tp in dl_img:
            try:
                bimgs, boxes, labels, genus = next(it_box)
            except StopIteration:
                it_box = iter(dl_box)
                bimgs, boxes, labels, genus = next(it_box)
            nbx = len(bimgs)
            x = ((torch.cat([bimgs, ximg]).to(dev, non_blocking=True).float() / 255) - mean) / std
            box_on = genus[3].tolist() if live else [True] * nbx
            targets = [{"boxes": b.to(dev, non_blocking=True), "labels": l.to(dev, non_blocking=True), "box_loss": bool(o)}
                       for b, l, o in zip(boxes, labels, box_on)]
            with amp:
                out = forward_raw(net, x)
            l_box, l_cls = ssd_loss_batched(model, {k: out[k][:nbx] for k in ("cls_logits", "bbox_regression")}, anchors0, targets,
                                            neg_floor=args.neg_floor, inside_ignore=args.inside_ignore)
            il = out["image_logits"].float()
            if live:
                # boxed stream: soft whole-frame targets (photo: its genus; composite: pasted genera by area;
                # plant-free frame: uniform, weighted --neg-image-weight); unboxed stream: its label as before
                sidx, sw, uni, _ = (t.to(dev) for t in genus)
                lpb = F.log_softmax(il[:nbx], -1)
                per = -(sw * lpb.gather(1, sidx)).sum(1)
                per = torch.where(uni, -args.neg_image_weight * lpb.mean(1), per)
                has = uni | (sw.sum(1) > 0)
                l_un = F.cross_entropy(il[nbx:], y.to(dev), reduction="sum")
                l_hard = (per[has].sum() + l_un) / max(1, int(has.sum()) + len(y))
            else:
                hard = torch.cat([genus.to(dev) - 1, y.to(dev)])          # mosaics: genus 0 -> -1, ignored
                l_hard = F.cross_entropy(il, hard, ignore_index=-1)
            logp = F.log_softmax(il[nbx:], -1)
            l_kd = -(tp.to(dev) * logp.gather(1, tidx.to(dev))).sum(1).mean()
            l_img = (1 - args.kd) * l_hard + args.kd * l_kd
            loss = l_box + l_cls + args.img_weight * l_img
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{total} box {l_box.item():.3f} cls {l_cls.item():.3f} hard {l_hard.item():.3f} "
                      f"kd {l_kd.item():.3f} lr {max(sched.get_last_lr()):.2e} "
                      f"{step * (args.batch_box + args.batch_img) / (time.time() - t0):.0f} img/s", flush=True)
                log.log(step, "train", loss_box=l_box.item(), loss_cls=l_cls.item(), loss_hard=l_hard.item(), loss_kd=l_kd.item())
            if step % args.eval_every == 0:
                torch.save(ckpt(evaluate_now()), out_dir / "last.pt")
            if step >= total:
                done = True
                break
        if done:
            break
    row = evaluate_now() if step % args.eval_every else last_row[0]
    torch.save(ckpt(row), out_dir / "final.pt")          # the last step, whether or not it was selected
    if args.max_steps:
        print(f"smoke run: best selection metric ({args.select}) {best:.4f} at step {best_step}; no test", flush=True)
        log.close()
        return
    ck = torch.load(out_dir / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    test = image_eval(model, net, splits["test"], size, family_of, dev, mean, std, args.workers_img)
    test.update(mosaic_eval(model, net, splits["test"], size, anchors0, dev, mean, std, args.workers_img))
    test.update(encounter_eval(model, net, regional_pool(root, "test", gi), size, enc, dev, mean, std, args.workers_img))
    if live:
        negs_test = negm[(negm.kind == "plant_free") & (negm.split == "test")].sample(1500, random_state=0).path.to_numpy()
        test_items = [it for it in splits["test"] if it[0] not in fb_paths]
        test_comp = LiveBoxSet(root, test_items, size, 0.0, "jitter", 0.0, 1.0, 0.0, negs_test,
                               negm[(negm.kind == "vegetation") & (negm.split == "test")].path.to_numpy(), set(), test_items)
        test.update(composite_eval(model, net, test_comp, size, anchors0, dev, mean, std, args.workers_img))
        test.update(negative_eval(model, net, root, negs_test, size, dev, mean, std, args.workers_img))
        test.update(trait_eval(model, net, trait_items(root, gi, "test"), size, len(genera), dev, mean, std, args.workers_img))
    det = evaluate(model, net, DataLoader(BoxSet(root, splits["test"], size, False), batch_size=64, num_workers=args.workers_img,
                                          collate_fn=collate), anchors0, size, family_of, dev, mean, std, False, amp)
    print(f"best step {best_step}: test {test} | detection {det}", flush=True)
    log.log(best_step, "test", **{k: v for k, v in {**det, **test}.items() if k != "n"})
    log.summary("detector_live", name=args.name, init=args.init, image_head=args.image_head, kd=args.kd, low_light=args.low_light,
                epochs=args.epochs, steps=step, best_step=best_step, n_unboxed=len(u), hours=round((time.time() - t_start) / 3600, 2),
                **{f"val_{k}": round(v, 4) for k, v in ck["val"].items()},
                **{f"test_{k}": round(v, 4) for k, v in test.items() if k != "n"},
                mosaic=args.mosaic, encounter_mix=args.encounter_mix, encounter_power=args.encounter_power, select=args.select, test_agnostic_ap50=round(det["agnostic_ap50"], 4), test_ap50=round(det["ap50"], 4),
                preset=args.preset, size=size)
    log.close()


if __name__ == "__main__":
    main()

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
of val; the regional slice is reported), test once at the end. Writes artifacts/detector/<name>/ and a row
to results/tables/detector_live.csv.
  python -m transferner.train.train_live_head --name mnv4_live"""
from __future__ import annotations

import argparse
import json
import math
import time

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
                             load_items, python_headers_for_triton, ssd_loss_batched, use_shift_add_depthwise)


def teacher(root, out_path, genera, args) -> pd.DataFrame:
    """BioCLIP's calibrated species distribution on every cached training embedding, summed to the
    detector's genera (mass on other genera dropped, then renormalised), top 10. Cached."""
    if out_path.exists():
        z = np.load(out_path, allow_pickle=True)
        return pd.DataFrame({"uuid": z["uuid"], "tidx": list(z["tidx"]), "tp": list(z["tp"])})
    meta = pd.read_parquet(root / "artifacts/img_emb/train.parquet")
    emb = np.load(root / "artifacts/img_emb/train.npy", mmap_mode="r")
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx")
    gi = {g: i for i, g in enumerate(genera)}
    G = len(genera)
    s2g = torch.tensor([gi.get(g, G) for g in sp.genus], device="cuda")
    pr = torch.load(root / "artifacts/probe.pt", map_location="cuda")
    W, b, scale = pr["W"].float(), pr["b"].float(), float(pr["scale"])
    T = float((root / "artifacts/temperature.txt").read_text())
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

    def __init__(self, root, paths, labels, tidx, tp, size, low_light_p):
        self.root, self.paths, self.labels, self.tidx, self.tp = root, paths, labels, tidx, tp
        self.size, self.low_light_p = size, low_light_p
        self.tf = v2.Compose([v2.RandomResizedCrop(size, scale=(0.2, 1.0), antialias=True), v2.RandomHorizontalFlip(),
                              v2.RandomPhotometricDistort(p=0.5)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(decode_image(str(self.root / self.paths[i]), mode=ImageReadMode.RGB))
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
    ap.add_argument("--select", choices=["val", "encounter"], default="val",
                    help="checkpoint selection: val whole-frame genus top-1, or encounter-weighted regional val top-1")
    ap.add_argument("--eval-every", type=int, default=4000, help="steps")
    ap.add_argument("--workers-img", type=int, default=14)
    ap.add_argument("--workers-box", type=int, default=6)
    ap.add_argument("--limit-img", type=int, default=0, help="smoke tests: unboxed photos")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke tests: total steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    root, dev = data_root(), "cuda"
    size = args.size or cfg["model"]["input"]
    args.size = size
    init = torch.load(root / args.init, map_location="cpu")
    genera, family_of, families, splits = load_items(root, args.boxes, init["args"].get("fallback", "keep"), 0, args.seed)
    out_dir = root / "artifacts/detector" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps({"genera": genera, "families": families, "family_of": family_of.tolist()}))
    gi = {g: i for i, g in enumerate(genera)}

    # unboxed stream: every non-specimen training photo whose genus the detector knows (D-020: no specimens)
    t = teacher(root, out_dir / "teacher_train.npz", genera, args)
    meta = pd.read_parquet(root / "artifacts/img_emb/train.parquet", columns=["uuid", "class_idx", "is_specimen"])
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
        w = (target / np.maximum(n_g.reindex(range(len(genera))).fillna(1).to_numpy(), 1))[u.g.to_numpy()]
        top50 = np.argsort(-enc)[:50]
        print(f"encounter mix {args.encounter_mix}: top-50 most-met genera get {target[top50].sum():.3f} of samples "
              f"(balanced alone {bal[top50].sum():.3f}; their sighting share {enc[top50].sum():.2f})", flush=True)
    ds_img = ImgSet(root, u.path.to_numpy(), u.g.to_numpy(), np.stack(u.tidx.to_numpy()), np.stack(u.tp.to_numpy()), size, args.low_light)
    dl_img = DataLoader(ds_img, batch_size=args.batch_img, num_workers=args.workers_img, pin_memory=True, drop_last=True,
                        persistent_workers=True, prefetch_factor=4,
                        sampler=WeightedRandomSampler(torch.as_tensor(w), len(u), replacement=True,
                                                      generator=torch.Generator().manual_seed(args.seed)))
    train = splits["train"]
    nb = np.bincount([lab[0] for _, _, lab in train], minlength=len(genera) + 1)
    wb = np.array([nb[lab[0]] for _, _, lab in train], float) ** -args.sampler_power
    dl_box = DataLoader(MosaicBoxSet(root, train, size, True, args.mosaic, args.low_light), batch_size=args.batch_box,
                        num_workers=args.workers_box, collate_fn=collate, pin_memory=True, drop_last=True,
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

    def evaluate_now():
        nonlocal best, best_step
        m = image_eval(model, net, splits["val"], size, family_of, dev, mean, std, args.workers_img)
        r = image_eval(model, net, val_reg, size, family_of, dev, mean, std, args.workers_img)
        mo = mosaic_eval(model, net, splits["val"], size, anchors0, dev, mean, std, args.workers_img)
        en = encounter_eval(model, net, pool_val, size, enc, dev, mean, std, args.workers_img)
        row = {**{k: v for k, v in m.items() if k != "n"}, **{f"regional_{k}": v for k, v in r.items() if k != "n"}, **mo, **en}
        log.log(step, "val", **row)
        print(f"step {step} val " + " ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
        key = en["encounter_top1"] if args.select == "encounter" else m["image_genus_top1"]
        if key > best:
            best, best_step = key, step
            torch.save({"model": model.state_dict(), "epoch": step, "args": {**init["args"], **vars(args), "head": "shared",
                        "aux_image": 1.0}, "val": row}, out_dir / "best.pt")

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
            targets = [{"boxes": b.to(dev, non_blocking=True), "labels": l.to(dev, non_blocking=True)} for b, l in zip(boxes, labels)]
            with amp:
                out = forward_raw(net, x)
            l_box, l_cls = ssd_loss_batched(model, {k: out[k][:nbx] for k in ("cls_logits", "bbox_regression")}, anchors0, targets)
            il = out["image_logits"].float()
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
                evaluate_now()
                torch.save({"model": model.state_dict(), "epoch": step, "args": vars(args)}, out_dir / "last.pt")
            if step >= total:
                done = True
                break
        if done:
            break
    if step % args.eval_every:
        evaluate_now()
    if args.max_steps:
        print(f"smoke run: best selection metric ({args.select}) {best:.4f} at step {best_step}; no test", flush=True)
        log.close()
        return
    ck = torch.load(out_dir / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    test = image_eval(model, net, splits["test"], size, family_of, dev, mean, std, args.workers_img)
    test.update(mosaic_eval(model, net, splits["test"], size, anchors0, dev, mean, std, args.workers_img))
    test.update(encounter_eval(model, net, regional_pool(root, "test", gi), size, enc, dev, mean, std, args.workers_img))
    det = evaluate(model, net, DataLoader(BoxSet(root, splits["test"], size, False), batch_size=64, num_workers=args.workers_img,
                                          collate_fn=collate), anchors0, size, family_of, dev, mean, std, False, amp)
    print(f"best step {best_step}: test {test} | detection {det}", flush=True)
    log.log(best_step, "test", **{k: v for k, v in {**det, **test}.items() if k != "n"})
    log.summary("detector_live", name=args.name, init=args.init, image_head=args.image_head, kd=args.kd, low_light=args.low_light,
                epochs=args.epochs, steps=step, best_step=best_step, n_unboxed=len(u), hours=round((time.time() - t_start) / 3600, 2),
                **{f"val_{k}": round(v, 4) for k, v in ck["val"].items()},
                **{f"test_{k}": round(v, 4) for k, v in test.items() if k != "n"},
                mosaic=args.mosaic, encounter_mix=args.encounter_mix, encounter_power=args.encounter_power, select=args.select, test_agnostic_ap50=round(det["agnostic_ap50"], 4), test_ap50=round(det["ap50"], 4))
    log.close()


if __name__ == "__main__":
    main()

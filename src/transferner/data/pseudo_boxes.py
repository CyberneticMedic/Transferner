"""Step 08. Pseudo bounding boxes for the detector (Phase D; D-002, D-012, D-020).

OWLv2 (configs/detector.yaml: labels.pseudo_labeler, prompts, min_box_score) runs over a
stratified subset of non-specimen images (D-020): at most --per-genus images per genus, taken in
the fixed sha1-of-uuid order the efficiency sweep uses, from --splits. Each image keeps the split
BioCLIP assigned it (observation-grouped), so detector train/val/test never share an observation.

Per image, every prompt's score per predicted box is the sigmoid of OWLv2's class logit; a box keeps
its best prompt. Boxes at or above min_box_score go through class-agnostic NMS (--iou), at most
--max-boxes per image. Every box inherits the image's genus label (one subject per image).
Images with no box get one full-frame box (fallback=True, score NaN).

OWLv2 pads each image to a square (bottom/right) before resizing to 960, so its boxes are relative
to the padded square; they are mapped back to the original image here and clipped to [0, 1].

Writes manifests/boxes.parquet: uuid, split, class_idx, genus, family, x0, y0, x1, y1 (fractions of
the original image width/height), score, prompt, fallback. Resumable: labelled images are kept in
<parts>/part-NNNNN.parquet (--chunk images each); a restart, or a larger --per-genus, labels only the
selected images that no part holds yet.

boxes.parquet stays raw. `drop_nested()` is the training-time cleanup for the part-agnostic
detector (D-015): it drops boxes that sit >= 90% inside a larger box of the same image. On the
64-image smoke set it removes 19 of 77 boxes (13 'a leaf'); boxes/image 1.60 -> 1.21.

GPU runs go through scripts/q.sh. CPU smoke test (a few images, boxes drawn for a visual check):
  python -m transferner.data.pseudo_boxes --device cpu --limit 8 --draw 8 \\
      --out artifacts/pseudo_boxes_smoke/boxes.parquet --parts artifacts/pseudo_boxes_smoke/parts
(--draw writes <parts>/draw/*.jpg)"""
from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms

from ..utils.config import data_root, load


def select(root, splits: list[str], per_genus: int, limit: int) -> pd.DataFrame:
    """Non-specimen, kept images of `splits`, at most per_genus per genus, fixed hash order."""
    spl = pd.read_parquet(root / "manifests/splits.parquet", columns=["uuid", "class_idx", "split", "is_specimen"])
    spl = spl[spl.split.isin(splits) & ~spl.is_specimen.astype(bool)]
    img = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path", "keep"])
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "genus", "family"])
    d = spl.merge(img[img.keep], on="uuid").merge(sp, on="class_idx")
    d["_h"] = [hashlib.sha1(u.encode()).hexdigest() for u in d.uuid]
    d = d.sort_values("_h", kind="stable")
    if per_genus:
        d = d[d.groupby("genus").cumcount() < per_genus]
    d = d.drop(columns=["_h", "keep", "is_specimen"]).reset_index(drop=True)
    return d.iloc[:limit] if limit else d


class Images(Dataset):
    def __init__(self, root, paths, image_processor):
        self.root, self.paths, self.ip = root, list(paths), image_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.root / self.paths[i]).convert("RGB")
        px = self.ip(images=im, return_tensors="pt")["pixel_values"][0]
        return px, torch.tensor(im.size)                    # (w, h) of the original image


def boxes_for(logits, pred_boxes, sizes, min_score: float, iou: float, max_boxes: int):
    """logits (B, P, Q), pred_boxes (B, P, 4) cxcywh in [0, 1] of the padded square, sizes (B, 2) = (w, h).
    Yields per image: xyxy fractions of the original image (n, 4), scores (n,), prompt index (n,)."""
    probs = logits.float().sigmoid()
    sizes = sizes.to(logits.device)                         # DataLoader sizes are on the CPU
    score, label = probs.max(-1)                             # best prompt per predicted box
    cx, cy, bw, bh = pred_boxes.float().unbind(-1)
    xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1)
    for i in range(len(xyxy)):
        w, h = sizes[i].float()
        side = torch.maximum(w, h)                           # OWLv2 pads to a side x side square
        b = xyxy[i] * side / torch.stack([w, h, w, h])
        b = b.clamp(0, 1)
        keep = score[i] >= min_score
        b, s, lab = b[keep], score[i][keep], label[i][keep]
        ok = ((b[:, 2] - b[:, 0]) > 1e-3) & ((b[:, 3] - b[:, 1]) > 1e-3)
        b, s, lab = b[ok], s[ok], lab[ok]
        k = nms(b, s, iou)[:max_boxes]
        yield b[k].cpu().numpy(), s[k].cpu().numpy(), lab[k].cpu().numpy()


def drop_nested(boxes: pd.DataFrame, contain: float = 0.9) -> pd.DataFrame:
    """Training-time cleanup of boxes.parquet (the labeller's output stays raw): within each image,
    drop a box when at least `contain` of its area lies inside a larger box that is itself kept.
    This removes the single-leaflet and single-flower boxes OWLv2 puts inside a whole-plant box
    (round-3 smoke check); separate subjects side by side are untouched. Fallback rows are kept."""
    keep = np.ones(len(boxes), bool)
    xy = boxes[["x0", "y0", "x1", "y1"]].to_numpy(np.float64)
    area = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    for idx in boxes.groupby("uuid", sort=False).indices.values():
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(-area[idx], kind="stable")]       # largest first
        kept: list[int] = []
        for i in idx:
            for j in kept:
                iw = min(xy[i, 2], xy[j, 2]) - max(xy[i, 0], xy[j, 0])
                ih = min(xy[i, 3], xy[j, 3]) - max(xy[i, 1], xy[j, 1])
                if iw > 0 and ih > 0 and iw * ih >= contain * area[i]:
                    keep[i] = False
                    break
            else:
                kept.append(i)
    return boxes[keep].reset_index(drop=True)


def draw(root, rows: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for uuid, g in rows.groupby("uuid", sort=False):
        im = Image.open(root / g.path.iloc[0]).convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        for r in g.itertuples():
            dr.rectangle([r.x0 * w, r.y0 * h, r.x1 * w, r.y1 * h], outline="red" if r.fallback else "lime", width=3)
            dr.text((r.x0 * w + 4, r.y0 * h + 2), f"{r.prompt} {r.score:.2f}", fill="yellow")
        im.save(out_dir / f"{g.genus.iloc[0]}_{uuid[:8]}.jpg", quality=85)


@torch.no_grad()
def main() -> None:
    cfg = load("detector")["labels"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--per-genus", type=int, default=cfg.get("per_genus", 100), help="images per genus (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: first N selected images")
    ap.add_argument("--min-score", type=float, default=cfg["min_box_score"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-boxes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048, help="images per resumable part file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads when --device cpu")
    ap.add_argument("--out", default="manifests/boxes.parquet", help="relative to TRANSFERNER_DATA")
    ap.add_argument("--parts", default="artifacts/pseudo_boxes", help="resumable part files, relative")
    ap.add_argument("--draw", type=int, default=0, help="also render the first N images with boxes")
    args = ap.parse_args()

    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    root = data_root()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    rows = select(root, args.splits, args.per_genus, args.limit)
    print(f"{len(rows):,} images, {rows.genus.nunique():,} genera, splits {rows.split.value_counts().to_dict()}",
          flush=True)
    proc = Owlv2Processor.from_pretrained(cfg["pseudo_labeler"])
    model = Owlv2ForObjectDetection.from_pretrained(cfg["pseudo_labeler"]).to(args.device).eval()
    prompts = list(cfg["prompts"])
    text = proc.tokenizer(prompts, padding="max_length", return_tensors="pt").to(args.device)
    amp = torch.autocast("cuda", dtype=torch.float16, enabled=args.device.startswith("cuda"))

    parts = root / args.parts
    parts.mkdir(parents=True, exist_ok=True)
    # Resume by uuid, not by position: a larger --per-genus later labels only the new images.
    existing = sorted(parts.glob("part-*.parquet"))
    seen = set().union(*(set(pd.read_parquet(p, columns=["uuid"]).uuid) for p in existing)) if existing else set()
    todo = rows[~rows.uuid.isin(seen)].reset_index(drop=True)
    first = 1 + max((int(p.stem.split("-")[1]) for p in existing), default=-1)
    print(f"{len(rows) - len(todo):,} already labelled in {parts}, {len(todo):,} to go", flush=True)
    t0, done = time.time(), 0
    for k, c0 in enumerate(range(0, len(todo), args.chunk)):
        part = parts / f"part-{first + k:05d}.parquet"
        chunk = todo.iloc[c0:c0 + args.chunk].reset_index(drop=True)
        dl = DataLoader(Images(root, chunk.path, proc.image_processor), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
        out, j = [], 0
        for px, sizes in dl:
            n = len(px)
            with amp:
                o = model(input_ids=text.input_ids.repeat(n, 1), attention_mask=text.attention_mask.repeat(n, 1),
                          pixel_values=px.to(args.device))
            for b, s, lab in boxes_for(o.logits, o.pred_boxes, sizes, args.min_score, args.iou, args.max_boxes):
                r = chunk.iloc[j]
                if len(b) == 0:
                    b, s, lab, fb = np.array([[0, 0, 1, 1]], np.float32), np.array([np.nan]), None, True
                else:
                    fb = False
                out.append(pd.DataFrame({
                    "uuid": r["uuid"], "split": r["split"], "class_idx": r["class_idx"], "genus": r["genus"],
                    "family": r["family"], "x0": b[:, 0], "y0": b[:, 1], "x1": b[:, 2], "y1": b[:, 3],
                    "score": s.astype(np.float32),
                    "prompt": "full_frame" if fb else [prompts[k] for k in lab], "fallback": fb}))
                j += 1
            done += n
        tmp = part.with_suffix(".tmp")
        pd.concat(out, ignore_index=True).to_parquet(tmp, index=False)
        tmp.rename(part)                                     # a killed run never leaves a half-written part
        rate = done / (time.time() - t0)
        print(f"{part.name}: {min(c0 + args.chunk, len(todo)):,}/{len(todo):,} images ({rate:.1f} img/s)", flush=True)

    boxes = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part-*.parquet"))], ignore_index=True)
    boxes = boxes[boxes.uuid.isin(rows.uuid)].reset_index(drop=True)
    dst = root / args.out
    dst.parent.mkdir(parents=True, exist_ok=True)
    boxes.to_parquet(dst, index=False)
    per = boxes.groupby("uuid").agg(n=("x0", "size"), fb=("fallback", "first"))
    area = ((boxes.x1 - boxes.x0) * (boxes.y1 - boxes.y0))[~boxes.fallback]
    print(f"wrote {dst}: {len(boxes):,} boxes on {len(per):,} images; fallback {per.fb.mean():.1%}; "
          f"boxes/image {per.n[~per.fb].mean():.2f}; median box area {area.median():.2f}; "
          f"prompts {boxes.prompt.value_counts().to_dict()}", flush=True)
    if args.draw:
        first = rows.uuid.iloc[:args.draw]
        draw(root, boxes[boxes.uuid.isin(first)].merge(rows[["uuid", "path"]], on="uuid"),
             parts / "draw")


if __name__ == "__main__":
    main()

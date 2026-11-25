"""BioCLIP check of every OWLv2 box, and the D-046 box set (user request 2026-11-24: "additional data that has
distant plants or plants that are in a cluttered scene").

OWLv2 boxes "a plant" or "a tree" wherever one is; on a cluttered or distant scene the box can land on a
neighbouring plant, a person or a fence instead of the photo's subject, and the detector then learns that
genus there. Each non-fallback box is cropped (10% margin, as the composites cut it), embedded with BioCLIP and
scored with the teacher probe (calibrated, summed to genera): crop_rank is the rank of the photo's genus among
all genera on the crop, crop_p its probability; photo_p is the same genus' probability on the whole photo (cached
embedding). A box is kept when its genus is within --max-rank on the crop, or when BioCLIP cannot name the
genus on the whole photo either (photo rank > --max-rank: the crop then says nothing about the box).

Box set v3 (--out): manifests/boxes.parquet, with the photos re-boxed under habit prompts (boxes_v2_rebox:
every photo with a box its genus' growth habit rules out, e.g. a trunk on a forb) replaced, plus the new photos
of boxes_v2 (never-boxed regional photos, the unboxed perennial genera, scene-ranked ToL photos); then boxes
failing the check dropped, and photos left with no box dropped (they stay in the unboxed whole-frame stream).
The dropped boxes on photos that keep others go to --ignore-out, so a trainer can ignore them instead of
reading them as background (D-046b).
  python -m transferner.data.box_check [--max-rank 10]          # scores are cached in artifacts/box_check/"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models import bioclip
from ..utils.config import data_root


def assemble(root, base: str, rebox: str, extra: str) -> pd.DataFrame:
    b = pd.read_parquet(root / base).assign(src="v1")
    parts = [b]
    if (root / rebox).exists():
        r = pd.read_parquet(root / rebox).assign(src="rebox")
        parts = [b[~b.uuid.isin(set(r.uuid))], r]
    if (root / extra).exists():
        e = pd.read_parquet(root / extra).assign(src="v2")
        have = set().union(*(set(p.uuid) for p in parts))
        parts.append(e[~e.uuid.isin(have)])
    out = pd.concat(parts, ignore_index=True)
    print("assembled: " + ", ".join(f"{k} {v:,} photos" for k, v in out.groupby("src").uuid.nunique().items()), flush=True)
    return out


class Crops(Dataset):
    """One item per photo: all its box crops, preprocessed (decoded once)."""

    def __init__(self, root, photos: list, pre):
        self.root, self.photos, self.pre = root, photos, pre

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, i):
        path, boxes = self.photos[i]
        try:
            im = Image.open(self.root / path).convert("RGB")
        except Exception:                                    # noqa: BLE001 - an unreadable photo: all boxes unscored
            return torch.zeros(0, 3, 224, 224), i
        w, h = im.size
        out = []
        for x0, y0, x1, y1 in boxes:
            mx, my = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            c = im.crop((max(0, int((x0 - mx) * w)), max(0, int((y0 - my) * h)), min(w, int((x1 + mx) * w) + 1),
                         min(h, int((y1 + my) * h) + 1)))
            out.append(self.pre(c))
        return torch.stack(out), i


def collate(batch):
    x, idx = zip(*batch)
    n = [len(t) for t in x]
    return torch.cat(x), torch.tensor(np.repeat(idx, n))


def genus_scores(root, emb: torch.Tensor, gidx: torch.Tensor, W, b, scale, T, s2g, G) -> tuple[np.ndarray, np.ndarray]:
    p = ((scale * emb @ W.T + b) / T).softmax(-1)
    g = torch.zeros(len(emb), G, device=emb.device).index_add_(1, s2g, p)
    pg = g.gather(1, gidx[:, None])[:, 0]
    rank = (g > pg[:, None]).sum(1) + 1
    return pg.float().cpu().numpy(), rank.cpu().numpy()


@torch.no_grad()
def score(root, bx: pd.DataFrame, args) -> pd.DataFrame:
    dev = args.device
    sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "genus"]).sort_values("class_idx")
    genera = sorted(sp.genus.unique())
    gi = {g: i for i, g in enumerate(genera)}
    s2g = torch.tensor([gi[g] for g in sp.genus], device=dev)
    pr = torch.load(root / args.probe, map_location=dev)
    W, b, scale = pr["W"].float(), pr["b"].float(), float(pr["scale"])
    T = float((root / args.temperature).read_text())
    G = len(genera)
    real = bx[~bx.fallback].copy()
    real["row"] = np.arange(len(real))
    paths = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    real = real.merge(paths, on="uuid")
    photos, rows = [], []
    for (u, path), d in real.groupby(["uuid", "path"], sort=False):
        photos.append((path, d[["x0", "y0", "x1", "y1"]].to_numpy(np.float32)))
        rows.append(d.index.to_numpy())
    genus_of = real.genus.map(gi).to_numpy()
    model, pre, _ = bioclip.load(torch.device(dev))
    visual = model.visual.eval()
    crop_p, crop_rank = np.full(len(real), np.nan, np.float32), np.zeros(len(real), np.int32)
    dl = DataLoader(Crops(root, photos, pre), batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                    pin_memory=True, prefetch_factor=4)
    for x, idx in tqdm(dl, desc=f"crops of {len(photos):,} photos", mininterval=60):
        if not len(x):
            continue
        with torch.autocast("cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16):
            e = F.normalize(visual(x.to(dev, non_blocking=True)).float(), dim=-1)
        order = np.concatenate([rows[i] for i in idx.unique_consecutive().numpy()])   # crops come in item order
        assert len(order) == len(e), (len(order), len(e))
        p, k = genus_scores(root, e, torch.as_tensor(genus_of[order], device=dev), W, b, scale, T, s2g, G)
        crop_p[order], crop_rank[order] = p, k
    # the whole photo, from the cached embeddings of the same teacher input
    photo_p, photo_rank = np.full(len(real), np.nan, np.float32), np.zeros(len(real), np.int32)
    for split in ("train", "val", "test"):
        meta = pd.read_parquet(root / args.emb / f"{split}.parquet", columns=["uuid"])
        meta["erow"] = np.arange(len(meta))
        m = real[["uuid"]].reset_index().merge(meta, on="uuid")
        if m.empty:
            continue
        E = np.load(root / args.emb / f"{split}.npy", mmap_mode="r")
        for s in range(0, len(m), 65536):
            c = m.iloc[s:s + 65536]
            e = torch.from_numpy(np.asarray(E[np.sort(c.erow.to_numpy())], np.float32)).to(dev)
            c = c.sort_values("erow")
            p, k = genus_scores(root, e, torch.as_tensor(genus_of[c["index"].to_numpy()], device=dev), W, b, scale, T, s2g, G)
            photo_p[c["index"].to_numpy()], photo_rank[c["index"].to_numpy()] = p, k
    real["crop_p"], real["crop_rank"], real["photo_p"], real["photo_rank"] = crop_p, crop_rank, photo_p, photo_rank
    return real.drop(columns=["path", "row"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="manifests/boxes.parquet")
    ap.add_argument("--rebox", default="manifests/boxes_v2_rebox.parquet")
    ap.add_argument("--extra", default="manifests/boxes_v2.parquet")
    ap.add_argument("--out", default="manifests/boxes_v3.parquet")
    ap.add_argument("--ignore-out", default="manifests/boxes_v3_ignore.parquet",
                    help="the rejected boxes on photos that keep others: ignore regions for the trainer (D-046b)")
    ap.add_argument("--cache", default="artifacts/box_check/scores.parquet")
    ap.add_argument("--emb", default="artifacts/img_emb_v2")
    ap.add_argument("--probe", default="artifacts/probe_v2.pt")
    ap.add_argument("--temperature", default="artifacts/temperature_v2.txt")
    ap.add_argument("--max-rank", type=int, default=10)
    ap.add_argument("--batch", type=int, default=96, help="photos per batch (each yields 1-10 crops)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: first N photos, no cache, no output")
    args = ap.parse_args()
    root = data_root()
    bx = assemble(root, args.base, args.rebox, args.extra)
    if args.limit:
        bx = bx[bx.uuid.isin(set(bx.uuid.drop_duplicates().sample(args.limit, random_state=0)))]
        sc = score(root, bx, args)
        print(sc[["genus", "prompt", "crop_p", "crop_rank", "photo_p", "photo_rank"]].describe().to_string())
        print(sc.sample(min(12, len(sc)), random_state=0)[["genus", "prompt", "x0", "y0", "x1", "y1", "crop_p", "crop_rank", "photo_rank"]].to_string())
        return
    cache = root / args.cache
    if cache.exists():
        sc = pd.read_parquet(cache)
    else:
        cache.parent.mkdir(parents=True, exist_ok=True)
        sc = score(root, bx, args)
        sc.to_parquet(cache, index=False)
    key = ["uuid", "x0", "y0", "x1", "y1", "prompt"]
    bx = bx.merge(sc[key + ["crop_p", "crop_rank", "photo_p", "photo_rank"]].drop_duplicates(key), on=key, how="left")
    real = ~bx.fallback
    area = (bx.x1 - bx.x0) * (bx.y1 - bx.y0)
    unsure = bx.photo_rank.isna() | (bx.photo_rank == 0) | (bx.photo_rank > args.max_rank)
    ok = ~real | (bx.crop_rank.between(1, args.max_rank)) | unsure
    bx["check_ok"] = ok
    for name, m in (("all real boxes", real), ("area >= 50%", real & (area >= 0.5)), ("area 10-50%", real & area.between(0.1, 0.5)),
                    ("area 2-10%", real & area.between(0.02, 0.1)), ("area < 2%", real & (area < 0.02))):
        d = bx[m]
        print(f"{name:16s} {len(d):>8,} boxes: crop genus rank <= {args.max_rank} {(d.crop_rank.between(1, args.max_rank)).mean():.1%}, "
              f"whole photo unsure {unsure[m].mean():.1%}, kept {d.check_ok.mean():.1%}", flush=True)
    print("kept by prompt: " + ", ".join(f"{k} {v:.0%}" for k, v in bx[real].groupby("prompt").check_ok.mean().items()), flush=True)
    print("kept by source: " + ", ".join(f"{k} {v:.0%}" for k, v in bx[real].groupby("src").check_ok.mean().items()), flush=True)
    out = bx[bx.check_ok]
    lost = set(bx.uuid) - set(out.uuid)
    rej = bx[real & ~bx.check_ok & bx.uuid.isin(set(out.uuid))]
    rej[["uuid", "split", "genus", "prompt", "x0", "y0", "x1", "y1", "crop_rank", "photo_rank"]].to_parquet(root / args.ignore_out, index=False)
    print(f"wrote {args.ignore_out}: {len(rej):,} rejected boxes on {rej.uuid.nunique():,} photos that keep other boxes", flush=True)
    out = out.drop(columns=["check_ok"])
    out.to_parquet(root / args.out, index=False)
    print(f"wrote {args.out}: {len(out):,} boxes on {out.uuid.nunique():,} photos ({len(lost):,} photos lost every box); "
          f"by split {out.drop_duplicates('uuid').split.value_counts().to_dict()}; genera with train boxes "
          f"{out[out.split == 'train'].genus.nunique():,}", flush=True)


if __name__ == "__main__":
    main()

"""Cache frozen BioCLIP image embeddings for train/val/test (one pass; probe trains on these).

Writes artifacts/img_emb/<split>.npy (float16, L2-normalized) and <split>.parquet
(uuid, class_idx, is_specimen) in the same row order. Resumable per split."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models import bioclip
from ..utils.config import data_root


class Images(Dataset):
    def __init__(self, root, paths, preprocess):
        self.root, self.paths, self.pre = root, paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.pre(Image.open(self.root / self.paths[i]).convert("RGB"))


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = data_root()
    out = root / "artifacts/img_emb"
    out.mkdir(parents=True, exist_ok=True)
    spl = pd.read_parquet(root / "manifests/splits.parquet")
    paths = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    spl = spl.merge(paths, on="uuid")
    model, pre, _ = bioclip.load()
    visual = model.visual

    for s in args.splits:
        if (out / f"{s}.npy").exists() and not args.force:
            print(f"{s}: cached")
            continue
        d = spl[spl.split == s].reset_index(drop=True)
        dl = DataLoader(Images(root, d.path.tolist(), pre), batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, persistent_workers=False, prefetch_factor=4)
        embs = []
        for x in tqdm(dl, desc=f"embed {s}"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = visual(x.to("cuda", non_blocking=True))
            embs.append(F.normalize(e.float(), dim=-1).half().cpu())
        np.save(out / f"{s}.npy", torch.cat(embs).numpy())
        d[["uuid", "class_idx", "is_specimen", "source"]].to_parquet(out / f"{s}.parquet", index=False)
        print(f"{s}: {len(d):,} embeddings")


if __name__ == "__main__":
    main()

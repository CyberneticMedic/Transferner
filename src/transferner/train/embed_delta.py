"""Incremental BioCLIP embeddings for a changed splits.parquet (D-045).

Re-embedding all ~2.3M photos for every label-set repair is hours of GPU. Embeddings depend only on the photo,
so this reuses every cached vector by uuid (artifacts/img_emb and the append-only delta store) and embeds only
the photos that have none (backfilled ToL rows, rows moved out of 'unused' by the D-009 cap, the month-balanced
pull, new species). It then assembles a full embedding set for the new splits in --out, in the layout
embed_images.py writes (<split>.npy float16 L2-normalised + <split>.parquet uuid/class_idx/is_specimen/source),
so linear_probe.py / calibrate.py / evaluate.py run on it with --emb.
Decoding matches embed_images.py (PIL draft decode at >= 224 px) unless --no-draft; --skew-check N embeds N
regional test photos both ways and reports the production probe's top-1 on each (audit capture_lookalike_
resolution-6: the cache used a reduced-scale JPEG decode the app does not).
  python -m transferner.train.embed_delta --out artifacts/img_emb_v2 [--skew-check 4000]"""
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

SPLITS = ("train", "val", "test", "train_ext", "val_ext", "test_season")


class Images(Dataset):
    def __init__(self, root, paths, preprocess, draft=True):
        self.root, self.paths, self.pre, self.draft = root, paths, preprocess, draft

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.root / self.paths[i])
        if self.draft:
            im.draft("RGB", (224, 224))
        return self.pre(im.convert("RGB"))


@torch.no_grad()
def embed(visual, pre, root, paths, batch, workers, draft=True) -> np.ndarray:
    dl = DataLoader(Images(root, list(paths), pre, draft), batch_size=batch, num_workers=workers, pin_memory=True, prefetch_factor=4)
    out = []
    for x in tqdm(dl, desc=f"embed {len(paths):,}"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = visual(x.to("cuda", non_blocking=True))
        out.append(F.normalize(e.float(), dim=-1).half().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, visual.output_dim), np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/img_emb")
    ap.add_argument("--store", default="artifacts/img_emb_delta", help="append-only store of new embeddings")
    ap.add_argument("--out", default="artifacts/img_emb_v2")
    ap.add_argument("--no-draft", action="store_true")
    ap.add_argument("--skew-check", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    root = data_root()
    src, store, out = root / args.src, root / args.store, root / args.out
    store.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    model, pre, _ = bioclip.load()
    visual = model.visual.eval()

    if args.skew_check:
        pr = torch.load(root / "artifacts/probe.pt", map_location="cuda")
        T = float((root / "artifacts/temperature.txt").read_text())
        m = pd.read_parquet(src / "test.parquet")
        m["row"] = np.arange(len(m))
        m = m[m.source == "gbif_regional"].sample(args.skew_check, random_state=0)
        paths = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]).set_index("uuid").path
        res = {}
        for name, draft in (("draft", True), ("full", False)):
            e = torch.from_numpy(embed(visual, pre, root, paths[m.uuid].to_numpy(), args.batch, args.workers, draft)).float().cuda()
            p = ((pr["scale"] * e @ pr["W"].float().T + pr["b"].float()) / T).argmax(1).cpu().numpy()
            res[name] = (p == m.class_idx.to_numpy()).mean()
        cached = torch.from_numpy(np.load(src / "test.npy", mmap_mode="r")[np.sort(m.row.to_numpy())].astype(np.float32))
        print(f"decode skew on {args.skew_check:,} regional test photos: top-1 draft {res['draft']:.4f}, full decode {res['full']:.4f} "
              f"(cached vectors: {len(cached):,})", flush=True)

    # index of every vector we already have
    have = {}
    for s in SPLITS:
        if (src / f"{s}.parquet").exists():
            for i, u in enumerate(pd.read_parquet(src / f"{s}.parquet", columns=["uuid"]).uuid):
                have[u] = (str(src / f"{s}.npy"), i)
    for p in sorted(store.glob("delta-*.parquet")):
        for i, u in enumerate(pd.read_parquet(p, columns=["uuid"]).uuid):
            have[u] = (str(p.with_suffix(".npy")), i)
    spl = pd.read_parquet(root / "manifests/splits.parquet")
    spl = spl[spl.split.isin(SPLITS)].merge(pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")
    need = spl[~spl.uuid.isin(have)]
    print(f"{len(spl):,} rows in {list(SPLITS)}; cached {spl.uuid.isin(have).sum():,}; to embed {len(need):,} "
          f"({need.split.value_counts().to_dict()})", flush=True)
    if len(need):
        k = len(list(store.glob("delta-*.parquet")))
        e = embed(visual, pre, root, need.path.to_numpy(), args.batch, args.workers, not args.no_draft)
        np.save(store / f"delta-{k:03d}.npy", e)
        need[["uuid"]].to_parquet(store / f"delta-{k:03d}.parquet", index=False)
        for i, u in enumerate(need.uuid):
            have[u] = (str(store / f"delta-{k:03d}.npy"), i)
    arrays = {}
    for s in SPLITS:
        d = spl[spl.split == s].reset_index(drop=True)
        if d.empty:
            continue
        dst = np.lib.format.open_memmap(out / f"{s}.npy", mode="w+", dtype=np.float16, shape=(len(d), visual.output_dim))
        loc = pd.DataFrame([have[u] for u in d.uuid], columns=["file", "row"])
        for f, g in loc.groupby("file"):
            a = arrays.setdefault(f, np.load(f, mmap_mode="r"))
            order = g.row.to_numpy().argsort()
            dst[g.index.to_numpy()[order]] = a[g.row.to_numpy()[order]]
        dst.flush(); del dst
        d[["uuid", "class_idx", "is_specimen", "source"]].to_parquet(out / f"{s}.parquet", index=False)
        print(f"{s}: {len(d):,} rows", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

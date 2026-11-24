"""Open Images V7 plant boxes: real small and distant plants in everyday clutter (D-046; user request 2026-11-24,
"additional data that has distant plants or plants in a cluttered scene").

The D-044 live model boxes composites well (AP50 0.86) but real scenes poorly: against OWLv2 boxes, AP50 is 0.19 on
photos whose plant fills under 10% of the frame and 0.27 on scene photos, against 0.72 on single large plants
(scratchpad mining, 2026-11-24). Open Images has human-drawn boxes for Tree, Plant, Palm tree, Houseplant, Maple and
Willow on Flickr street, park, yard and field photos: the median plant box covers 2% of the frame. Annotations are
CC BY 4.0, images CC BY 2.0 (license and author kept per image). A survey and an adversarial check of 15 sources
picked it as the only verified source of real distant plant boxes in clutter.

Box roles, so the D-045 part-firing regression does not come back:
  target   Tree, Plant, Palm tree, Houseplant, Maple, Willow boxes that are not group boxes (whole plants)
  ignore   group boxes (tree lines, hedges, lawns) and bloom classes (Flower, Rose, Lavender, Lily, Common sunflower),
           which are drawn per flower: neither positive nor negative for the SSD loss
Images with any plant box marked IsDepiction (paintings, prints) or IsInside are dropped. A class is boxed
exhaustively only where it was verified present, so unboxed regions of an image are not background: the
trainer should treat them as ignore unless the image-level labels verify Tree and Plant absent.
Genus classes (Maple -> Acer, Willow -> Salix, Rose -> Rosa, Lavender -> Lavandula, Common sunflower -> Helianthus,
Lily -> Lilium) are kept as a genus column for whole-frame labels.

Writes manifests/openimages_plants.parquet (one row per box) and manifests/openimages_plants_images.parquet (one
row per image: path, OI split, license, author, url, counts, largest target box); --fetch downloads the selected
photos into raw/openimages_plants/<train|validation>/ (resumable).
  python -m transferner.data.openimages_plants --csv            # annotation CSVs (2.3 GB), once
  python -m transferner.data.openimages_plants [--max-train 100000] [--fetch]"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import data_root

RAW = "raw/openimages_plants"
CSV = {
    "train-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "validation-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "validation-annotations-human-imagelabels.csv":
        "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv",
}
TARGET = {"/m/07j7r": "Tree", "/m/05s2s": "Plant", "/m/0cdl1": "Palm tree", "/m/03fp41": "Houseplant",
          "/m/0cffdh": "Maple", "/m/0mw_6": "Willow"}
BLOOM = {"/m/0c9ph5": "Flower", "/m/06m11": "Rose", "/m/04gth": "Lavender (Plant)", "/m/0jqgx": "Lily",
         "/m/0ftb8": "Common sunflower"}
GENUS = {"Maple": "Acer", "Willow": "Salix", "Rose": "Rosa", "Lavender (Plant)": "Lavandula",
         "Common sunflower": "Helianthus", "Lily": "Lilium"}
COLS = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]


def fetch_csvs(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, url in CSV.items():
        dst = d / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        t0 = time.time()
        with urllib.request.urlopen(url, timeout=60) as r, open(dst.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
        dst.with_suffix(".part").replace(dst)
        print(f"{name}: {dst.stat().st_size / 2**20:,.0f} MB in {time.time() - t0:.0f} s", flush=True)


def plant_boxes(path: Path) -> pd.DataFrame:
    mids = set(TARGET) | set(BLOOM)
    parts = [c[c.LabelName.isin(mids)] for c in pd.read_csv(path, usecols=COLS, chunksize=2_000_000)]
    b = pd.concat(parts, ignore_index=True)
    b["label"] = b.LabelName.map({**TARGET, **BLOOM})
    b["role"] = np.where(b.LabelName.isin(set(TARGET)) & (b.IsGroupOf != 1), "target", "ignore")
    b["genus"] = b.label.map(GENUS)
    b["area"] = (b.XMax - b.XMin) * (b.YMax - b.YMin)
    return b


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16)


def fetch_images(root: Path, img: pd.DataFrame, workers: int = 24) -> int:
    def one(row) -> bool:
        dst = root / row.path
        if dst.exists() and dst.stat().st_size > 0:
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://open-images-dataset.s3.amazonaws.com/{row.oi_split}/{row.image_id}.jpg"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dst)
                return True
            except Exception:                                   # noqa: BLE001 - a re-run retries
                time.sleep(2 + 4 * attempt)
        return False

    ok, t0 = 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, good in enumerate(ex.map(one, img.itertuples()), 1):
            ok += good
            if i % 5000 == 0:
                print(f"  {i:,}/{len(img):,} fetched ({ok:,} ok, {i / (time.time() - t0):.0f}/s)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="download the annotation CSVs and exit")
    ap.add_argument("--max-train", type=int, default=100_000, help="train images to keep (genus-class images first, then by hash)")
    ap.add_argument("--fetch", action="store_true", help="download the selected photos")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    root = data_root()
    d = root / RAW
    if args.csv:
        fetch_csvs(d)
        return
    neg = root / "raw/negatives/openimages"                     # D-044 already holds the train image list and labels
    rows = []
    for oi_split, box_csv, img_csv, lab_csv in (
            ("train", d / "train-annotations-bbox.csv", neg / "train-images-boxable-with-rotation.csv",
             neg / "oidv7-train-annotations-human-imagelabels.csv"),
            ("validation", d / "validation-annotations-bbox.csv", d / "validation-images-with-rotation.csv",
             d / "validation-annotations-human-imagelabels.csv")):
        b = plant_boxes(box_csv)
        bad = set(b[(b.IsDepiction == 1) | (b.IsInside == 1)].ImageID)
        b = b[~b.ImageID.isin(bad)]
        per = b.groupby("ImageID").agg(n_target=("role", lambda r: int((r == "target").sum())),
                                       n_ignore=("role", lambda r: int((r == "ignore").sum())),
                                       max_target=("area", lambda a: 0.0),
                                       genus=("genus", lambda g: next((x for x in g if isinstance(x, str)), None)))
        per["max_target"] = b[b.role == "target"].groupby("ImageID").area.max().reindex(per.index).fillna(0.0)
        per = per[per.n_target > 0]
        if oi_split == "train" and len(per) > args.max_train:
            pri = per.genus.notna().astype(int) * 2 + (per.max_target < 0.25).astype(int)   # genus images, then small plants
            order = pd.DataFrame({"pri": pri, "h": [h(i) for i in per.index]}, index=per.index).sort_values(["pri", "h"], ascending=[False, True])
            per = per.loc[order.index[:args.max_train]]
        lab = pd.read_csv(lab_csv, usecols=["ImageID", "LabelName", "Confidence"])
        lab = lab[lab.ImageID.isin(per.index) & lab.LabelName.isin(["/m/07j7r", "/m/05s2s"])]
        ver = lab.assign(k=lab.LabelName.map({"/m/07j7r": "tree", "/m/05s2s": "plant"}) + np.where(lab.Confidence > 0, "_yes", "_no"))
        ver = ver.groupby("ImageID").k.apply(lambda s: ",".join(sorted(s)))
        meta = pd.read_csv(img_csv, usecols=["ImageID", "License", "Author", "OriginalLandingURL"]).set_index("ImageID")
        per = per.join(meta, how="left").join(ver.rename("verified"), how="left")
        per["oi_split"] = oi_split
        per["path"] = [f"{RAW}/{oi_split}/{i}.jpg" for i in per.index]
        rows.append((b[b.ImageID.isin(per.index)].assign(oi_split=oi_split), per))
        print(f"{oi_split}: {len(per):,} images kept ({len(bad):,} with depiction/inside boxes dropped); boxes "
              f"{b[b.ImageID.isin(per.index)].role.value_counts().to_dict()}; largest target box under 10% of the frame: "
              f"{(per.max_target < 0.10).mean():.1%}, under 25%: {(per.max_target < 0.25).mean():.1%}; "
              f"genus classes: {per.genus.value_counts().to_dict()}", flush=True)
    boxes = pd.concat([r[0] for r in rows], ignore_index=True)
    img = pd.concat([r[1] for r in rows]).rename_axis("image_id").reset_index()
    img = img.rename(columns={"License": "license", "Author": "author", "OriginalLandingURL": "url"})
    boxes = boxes.rename(columns={"ImageID": "image_id", "XMin": "x0", "XMax": "x1", "YMin": "y0", "YMax": "y1"})[
        ["image_id", "oi_split", "label", "role", "genus", "x0", "y0", "x1", "y1", "area", "IsOccluded", "IsTruncated", "IsGroupOf"]]
    boxes.to_parquet(root / "manifests/openimages_plants.parquet", index=False)
    img.to_parquet(root / "manifests/openimages_plants_images.parquet", index=False)
    print(f"wrote manifests/openimages_plants.parquet ({len(boxes):,} boxes) and openimages_plants_images.parquet ({len(img):,} images)", flush=True)
    if args.fetch:
        free = shutil.disk_usage(root).free / 2**30
        need = 0.35 * len(img) / 1024
        assert free > need + 200, f"only {free:.0f} GB free for ~{need:.0f} GB"
        ok = fetch_images(root, img, args.workers)
        print(f"fetched {ok:,}/{len(img):,} photos into {RAW}/", flush=True)


if __name__ == "__main__":
    main()

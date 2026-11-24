"""Background and true-negative imagery for the live detector (user request 2026-11-24; D-044).

Two kinds of frames, from three sources downloaded by scripts/fetch_negatives.sh:
  plant_free   frames with no plant at all, to teach objectness and the whole-frame head "no plant here" and to
               calibrate the plant-presence gate:
                 COCO 2017      no stuff pixels of tree/grass/bush/plant-other/flower/leaves/moss/straw/branch, nor of
                                playingfield/hill/mountain (usually turf or vegetated slopes), no potted plant, and no
                                plant food (stuff fruit/vegetable/salad; things banana, apple, orange, broccoli, carrot)
                 Open Images    human-verified "Plant" absent, and no vegetation label verified present
  vegetation   scenes dominated by unlabelled vegetation (forest, field, meadow): canvases for seam-free
               copy-paste composites (audit segmentation_localization-01), and "plants, but not one subject":
                 COCO 2017      plant stuff >= 50% of the pixels
                 Places365      val_large photos of natural vegetated categories (forest, field, meadow, ...)
                 Open Images    verified Forest / Woodland / Grassland / Meadow / Jungle / Vegetation present
Splits are by a hash of the image id (80/10/10), so negatives never cross train/val/test.
Writes $TRANSFERNER_DATA/manifests/negatives.parquet; with --fetch-openimages N also downloads N Open Images
photos per kind (resumable, 16 at a time, from the public S3 bucket) into raw/negatives/openimages/<kind>/.
  python -m transferner.data.negatives [--fetch-openimages 30000]"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..utils.config import data_root

PLANT_STUFF = {"tree", "grass", "bush", "plant-other", "flower", "leaves", "moss", "straw", "branch"}
# not plant labels, but usually turf or vegetated slopes: never plant-free (0.6 review: 6% of the first
# plant_free set were sports fields, which are lawns of detector genera)
MAYBE_PLANT_STUFF = {"playingfield", "hill", "mountain"}
FOOD_STUFF = {"fruit", "vegetable", "salad"}
PLANT_THINGS = {"potted plant", "banana", "apple", "orange", "broccoli", "carrot"}
OI_PLANT = "/m/05s2s"
OI_VEG = {"/m/07j7r": "Tree", "/m/0c9ph5": "Flower", "/m/08t9c_": "Grass", "/m/03fp41": "Houseplant", "/m/09t49": "Leaf",
          "/m/0hkmy": "Fern", "/m/0184yy": "Moss", "/m/01fnns": "Vegetation", "/m/02zr8": "Forest", "/m/0hnc1": "Woodland",
          "/m/01c7cq": "Grassland", "/m/025st_8": "Meadow", "/m/01y3fy": "Jungle", "/m/0fm3zh": "Flowerpot", "/m/02xwb": "Fruit",
          "/m/0f4s2w": "Vegetable", "/m/0brn2d": "Field", "/m/03d28y3": "Natural landscape"}
OI_SCENE = {"/m/02zr8", "/m/0hnc1", "/m/01c7cq", "/m/025st_8", "/m/01y3fy", "/m/01fnns"}
PLACES_VEG = ("forest", "forest_path", "forest_road", "field/wild", "field/cultivated", "field_road", "rainforest", "bamboo_forest",
              "swamp", "marsh", "pasture", "tree_farm", "orchard", "vineyard", "corn_field", "wheat_field", "rice_paddy", "hayfield",
              "botanical_garden", "formal_garden", "vegetable_garden", "japanese_garden", "topiary_garden", "yard", "lawn", "park",
              "creek", "river", "picnic_area", "campsite", "valley")


def split_of(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def coco(root: Path) -> pd.DataFrame:
    d = root / "raw/negatives/coco"
    rows = []
    for split in ("train2017", "val2017"):
        stuff = json.loads((d / f"annotations/stuff_{split}.json").read_text())
        inst = json.loads((d / f"annotations/instances_{split}.json").read_text())
        sname = {c["id"]: c["name"] for c in stuff["categories"]}
        iname = {c["id"]: c["name"] for c in inst["categories"]}
        size = {im["id"]: (im["file_name"], im["width"] * im["height"], im.get("license", 0)) for im in inst["images"]}
        veg, food = {}, set()
        for a in stuff["annotations"]:
            n = sname[a["category_id"]]
            if n in PLANT_STUFF:
                veg[a["image_id"]] = veg.get(a["image_id"], 0) + a["area"]
            elif n in FOOD_STUFF or n in MAYBE_PLANT_STUFF:
                food.add(a["image_id"])                                   # excluded from plant_free and vegetation
        for a in inst["annotations"]:
            if iname[a["category_id"]] in PLANT_THINGS:
                food.add(a["image_id"])
        for i, (fn, area, lic) in size.items():
            v = veg.get(i, 0) / max(area, 1)
            kind = "plant_free" if (i not in veg and i not in food) else "vegetation" if v >= 0.5 and i not in food else None
            if kind:
                rows.append({"path": f"raw/negatives/coco/{split}/{fn}", "source": "coco", "kind": kind, "veg_frac": round(v, 3),
                             "license": f"coco:{lic}", "split": split_of(f"coco{i}")})
        del stuff, inst
    return pd.DataFrame(rows)


def places(root: Path) -> pd.DataFrame:
    d = root / "raw/negatives/places"
    cats = [l.split()[0][3:] for l in (d / "categories_places365.txt").read_text().splitlines()]
    val = pd.read_csv(d / "places365_val.txt", sep=" ", names=["file", "cat"])
    val["category"] = val.cat.map(lambda c: cats[c])
    keep = val[val.category.map(lambda c: any(c == p or c.startswith(p + "/") for p in PLACES_VEG))]
    return pd.DataFrame({"path": "raw/negatives/places/val_large/" + keep.file, "source": "places365", "kind": "vegetation",
                         "veg_frac": float("nan"), "license": "places365:research", "split": [split_of("pl" + f) for f in keep.file],
                         "category": keep.category})


def openimages_select(root: Path, per_kind: int, seed: int = 0) -> pd.DataFrame:
    d = root / "raw/negatives/openimages"
    boxable = pd.read_csv(d / "train-images-boxable-with-rotation.csv", usecols=["ImageID", "License", "Author", "OriginalLandingURL"])
    lab = pd.read_csv(d / "oidv7-train-annotations-human-imagelabels.csv", usecols=["ImageID", "LabelName", "Confidence"],
                      dtype={"Confidence": "float32"})
    lab = lab[lab.LabelName.isin(set(OI_VEG) | {OI_PLANT})]
    pos = lab[lab.Confidence > 0]
    neg_plant = set(lab[(lab.LabelName == OI_PLANT) & (lab.Confidence == 0)].ImageID)
    veg_pos = set(pos.ImageID)
    free = boxable[boxable.ImageID.isin(neg_plant - veg_pos)]
    scene = boxable[boxable.ImageID.isin(set(pos[pos.LabelName.isin(OI_SCENE)].ImageID))]
    out = []
    for kind, df in (("plant_free", free), ("vegetation", scene)):
        df = df.sample(min(per_kind, len(df)), random_state=seed)
        out.append(pd.DataFrame({"path": [f"raw/negatives/openimages/{kind}/{i}.jpg" for i in df.ImageID], "source": "openimages", "kind": kind,
                                 "veg_frac": float("nan"), "license": df.License.values, "split": [split_of("oi" + i) for i in df.ImageID],
                                 "image_id": df.ImageID.values, "author": df.Author.values, "url": df.OriginalLandingURL.values}))
        print(f"open images {kind}: {len(df):,} selected (of {len(free if kind == 'plant_free' else scene):,})", flush=True)
    return pd.concat(out, ignore_index=True)


def fetch(root: Path, df: pd.DataFrame, workers: int = 16) -> int:
    import urllib.request

    def one(row) -> bool:
        dst = root / row.path
        if dst.exists() and dst.stat().st_size > 0:
            return True
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://open-images-dataset.s3.amazonaws.com/train/{row.image_id}.jpg"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dst)
                return True
            except Exception:
                time.sleep(2 + 4 * attempt)
        return False

    ok = 0
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, good in enumerate(ex.map(one, df.itertuples()), 1):
            ok += good
            if i % 2000 == 0:
                print(f"  {i:,}/{len(df):,} fetched ({ok:,} ok, {i / (time.time() - t0):.0f}/s)", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-openimages", type=int, default=0, help="Open Images photos per kind to select and download")
    args = ap.parse_args()
    root = data_root()
    parts = [coco(root), places(root)]
    print(f"coco: {len(parts[0]):,} ({parts[0].kind.value_counts().to_dict()}); places365 vegetation: {len(parts[1]):,}", flush=True)
    if args.fetch_openimages:
        oi = openimages_select(root, args.fetch_openimages)
        ok = fetch(root, oi)
        oi = oi[[(root / p).exists() for p in oi.path]]
        print(f"open images: {ok:,} fetched, {len(oi):,} on disk", flush=True)
        parts.append(oi)
    m = pd.concat(parts, ignore_index=True)
    out = root / "manifests/negatives.parquet"
    m.to_parquet(out, index=False)
    print(m.groupby(["source", "kind", "split"]).size().unstack(fill_value=0).to_string())
    print(f"wrote {out}: {len(m):,} frames")


if __name__ == "__main__":
    main()

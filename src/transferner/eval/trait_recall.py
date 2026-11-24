"""Genus-balanced whole-frame recall on the genera that have cited trait rows (edible / conditional / toxic),
over all non-specimen val photos (ToL + regional), capped per genus. The audit's guard metric for D-040:
encounter weighting down-samples rare garden edibles and hemlocks, and the regional pool cannot see them.
  python -m transferner.eval.trait_recall artifacts/detector/mnv4_live/best.pt 320 [CKPT SIZE ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..train.train_detector import BoxSet, Net, build, collate, forward_raw, use_shift_add_depthwise
from ..utils.config import data_root, load

CAP = 40
root = data_root()
repo = Path(__file__).resolve().parents[3]
sp = pd.read_parquet(root / "manifests/species.parquet", columns=["class_idx", "scientific_name", "genus"])
tr = pd.read_csv(repo / "data/traits/species_traits.csv")
cats = tr.groupby("scientific_name").category.agg(set)
sp["edible"] = sp.scientific_name.map(lambda n: bool(cats.get(n, set()) & {"edible", "conditional"}))
sp["toxic"] = sp.scientific_name.map(lambda n: "toxic" in cats.get(n, set()))
g_ed, g_tox = set(sp.genus[sp.edible]), set(sp.genus[sp.toxic])
m = pd.read_parquet(root / "artifacts/img_emb/val.parquet", columns=["uuid", "class_idx", "is_specimen", "source"])
m = m[~m.is_specimen.fillna(False).astype(bool)].merge(sp[["class_idx", "genus"]], on="class_idx")
m = m.merge(pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"]), on="uuid")


def run(ckpt, size):
    ck = torch.load(root / ckpt, map_location="cpu")
    genera = json.loads((root / Path(ckpt).parent / "labels.json").read_text())["genera"]
    gi = {g: i for i, g in enumerate(genera)}
    d = m[m.genus.isin(gi) & m.genus.isin(g_ed | g_tox)]
    d = d.sample(frac=1, random_state=0).groupby("genus").head(CAP)
    model = build(len(genera), load("detector")["model"]["backbone"], size, pretrained=False, head="shared",
                  image_aux=True, image_head=1280)
    model.load_state_dict(ck["model"], strict=True)
    model.cuda().eval()
    use_shift_add_depthwise(model)
    net = Net(model)
    mean = torch.tensor(model.backbone.mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(model.backbone.std, device="cuda").view(1, 3, 1, 1)
    box = np.zeros((1, 4), np.float32)
    items = [(p, box, np.array([gi[g] + 1])) for p, g in zip(d.path, d.genus)]
    hit = np.zeros(len(genera)); cnt = np.zeros(len(genera))
    dl = DataLoader(BoxSet(root, items, size, False), batch_size=128, num_workers=8, collate_fn=collate)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for imgs, _, _, genus in dl:
            p = forward_raw(net, ((imgs.cuda().float() / 255) - mean) / std)["image_logits"].argmax(1).cpu().numpy()
            y = genus.numpy() - 1
            np.add.at(cnt, y, 1); np.add.at(hit, y, p == y)
    rec = pd.Series(np.where(cnt > 0, hit / np.maximum(cnt, 1), np.nan), index=genera)
    out = {"edible_genera": rec[[g for g in genera if g in g_ed]].mean(), "toxic_genera": rec[[g for g in genera if g in g_tox]].mean(),
           "n_genera": int((cnt > 0).sum()), "n_photos": int(cnt.sum())}
    watch = ["Ficus", "Punica", "Feijoa", "Mentha", "Asparagus", "Cicuta", "Conium", "Sambucus", "Hemerocallis", "Ilex", "Morus", "Vitis"]
    return out, rec.reindex(watch)


res = {}
args = sys.argv[1:]
for ck, sz in zip(args[::2], args[1::2]):
    out, w = run(ck, int(sz))
    res[f"{ck.split('/')[-2]}@{sz}"] = w
    print(ck, sz, {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}, flush=True)
print(pd.DataFrame(res).round(3).to_string())

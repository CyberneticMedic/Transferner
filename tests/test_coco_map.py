"""metrics.coco_map must match pycocotools' COCOeval on random detections (run: python tests/test_coco_map.py)."""
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from transferner.eval.metrics import coco_map


def random_case(seed, n_img=300, n_cls=25):
    r = np.random.default_rng(seed)
    gi, gc, gb, di, dc, ds, db = [], [], [], [], [], [], []
    for i in range(n_img):
        c = r.integers(n_cls)                                     # one class per image, like genus boxes
        for _ in range(r.integers(1, 4)):
            x0, y0 = r.uniform(0, 200, 2); w, h = r.uniform(10, 120, 2)
            b = [x0, y0, x0 + w, y0 + h]
            gi.append(i); gc.append(c); gb.append(b)
            for _ in range(r.integers(0, 4)):                     # jittered detections, some wrong class
                j = np.array(b) + r.normal(0, r.choice([2, 10, 30]), 4)
                j[2:] = np.maximum(j[2:], j[:2] + 1)
                di.append(i); dc.append(c if r.random() < 0.7 else r.integers(n_cls)); ds.append(r.random()); db.append(j)
        for _ in range(r.integers(0, 3)):                         # background false positives
            x0, y0 = r.uniform(0, 250, 2)
            di.append(i); dc.append(r.integers(n_cls)); ds.append(r.random() * 0.5); db.append([x0, y0, x0 + 30, y0 + 30])
    return [np.array(v) for v in (di, dc, ds, db, gi, gc, gb)], n_img, n_cls


def pycoco(di, dc, ds, db, gi, gc, gb, n_img, n_cls):
    gt = COCO()
    gt.dataset = {"images": [{"id": i, "width": 320, "height": 320} for i in range(n_img)],
                  "categories": [{"id": c} for c in range(n_cls)],
                  "annotations": [{"id": k + 1, "image_id": int(i), "category_id": int(c), "iscrowd": 0,
                                   "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                                   "area": float((b[2] - b[0]) * (b[3] - b[1]))}
                                  for k, (i, c, b) in enumerate(zip(gi, gc, gb))]}
    gt.createIndex()
    arr = np.column_stack([di, db[:, 0], db[:, 1], db[:, 2] - db[:, 0], db[:, 3] - db[:, 1], ds, dc])
    ev = COCOeval(gt, gt.loadRes(arr), "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[1], ev.stats[2]


if __name__ == "__main__":
    for seed in range(3):
        (di, dc, ds, db, gi, gc, gb), n_img, n_cls = random_case(seed)
        ours = coco_map(di, dc, ds, db, gi, gc, gb)
        ref = pycoco(di, dc, ds, db, gi, gc, gb, n_img, n_cls)
        print(f"seed {seed}: ours map {ours['map']:.6f} ap50 {ours['ap50']:.6f} ap75 {ours['ap75']:.6f} | "
              f"pycocotools {ref[0]:.6f} {ref[1]:.6f} {ref[2]:.6f}")
        assert np.allclose([ours["map"], ours["ap50"], ours["ap75"]], ref, atol=1e-6), "mismatch"
    print("coco_map matches pycocotools")

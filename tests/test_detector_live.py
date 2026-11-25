"""Live detector data and export pieces on synthetic inputs (run: python tests/test_detector_live.py; CPU, no data):
the ignore boxes BoxSet carries through its crop (D-046b) and the app graph's logit shift (D-048)."""
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transferner.export.detector_app import AppExport
from transferner.train.train_detector import BoxSet, build
from transferner.utils.config import load


def _photo(d: Path) -> str:
    r = np.random.default_rng(0)
    Image.fromarray(r.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(d / "a.jpg")
    return "a.jpg"


def test_boxset_ignore_boxes_follow_the_targets():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        p = _photo(root)
        frac = np.array([[0.1, 0.1, 0.6, 0.7]], np.float32)
        lab = np.array([5])
        ign = {p: np.array([[0.7, 0.2, 0.9, 0.4], [0.05, 0.8, 0.15, 0.95]], np.float32)}
        ev = BoxSet(root, [(p, frac, lab)], 448, False, ignore=ign)
        img, b, l, g, ig = ev.sample(0)
        assert img.shape == (3, 448, 448) and g == 5 and l.tolist() == [5]
        assert torch.allclose(b, torch.tensor(frac) * 448, atol=1e-3)
        assert torch.allclose(ig, torch.tensor(ign[p]) * 448, atol=1e-3)
        assert len(ev[0]) == 4 and torch.equal(ev[0][1], b)                  # the 4-tuple callers see no change
        tr = BoxSet(root, [(p, frac, lab)], 448, True, ignore=ign)
        for s in range(20):
            torch.manual_seed(s)
            img, b, l, g, ig = tr.sample(0)
            assert (l == 5).all() and len(b) == len(l) and ig.shape[1] == 4
            assert (ig >= 0).all() and (ig <= 448).all()
        no = BoxSet(root, [(p, frac, lab)], 448, False)
        assert no.sample(0)[4].shape == (0, 4)


def test_logit_shift_keeps_the_app_softmax():
    torch.manual_seed(0)
    G = 12
    m = build(G, load("detector")["model"]["backbone"], 224, pretrained=False, head="shared", image_aux=True,
              image_head=64).eval()
    x = torch.randn(3, 3, 224, 224)
    bias0 = m.head.image_cls[-1].bias.detach().clone()
    with torch.no_grad():
        ref = AppExport(m, regions=True).eval()(x)
        c = 40.0
        out = AppExport(m, regions=True, logit_shift=c).eval()(x)
    assert (out[0] >= 0).all() and (out[3] >= 0).all()
    assert torch.allclose(out[0], torch.relu(ref[0] + c), atol=1e-4)
    assert torch.allclose(out[3], torch.relu(ref[3] + c), atol=1e-4)
    assert torch.allclose(out[0].softmax(-1), ref[0].softmax(-1), atol=1e-6)
    assert torch.allclose(out[3].softmax(-1), ref[3].softmax(-1), atol=1e-6)
    assert torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])     # objectness and boxes untouched
    assert torch.equal(m.head.image_cls[-1].bias, bias0)                     # the trained model is not changed


if __name__ == "__main__":
    test_boxset_ignore_boxes_follow_the_targets()
    test_logit_shift_keeps_the_app_softmax()
    print("ok")

"""Step 09. Train the live detector (D-002, D-012): SSDLite on MobileNetV4-Conv-Medium, one class per genus.

Boxes come from manifests/boxes.parquet (data/pseudo_boxes, OWLv2), cleaned at load with
pseudo_boxes.drop_nested(). Each image keeps BioCLIP's observation-grouped split, so detector
train/val/test never share an observation. Train on train, pick the epoch on val genus AP50, and
report test once with the chosen checkpoint. Detection classes are the genera that have train
boxes (sorted); family scores are sums of member-genus probabilities (configs/detector.yaml:
heads.family = rollup). Specimens never reach here (D-020: pseudo_boxes skips them).

Architecture: torchvision's SSDLite recipe (6 feature maps at 320 px: 20, 10, 5, 3, 2, 1; anchors
[2, 3] at scales 0.2-0.95; 3:1 hard-negative mining; softmax with background) on timm's
mobilenetv4_conv_medium features at stride 16 (160 ch) and 32 (960 ch), plus four torchvision
SSDLite extra blocks. Postprocessing is ours, not torchvision's: SSD.postprocess_detections loops
over every class in Python, which is unusable at 1,362 classes. Ours takes the top-k (anchor,
class) pairs, then class-wise NMS.

Field augmentation (train.aug = field): SSD photometric distort, zoom-out (small subjects at a
distance), IoU crop, flip, then on the 320 px frame: downscale-upscale, motion blur and JPEG,
which a walking phone's video frames have and observation photos lack.

Metrics on val/test are against held-out OWLv2 pseudo-boxes on observation photos: box AP measures
agreement with the labeller plus the genus, not field accuracy (that is the field video set,
D-018). Reported: genus mAP@[.5:.95], AP50, AP75; class-agnostic AP50 ("found the plant"); and
image-level genus top-1/top-5 and family top-1 from the most confident anchor.

  python -m transferner.train.train_detector --name mnv4_ssdlite              # full run (queue it)
  python -m transferner.train.train_detector --name smoke --limit 2048 --epochs 1   # smoke test
Writes artifacts/detector/<name>/{best.pt,last.pt,labels.json}, results/runs/<run>/metrics.csv and a row in
results/tables/detector.csv."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssdlite import SSDLiteHead, _extra_block, _normal_init
from torchvision.ops import batched_nms, box_iou, nms
from torchvision.transforms import v2

from ..data.pseudo_boxes import drop_nested
from ..eval.metrics import coco_map
from ..utils.config import data_root, load
from ..utils.csvlog import RunLog


# ----------------------------------------------------------------------------------------- model
class Backbone(nn.Module):
    """timm MNv4 features at stride 16 and 32, then four SSDLite extra blocks (10 -> 5 -> 3 -> 2 -> 1)."""

    def __init__(self, name: str, pretrained: bool, norm):
        super().__init__()
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(3, 4))
        c4, c5 = self.body.feature_info.channels()
        widths, blocks, c = [512, 256, 256, 128], [], c5
        for w in widths:
            blocks.append(_extra_block(c, w, norm))
            c = w
        self.extra = nn.ModuleList(blocks)
        _normal_init(self.extra)
        self.out_channels = [c4, c5, *widths]
        self.mean = self.body.pretrained_cfg["mean"]
        self.std = self.body.pretrained_cfg["std"]

    def forward(self, x):
        feats = list(self.body(x))
        for b in self.extra:
            feats.append(b(feats[-1]))
        return OrderedDict((str(i), f) for i, f in enumerate(feats))


class SharedGenusHead(nn.Module):
    """SSDLite head with the genus classifier factorised out (D-033). torchvision's SSDLite head has a
    separate classifier per (level, anchor shape): 6 x 6 = 36 weight vectors per genus, each fed a sliver
    of ~70 images per genus, and the large boxes that dominate here are classified from the randomly
    initialised extra blocks. Here, per level, a depthwise 3x3 + 1x1 gives the background logit per anchor
    (whether an anchor fits a box depends on its shape) and a depthwise 3x3 + 1x1 gives a shared-width
    embedding per location; ONE 1x1 genus classifier is shared by all levels and anchors. Output layout
    and loss are SSD's: (B, A, 1 + G) logits with the location's genus logits repeated over its anchors.
    Background bias starts at P(background) = 0.99 (focal-loss prior), so early training is not swamped
    by the 3:1 negatives."""

    def __init__(self, in_channels: list[int], num_anchors: list[int], n_genera: int, norm, embed: int = 256,
                 image_aux: bool = False, image_head: int = 0):
        super().__init__()
        from torchvision.models.detection.ssdlite import SSDLiteRegressionHead
        from torchvision.ops.misc import Conv2dNormActivation
        dw = lambda c: Conv2dNormActivation(c, c, kernel_size=3, groups=c, norm_layer=norm,
                                            activation_layer=nn.ReLU6)
        self.num_anchors = num_anchors
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm)
        self.bg = nn.ModuleList(nn.Sequential(dw(c), nn.Conv2d(c, a, 1)) for c, a in zip(in_channels, num_anchors))
        self.embed = nn.ModuleList(
            nn.Sequential(dw(c), Conv2dNormActivation(c, embed, kernel_size=1, norm_layer=norm,
                                                      activation_layer=nn.ReLU6)) for c in in_channels)
        self.genus = nn.Conv2d(embed, n_genera, 1)
        # Image-level branch (--aux-image): global-pooled stride-32 features -> same embedding width -> the
        # SAME genus classifier, trained on the photo's genus. Every photo supervises the classifier every
        # step, and it is a cheap whole-frame genus readout (pool + 1x1), NPU-friendly.
        self.image_embed = nn.Sequential(nn.AdaptiveAvgPool2d(1), Conv2dNormActivation(
            in_channels[1], embed, kernel_size=1, norm_layer=None, activation_layer=nn.ReLU6)) if image_aux and not image_head else None
        # Dedicated whole-frame classifier (train_live_head.py, D-036): MobileNetV4's own classifier head
        # (pool -> 1x1 960 -> image_head, BN, ReLU -> classifier), the live label of the app. Replaces the
        # image branch above when set.
        self.image_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels[1], image_head, 1, bias=False), nn.BatchNorm2d(image_head),
            nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Conv2d(image_head, n_genera, 1)) if image_head else None
        _normal_init(self.bg)
        _normal_init(self.embed)
        nn.init.normal_(self.genus.weight, std=0.01)
        nn.init.zeros_(self.genus.bias)
        for s in self.bg:
            nn.init.constant_(s[-1].bias, math.log(n_genera) + math.log(99.0))

    def forward(self, feats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = []
        for f, bg, em, a in zip(feats, self.bg, self.embed, self.num_anchors):
            b = bg(f)                                                    # (B, a, H, W)
            g = self.genus(em(f))                                        # (B, G, H, W)
            B, G, H, W = g.shape
            b = b.permute(0, 2, 3, 1).reshape(B, H * W, a, 1)
            g = g.permute(0, 2, 3, 1).reshape(B, H * W, 1, G).expand(B, H * W, a, G)
            logits.append(torch.cat([b, g], -1).reshape(B, H * W * a, G + 1))
        out = {"bbox_regression": self.regression_head(feats), "cls_logits": torch.cat(logits, 1)}
        if self.image_embed is not None or self.image_cls is not None:
            out["image_logits"] = self.image_logits(feats)
        return out

    def image_logits(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """(B, G) whole-frame genus logits from the stride-32 features."""
        if self.image_cls is not None:
            return self.image_cls(feats[1]).flatten(1)
        return self.genus(self.image_embed(feats[1])).flatten(1)


def build(n_genera: int, backbone: str, size: int, pretrained: bool = True, head: str = "ssdlite",
          image_aux: bool = False, image_head: int = 0) -> SSD:
    norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    body = Backbone(backbone, pretrained, norm)
    anchors = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    if head == "shared":
        h = SharedGenusHead(body.out_channels, anchors.num_anchors_per_location(), n_genera, norm, image_aux=image_aux,
                            image_head=image_head)
    else:
        h = SSDLiteHead(body.out_channels, anchors.num_anchors_per_location(), n_genera + 1, norm)
    return SSD(body, anchors, (size, size), n_genera + 1, image_mean=list(body.mean), image_std=list(body.std),
               head=h)


class ShiftAddDepthwise(nn.Conv2d):
    """Depthwise conv as a sum of k*k shifted, per-channel-scaled views. Same parameters and
    state_dict as nn.Conv2d (export and QAT see a plain conv). Under torch.compile Inductor fuses
    the taps into one Triton kernel per direction. On this ROCm stack MIOpen runs depthwise convs
    through per-group Winograd or naive kernels (17 ms fwd+bwd for 48 ch at 80x80, batch 64,
    about 100x the memory-bandwidth bound), and they took ~45% of each training step."""

    def forward(self, x):
        k, s, p = self.kernel_size[0], self.stride[0], self.padding[0]
        _, c, h, w = x.shape
        ho, wo = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        xp = F.pad(x, (p, p, p, p))
        wt = self.weight.to(x.dtype).view(c, k * k)
        y = None
        for i in range(k):
            for j in range(k):
                t = xp[:, :, i:i + s * (ho - 1) + 1:s, j:j + s * (wo - 1) + 1:s] * wt[:, i * k + j].view(1, c, 1, 1)
                y = t if y is None else y + t
        if self.bias is not None:
            y = y + self.bias.to(x.dtype).view(1, c, 1, 1)
        return y


def use_shift_add_depthwise(model: nn.Module) -> int:
    """Swap the class of every plain depthwise nn.Conv2d (square kernel, zero symmetric padding,
    no dilation) to ShiftAddDepthwise in place; returns how many were swapped."""
    n = 0
    for m in model.modules():
        if (type(m) is nn.Conv2d and m.groups == m.in_channels == m.out_channels and m.groups > 1
                and m.kernel_size[0] == m.kernel_size[1] and m.stride[0] == m.stride[1]
                and isinstance(m.padding, tuple) and m.padding[0] == m.padding[1]
                and m.dilation == (1, 1) and m.padding_mode == "zeros"):
            m.__class__ = ShiftAddDepthwise
            n += 1
    return n


def python_headers_for_triton() -> None:
    """Triton's launcher compiles a C helper against Python.h. .venv runs on the system Python 3.12,
    whose headers (python3.12-dev) are not installed, so borrow a uv-managed CPython 3.12's headers
    (`uv python install 3.12`, no sudo, nothing system-wide). Same minor version, same C API."""
    import glob
    import os
    import sys
    import sysconfig
    if os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")):
        return
    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    hits = sorted(glob.glob(os.path.expanduser(f"~/.local/share/uv/python/cpython-{v}.*/include/python{v}")))
    if not hits:
        raise RuntimeError(f"torch.compile needs Python.h: run `uv python install {v}` or pass --no-compile")
    os.environ["C_INCLUDE_PATH"] = os.pathsep.join(filter(None, [hits[-1], os.environ.get("C_INCLUDE_PATH")]))


class Net(nn.Module):
    """Backbone + head as one module, the unit that is torch.compile'd (loss and postprocessing stay eager)."""

    def __init__(self, ssd: SSD):
        super().__init__()
        self.backbone, self.head = ssd.backbone, ssd.head

    def forward(self, x):
        return self.head(list(self.backbone(x).values()))


def forward_raw(net: nn.Module, x: torch.Tensor):
    """Normalised (B, 3, S, S) input -> head outputs {'bbox_regression', 'cls_logits'}."""
    return net(x)


def anchors_for(model: SSD, size: int, dev) -> torch.Tensor:
    with torch.no_grad():
        was = model.training
        x0 = torch.zeros(1, 3, size, size, device=dev)
        feats = list(model.backbone.eval()(x0).values())
        model.backbone.train(was)
        return model.anchor_generator(ImageList(x0, [(size, size)]), feats)[0]


def ssd_loss_batched(model: SSD, out, anchors0: torch.Tensor, targets, iou_thresh: float = 0.5):
    """torchvision's SSD matching + loss (SSDMatcher at 0.5, smooth-L1 on encoded boxes, softmax CE with
    3:1 hard-negative mining, both normalised by the foreground count) for one shared anchor set, vectorised
    over the batch. torchvision loops over images and launches ~700 small kernels per step of 64 (87 ms vs
    30 ms fwd+bwd); losses and gradients are identical to ssd_loss (torchvision's compute_loss) on random
    batches, images without boxes included."""
    reg, logits = out["bbox_regression"].float(), out["cls_logits"].float()
    B, A, C = logits.shape
    G = max(1, max(len(t["boxes"]) for t in targets))
    gt = anchors0.new_zeros(B, G, 4)
    lab = torch.zeros(B, G, dtype=torch.int64, device=anchors0.device)
    valid = torch.zeros(B, G, dtype=torch.bool, device=anchors0.device)
    for i, t in enumerate(targets):
        n = len(t["boxes"])
        gt[i, :n], lab[i, :n], valid[i, :n] = t["boxes"], t["labels"], True
    iou = box_iou(gt.view(-1, 4), anchors0).view(B, G, A).masked_fill(~valid[..., None], -1.0)
    best_v, matched = iou.max(1)                                          # best GT per anchor
    matched = matched.masked_fill(best_v < iou_thresh, -1)
    best_a = iou.argmax(2)                                                # every GT keeps its best anchor
    bi, gi = valid.nonzero(as_tuple=True)
    matched[bi, best_a[bi, gi]] = gi
    fg = matched >= 0
    n_fg = int(fg.sum())
    m = matched.clamp(min=0)
    fb, fa = fg.nonzero(as_tuple=True)
    target = model.box_coder.encode_single(gt[fb, m[fb, fa]], anchors0[fa])
    bbox = F.smooth_l1_loss(reg[fb, fa], target, reduction="sum")
    cls_t = torch.where(fg, lab.gather(1, m), 0)
    cls = F.cross_entropy(logits.view(-1, C), cls_t.view(-1), reduction="none").view(B, A)
    n_neg = model.neg_to_pos_ratio * fg.sum(1, keepdim=True)
    neg = cls.detach().masked_fill(fg, -float("inf"))
    _, idx = neg.sort(1, descending=True)
    hard = idx.sort(1)[1] < n_neg
    N = max(1, n_fg)
    return bbox / N, (cls[fg].sum() + cls[hard].sum()) / N


def ssd_loss(model: SSD, out, anchors, targets):
    matched = []
    for a, t in zip(anchors, targets):
        if t["boxes"].numel() == 0:
            matched.append(torch.full((a.size(0),), -1, dtype=torch.int64, device=a.device))
        else:
            matched.append(model.proposal_matcher(box_iou(t["boxes"], a)))
    out = {k: v.float() for k, v in out.items()}
    losses = model.compute_loss(targets, out, anchors, matched)
    return losses["bbox_regression"], losses["classification"]


@torch.no_grad()
def detect(box_coder, out, anchors0: torch.Tensor, size: int, topk: int = 300, max_det: int = 100,
           thresh: float = 0.01, iou: float = 0.55):
    """Per image: class-aware detections (top-k anchor x genus pairs -> class-wise NMS), class-agnostic
    detections scored by objectness 1 - P(background) (NMS), and the genus distribution (normalised over
    genera) at the anchor with the highest objectness. The best single genus probability would be a poor
    objectness score: spread over 1,362 genera it mostly measures genus confusion, not plant vs background."""
    full = out["cls_logits"].float().softmax(-1)
    probs = full[..., 1:]                                                # (B, A, G), background dropped
    B, A, G = probs.shape
    boxes = box_coder.decode_single(out["bbox_regression"].float().reshape(-1, 4),
                                          anchors0.repeat(B, 1)).reshape(B, A, 4).clamp(0, size)
    best = 1.0 - full[..., 0]                                            # objectness (B, A)
    top_anchor = best.argmax(1)
    ar = torch.arange(B, device=probs.device)
    dist = probs[ar, top_anchor] / best[ar, top_anchor, None].clamp(min=1e-12)   # (B, G)
    res = []
    for i in range(B):
        s, idx = probs[i].flatten().topk(topk)
        a, g = idx // G, idx % G
        m = s >= thresh
        a, g, s = a[m], g[m], s[m]
        k = batched_nms(boxes[i, a], s, g, iou)[:max_det]
        sa, aa = best[i].topk(topk)
        ka = nms(boxes[i, aa], sa, iou)[:max_det]
        res.append({"box": boxes[i, a[k]].cpu().numpy(), "genus": g[k].cpu().numpy(), "score": s[k].cpu().numpy(),
                    "abox": boxes[i, aa[ka]].cpu().numpy(), "ascore": sa[ka].cpu().numpy()})
    return res, dist


# ------------------------------------------------------------------------------------------ data
def motion_kernel(k: int, angle: float) -> torch.Tensor:
    ker = torch.zeros(k, k)
    c = (k - 1) / 2
    t = torch.linspace(-c, c, 4 * k)
    xs = (c + t * math.cos(angle)).round().long().clamp(0, k - 1)
    ys = (c + t * math.sin(angle)).round().long().clamp(0, k - 1)
    ker[ys, xs] = 1.0
    return ker / ker.sum()


def low_light(img: torch.Tensor) -> torch.Tensor:
    """uint8 (3, S, S) -> a phone's low-light frame: the scene at 15-60% of the light, sensor noise, then the
    auto-exposure gain back up (amplifying the noise), a slight warm or cool cast, and lifted blacks."""
    f = float(torch.empty(()).uniform_(0.15, 0.6))
    x = img.float() * f
    x = x + torch.randn_like(x) * float(torch.empty(()).uniform_(1.0, 4.0))
    x = x / f * float(torch.empty(()).uniform_(0.6, 1.0))
    x = x * torch.empty(3, 1, 1).uniform_(0.9, 1.1) + float(torch.empty(()).uniform_(0, 12))
    return x.round().clamp(0, 255).byte()


def field_degrade(img: torch.Tensor, size: int, low_light_p: float = 0.0) -> torch.Tensor:
    """uint8 (3, S, S) -> walking-video artefacts: low light (when low_light_p), downscale-upscale, motion
    blur, JPEG."""
    if low_light_p and torch.rand(()) < low_light_p:
        img = low_light(img)
    if torch.rand(()) < 0.25:
        f = float(torch.empty(()).uniform_(0.3, 0.7))
        x = img[None].float()
        x = F.interpolate(x, size=max(8, int(size * f)), mode="bilinear", antialias=True, align_corners=False)
        img = F.interpolate(x, size=size, mode="bilinear", align_corners=False)[0].round().clamp(0, 255).byte()
    if torch.rand(()) < 0.25:
        k = int(torch.randint(1, 5, ())) * 2 + 1                         # 3, 5, 7, 9
        ker = motion_kernel(k, float(torch.rand(())) * math.pi).expand(3, 1, k, k)
        img = F.conv2d(img[None].float(), ker, padding=k // 2, groups=3)[0].round().clamp(0, 255).byte()
    if torch.rand(()) < 0.3:
        img = v2.JPEG((30, 90))(img)
    return img


class BoxSet(Dataset):
    def __init__(self, root, items: list, size: int, train: bool, low_light_p: float = 0.0):
        self.root, self.items, self.size, self.train, self.low_light_p = root, items, size, train, low_light_p
        if train:
            self.tf = v2.Compose([
                v2.RandomPhotometricDistort(p=0.5),
                v2.RandomZoomOut(fill={tv_tensors.Image: (124, 116, 104), "others": 0}, side_range=(1.0, 3.0), p=0.3),
                v2.RandomIoUCrop(),
                v2.RandomHorizontalFlip(),
                v2.Resize((size, size), antialias=True),
                v2.SanitizeBoundingBoxes(min_size=2),
            ])
        else:
            self.tf = v2.Resize((size, size), antialias=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, frac, lab = self.items[i]
        img = decode_image(str(self.root / path), mode=ImageReadMode.RGB)
        h, w = img.shape[-2:]
        boxes = tv_tensors.BoundingBoxes(torch.tensor(frac) * torch.tensor([w, h, w, h]), format="XYXY",
                                         canvas_size=(h, w))
        img, tgt = self.tf(tv_tensors.Image(img), {"boxes": boxes, "labels": torch.as_tensor(lab)})
        img = img.as_subclass(torch.Tensor)
        if self.train:
            img = field_degrade(img, self.size, self.low_light_p)
        return img, tgt["boxes"].as_subclass(torch.Tensor).float(), tgt["labels"].long(), int(lab[0])


def collate(batch):
    """-> uint8 images (B, 3, S, S), per-image boxes and labels (1 + genus), image genus label (B,) (1 + genus)."""
    imgs, boxes, labels, genus = zip(*batch)
    return torch.stack(imgs), list(boxes), list(labels), torch.tensor(genus)


def load_items(root, boxes_path: str, fallback: str, limit: int, seed: int):
    """-> genera (sorted, train), family_of (per genus), {split: [(path, frac boxes, labels)]}, image genus per split."""
    bx = pd.read_parquet(root / boxes_path)
    if fallback == "drop":
        bx = bx[~bx.fallback]
    bx = drop_nested(bx)
    img = pd.read_parquet(root / "manifests/images.parquet", columns=["uuid", "path"])
    bx = bx.merge(img, on="uuid")
    genera = sorted(bx.loc[bx.split == "train", "genus"].unique())
    gi = {g: i for i, g in enumerate(genera)}
    fam = bx.drop_duplicates("genus").set_index("genus").family
    families = sorted(fam[genera].unique())
    fi = {f: i for i, f in enumerate(families)}
    family_of = np.array([fi[fam[g]] for g in genera])
    unseen = ~bx.genus.isin(gi)
    if unseen.any():
        print(f"dropping {bx[unseen].uuid.nunique():,} val/test images of {bx[unseen].genus.nunique()} genera "
              f"with no train boxes", flush=True)
    bx = bx[~unseen].copy()
    bx["label"] = bx.genus.map(gi) + 1                                     # 0 = background
    splits = {}
    for split, d in bx.groupby("split", sort=False):
        d = d.sort_values("uuid", kind="stable")
        u, start = np.unique(d.uuid.to_numpy(), return_index=True)
        if limit:
            keep = np.random.default_rng(seed).permutation(len(u))[:limit if split == "train" else limit // 4]
            sel = np.zeros(len(u), bool)
            sel[keep] = True
        else:
            sel = np.ones(len(u), bool)
        xy = d[["x0", "y0", "x1", "y1"]].to_numpy(np.float32)
        lab = d.label.to_numpy()
        path = d.path.to_numpy()
        end = np.append(start[1:], len(d))
        splits[split] = [(path[s], xy[s:e], lab[s:e]) for s, e, k in zip(start, end, sel) if k]
    return genera, family_of, families, splits


# ------------------------------------------------------------------------------------------ eval
def evaluate(model, net, loader, anchors0, size, family_of, dev, mean, std, channels_last, amp) -> dict:
    """Score the (compiled) PyTorch model; see evaluate_with."""
    model.eval()

    def run(x):
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with amp:
            return forward_raw(net, x)
    try:
        return evaluate_with(run, model.box_coder, loader, anchors0, size, family_of, dev, mean, std)
    finally:
        model.train()


@torch.no_grad()
def evaluate_with(run, box_coder, loader, anchors0, size, family_of, dev, mean, std) -> dict:
    """run(x) -> {'bbox_regression', 'cls_logits'} for normalised (B, 3, S, S) input on dev: the PyTorch
    model, a PT2E-quantized graph or a LiteRT interpreter, all scored by the same postprocessing and metrics."""
    dets = {k: [] for k in ("img", "cls", "score", "box")}
    adets = {k: [] for k in ("img", "score", "box")}
    gts = {k: [] for k in ("img", "cls", "box")}
    top1 = top5 = fam1 = n = 0
    itop1 = itop5 = ifam1 = 0
    fam_t = torch.as_tensor(family_of, device=dev)
    n_fam = int(family_of.max()) + 1
    for imgs, boxes, labels, genus in loader:
        x = ((imgs.to(dev, non_blocking=True).float() / 255) - mean) / std
        out = run(x)
        res, dist = detect(box_coder, out, anchors0, size)
        y = genus.to(dev) - 1                                                  # image genus
        if "image_logits" in out:
            ip = out["image_logits"].float().softmax(-1)
            itop1 += int((ip.argmax(1) == y).sum())
            itop5 += int((ip.topk(5, 1).indices == y[:, None]).any(1).sum())
            ifd = torch.zeros(len(y), n_fam, device=dev).index_add_(1, fam_t, ip)
            ifam1 += int((ifd.argmax(1) == fam_t[y]).sum())
        top1 += int((dist.argmax(1) == y).sum())
        top5 += int((dist.topk(5, 1).indices == y[:, None]).any(1).sum())
        fd = torch.zeros(len(y), n_fam, device=dev).index_add_(1, fam_t, dist)
        fam1 += int((fd.argmax(1) == fam_t[y]).sum())
        for i, r in enumerate(res):
            k = n + i
            dets["img"].append(np.full(len(r["score"]), k)); dets["cls"].append(r["genus"])
            dets["score"].append(r["score"]); dets["box"].append(r["box"])
            adets["img"].append(np.full(len(r["ascore"]), k)); adets["score"].append(r["ascore"])
            adets["box"].append(r["abox"])
            gts["img"].append(np.full(len(labels[i]), k)); gts["cls"].append(labels[i].numpy() - 1)
            gts["box"].append(boxes[i].numpy())
        n += len(res)
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    dets, adets, gts = cat(dets), cat(adets), cat(gts)
    m = coco_map(dets["img"], dets["cls"], dets["score"], dets["box"], gts["img"], gts["cls"], gts["box"])
    ma = coco_map(adets["img"], np.zeros(len(adets["img"]), int), adets["score"], adets["box"],
                  gts["img"], np.zeros(len(gts["img"]), int), gts["box"])
    res = {"map": m["map"], "ap50": m["ap50"], "ap75": m["ap75"], "agnostic_ap50": ma["ap50"],
           "agnostic_map": ma["map"], "genus_top1": top1 / n, "genus_top5": top5 / n, "family_top1": fam1 / n}
    if itop1 or itop5 or ifam1:
        res.update(image_genus_top1=itop1 / n, image_genus_top5=itop5 / n, image_family_top1=ifam1 / n)
    return {**res, "n_images": n}


# ----------------------------------------------------------------------------------------- train
def main() -> None:
    cfg = load("detector")
    tc = cfg["train"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="mnv4_ssdlite")
    ap.add_argument("--boxes", default="manifests/boxes.parquet", help="relative to TRANSFERNER_DATA")
    ap.add_argument("--epochs", type=int, default=tc["epochs"])
    ap.add_argument("--batch", type=int, default=tc["batch"])
    ap.add_argument("--lr", type=float, default=tc["lr"])
    ap.add_argument("--wd", type=float, default=tc.get("weight_decay", 0.05))
    ap.add_argument("--warmup", type=int, default=1000, help="linear warmup steps, then cosine to 0")
    ap.add_argument("--sampler-power", type=float, default=tc.get("sampler_power", 0.5),
                    help="image weight = (train images of its genus)^-p; 0 = uniform, 1 = fully class-balanced")
    ap.add_argument("--fallback", choices=["keep", "drop"], default="keep",
                    help="images where OWLv2 found nothing carry one full-frame box (mostly frame-filling subjects)")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: train images (val/test get a quarter)")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--head", choices=["ssdlite", "shared"], default=cfg["model"].get("head", "ssdlite"),
                    help="ssdlite: torchvision's per-(level, anchor) classifiers; shared: one genus classifier (D-033)")
    ap.add_argument("--backbone-lr-mult", type=float, default=tc.get("backbone_lr_mult", 1.0),
                    help="lr of the pretrained MNv4 layers = lr x this (new extras and head: lr)")
    ap.add_argument("--aux-image", type=float, default=tc.get("aux_image", 0.0),
                    help="weight of the image-level genus loss through the shared classifier (needs --head shared)")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="A/B screens: run only the first N epochs of the full schedule; no test pass, no summary row")
    ap.add_argument("--dw-impl", choices=["shift", "miopen"], default="shift",
                    help="depthwise convs as fused shift-and-add (needs --compile to be fast) or MIOpen's kernels")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke tests: stop each epoch after N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", default="", help="checkpoint (relative to TRANSFERNER_DATA): score val and test, no training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    root, dev = data_root(), "cuda"
    size = cfg["model"]["input"]
    genera, family_of, families, splits = load_items(root, args.boxes, args.fallback, args.limit, args.seed)
    print({s: len(v) for s, v in splits.items()}, f"{len(genera)} genera, {len(families)} families", flush=True)
    out_dir = root / "artifacts/detector" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.json").write_text(json.dumps({"genera": genera, "families": families,
                                                     "family_of": family_of.tolist()}))

    train = splits["train"]
    n_g = np.bincount([lab[0] for _, _, lab in train], minlength=len(genera) + 1)
    w = np.array([n_g[lab[0]] for _, _, lab in train], float) ** -args.sampler_power
    sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    mk = lambda items, train_, **kw: DataLoader(BoxSet(root, items, size, train_), batch_size=args.batch,
                                                num_workers=args.workers, collate_fn=collate, pin_memory=True,
                                                persistent_workers=True, prefetch_factor=4, **kw)
    dl_train = mk(train, True, sampler=sampler, drop_last=True)
    dl_val = mk(splits["val"], False)

    if args.aux_image and args.head != "shared":
        ap.error("--aux-image needs --head shared")
    model = build(len(genera), cfg["model"]["backbone"], size, pretrained=not args.no_pretrained,
                  head=args.head, image_aux=args.aux_image > 0).to(dev)
    n_dw = use_shift_add_depthwise(model) if args.dw_impl == "shift" else 0
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    net = Net(model)
    if args.compile:
        python_headers_for_triton()
        net = torch.compile(net, dynamic=False)
    n_params = sum(p.numel() for p in model.parameters())
    mean = torch.tensor(model.backbone.mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(model.backbone.std, device=dev).view(1, 3, 1, 1)
    groups = {}                                  # (pretrained backbone?, decay?) -> params
    for n, p in model.named_parameters():
        groups.setdefault((n.startswith("backbone.body."), p.ndim > 1), []).append(p)
    opt = torch.optim.AdamW([{"params": ps, "weight_decay": args.wd if dec else 0.0,
                              "lr": args.lr * (args.backbone_lr_mult if pre else 1.0)}
                             for (pre, dec), ps in groups.items()], lr=args.lr)
    steps_per_epoch = min(len(dl_train), args.max_steps or len(dl_train))
    total = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler()
    amp = torch.autocast("cuda", dtype=torch.float16)
    anchors0 = anchors_for(model, size, dev)
    model.train()
    print(f"{n_params / 1e6:.1f}M params, {len(anchors0)} anchors, {steps_per_epoch} steps/epoch, "
          f"{n_dw} depthwise convs as shift-add, compile {args.compile}", flush=True)

    if args.eval_only:
        ck = torch.load(root / args.eval_only, map_location=dev)
        model.load_state_dict(ck["model"])
        for split in ("val", "test"):
            m = evaluate(model, net, mk(splits[split], False), anchors0, size, family_of, dev, mean, std,
                         args.channels_last, amp)
            print(f"{args.eval_only} (epoch {ck['epoch']}) {split}: " +
                  " ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)
        return

    log = RunLog(f"detector_{args.name}")
    best, best_ep, step = -1.0, -1, 0
    t_start = time.time()
    for ep in range(args.epochs):
        if args.stop_after and ep >= args.stop_after:
            break
        t0, seen, sums = time.time(), 0, torch.zeros(2, device=dev)
        for it, (imgs, boxes, labels, genus) in enumerate(dl_train):
            if it >= steps_per_epoch:
                break
            x = ((imgs.to(dev, non_blocking=True).float() / 255) - mean) / std
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            targets = [{"boxes": b.to(dev, non_blocking=True), "labels": l.to(dev, non_blocking=True)}
                       for b, l in zip(boxes, labels)]
            with amp:
                out = forward_raw(net, x)
            l_box, l_cls = ssd_loss_batched(model, out, anchors0, targets)
            loss = l_box + l_cls
            if "image_logits" in out:
                l_img = F.cross_entropy(out["image_logits"].float(), genus.to(dev, non_blocking=True) - 1)
                loss = loss + args.aux_image * l_img
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            seen += len(imgs)
            sums += torch.stack([l_box.detach(), l_cls.detach()])
            if it % 200 == 0:
                extra = f" img {l_img.item():.3f}" if "image_logits" in out else ""
                print(f"ep {ep} it {it}/{steps_per_epoch} box {l_box.item():.3f} cls {l_cls.item():.3f}{extra} "
                      f"lr {max(sched.get_last_lr()):.2e} {seen / (time.time() - t0):.0f} img/s", flush=True)
        n_it = it + (it < steps_per_epoch)
        sums = (sums / max(1, n_it)).tolist()
        log.log(ep, "train", loss_box=sums[0], loss_cls=sums[1], img_s=seen / (time.time() - t0))
        torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args)}, out_dir / "last.pt")
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1 or ep == args.stop_after - 1:
            m = evaluate(model, net, dl_val, anchors0, size, family_of, dev, mean, std, args.channels_last, amp)
            log.log(ep, "val", **m)
            print(f"ep {ep} val " + " ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "n_images") +
                  f" ({time.time() - t0:.0f} s)", flush=True)
            if m["ap50"] > best:
                best, best_ep = m["ap50"], ep
                torch.save({"model": model.state_dict(), "epoch": ep, "args": vars(args), "val": m},
                           out_dir / "best.pt")

    if args.stop_after:                          # A/B screen: selection data only, test stays untouched
        print(f"stopped after {args.stop_after} epochs; best val ap50 {best:.4f} at epoch {best_ep}", flush=True)
        log.close()
        return
    ck = torch.load(out_dir / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    val = ck["val"]
    test = evaluate(model, net, mk(splits["test"], False), anchors0, size, family_of, dev, mean, std,
                    args.channels_last, amp)
    print(f"best epoch {best_ep}: test " + " ".join(f"{k} {v:.4f}" for k, v in test.items()), flush=True)
    log.log(best_ep, "test", **test)
    log.summary("detector", name=args.name, backbone=cfg["model"]["backbone"], input=size, n_genera=len(genera),
                n_train=len(train), n_val=len(splits["val"]), n_test=len(splits["test"]), epochs=args.epochs,
                best_epoch=best_ep, batch=args.batch, lr=args.lr, wd=args.wd, sampler_power=args.sampler_power,
                fallback=args.fallback, params_m=round(n_params / 1e6, 2), limit=args.limit,
                hours=round((time.time() - t_start) / 3600, 2),
                **{f"val_{k}": round(v, 4) for k, v in val.items() if k != "n_images"},
                **{f"test_{k}": round(v, 4) for k, v in test.items() if k != "n_images"})
    log.close()


if __name__ == "__main__":
    main()

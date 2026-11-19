"""Efficient BioCLIP image-tower variants for the efficiency sweep (docs/ARCHITECTURE.md).

One forward function over the stock open_clip ViT-H/14 weights, with four independent knobs:

- res          input resolution (positional embedding resized bicubically); 224 = stock
- tome_r       token merging: r tokens merged per block by bipartite soft matching on the
               attention keys, with size-weighted averaging and proportional attention
               (Bolya et al., "Token Merging: Your ViT but Faster", ICLR 2023). Implemented
               here from the paper; the class token is never merged.
- drop         skip the last `drop` transformer blocks (ln_post + proj still apply; the
               probe must be retrained on the new features)
- quant        *simulated* quantization of the 4 linear layers in every block (qkv, out, fc1,
               fc2): weights fake-quantized in place (symmetric, per output channel or per
               group), activations fake-quantized at each linear input, either dynamically
               per token (XNNPACK-style 8da8w / 8da4w) or statically per tensor from a
               calibration pass (NPU-style W8A8), optionally after SmoothQuant
               (Xiao et al. 2023) folded into LayerNorm / V-projection. Layers named in
               fp16_layers stay in floating point ("partial" W8A8, D-011). Patch embedding,
               LayerNorms, softmax, GELU and the final projection stay FP16.

Compute runs in FP16 autocast (the on-device activation precision), so W8A16 / W4A16 here
are exact up to FP16 accumulation; W8A8 is a faithful simulation of the integer math for
the quantized linears. Layer names: b{i}.qkv, b{i}.out, b{i}.fc1, b{i}.fc2."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchvision import transforms as Tv

LINEARS = ("qkv", "out", "fc1", "fc2")


@dataclass
class Variant:
    res: int = 224
    tome_r: int = 0
    drop: int = 0
    w_bits: int = 16              # 16 = float weights
    w_group: int = 0              # 0 = per output channel, else group size along the input dim
    a_bits: int = 16              # 16 = float activations
    a_mode: str = "dynamic"       # dynamic (per token) | static (per tensor, calibrated)
    a_pct: float = 99.99          # static: clip activations to this percentile range
    smooth: float = 0.0           # SmoothQuant alpha, 0 = off (static mode only)
    fp16_layers: frozenset = field(default_factory=frozenset)

    @property
    def quantized(self) -> bool:
        return self.w_bits < 16 or self.a_bits < 16


# ----------------------------------------------------------------------------- quantization
def fq_weight(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Symmetric round-to-nearest fake quantization, per output row or per input group."""
    out, inp = w.shape
    g = group if group and inp % group == 0 else inp
    x = w.float().reshape(out, inp // g, g)
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return ((x / scale).round().clamp(-qmax, qmax) * scale).reshape(out, inp).to(w.dtype)


def fq_act_dynamic(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Asymmetric per-token fake quantization (scale/zero-point from each token's min/max)."""
    xf = x.float()
    lo, hi = xf.amin(-1, keepdim=True).clamp(max=0), xf.amax(-1, keepdim=True).clamp(min=0)
    qmax = 2 ** bits - 1
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    zp = (-lo / scale).round()
    return (((xf / scale).round() + zp).clamp(0, qmax) - zp).mul(scale).to(x.dtype)


def fq_act_static(x: torch.Tensor, lo: float, hi: float, bits: int) -> torch.Tensor:
    qmax = 2 ** bits - 1
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = max((hi - lo) / qmax, 1e-8)
    zp = round(-lo / scale)
    xf = x.float()
    return (((xf / scale).round() + zp).clamp(0, qmax) - zp).mul(scale).to(x.dtype)


class QState:
    """Per-layer activation quantization state, calibration statistics and SmoothQuant stats."""

    SAMPLES_PER_BATCH = 1 << 16

    def __init__(self, v: Variant):
        self.v = v
        self.active: set[str] | None = None       # None = every non-fp16 layer
        self.ranges: dict[str, tuple[float, float]] = {}
        self.mode = "run"                          # run | calib_range | calib_channel
        self._samples: dict[str, list[torch.Tensor]] = {}
        self._minmax: dict[str, list[float]] = {}
        self.chan_amax: dict[str, torch.Tensor] = {}
        self._gen: torch.Generator | None = None   # seeded: calibration ranges are reproducible

    def is_active(self, name: str) -> bool:
        if self.v.a_bits >= 16 or name in self.v.fp16_layers:
            return False
        return self.active is None or name in self.active

    def __call__(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "calib_channel":
            a = x.detach().float().abs().amax(dim=tuple(range(x.ndim - 1)))
            self.chan_amax[name] = torch.maximum(self.chan_amax[name], a) if name in self.chan_amax else a
            return x
        if self.mode == "calib_range":
            xf = x.detach().float().reshape(-1)
            if self._gen is None or self._gen.device != xf.device:
                self._gen = torch.Generator(device=xf.device).manual_seed(0)
            idx = torch.randint(0, xf.numel(), (self.SAMPLES_PER_BATCH,), device=xf.device, generator=self._gen)
            self._samples.setdefault(name, []).append(xf[idx].cpu())
            mm = self._minmax.setdefault(name, [math.inf, -math.inf])
            mm[0], mm[1] = min(mm[0], xf.min().item()), max(mm[1], xf.max().item())
            return x
        if not self.is_active(name):
            return x
        if self.v.a_mode == "dynamic":
            return fq_act_dynamic(x, self.v.a_bits)
        lo, hi = self.ranges[name]
        return fq_act_static(x, lo, hi, self.v.a_bits)

    def finish_range_calibration(self) -> None:
        p = self.v.a_pct
        for name, chunks in self._samples.items():
            if p >= 100:
                self.ranges[name] = tuple(self._minmax[name])
            else:
                s = torch.cat(chunks)
                lo, hi = torch.quantile(s, torch.tensor([(100 - p) / 100, p / 100])).tolist()
                self.ranges[name] = (lo, hi)
        self._samples.clear()


# ----------------------------------------------------------------------------- token merging
def bipartite_merge(metric: torch.Tensor, r: int, x: torch.Tensor, size: torch.Tensor):
    """Merge r tokens: alternate tokens into sets A (even) and B (odd); each A token proposes
    its most similar B token; the r most similar A tokens are averaged (size-weighted) into
    their B partners. Token 0 (class) is in A and protected. Returns (x, size)."""
    n = x.shape[1]
    r = min(r, (n - 1) // 2)
    if r <= 0:
        return x, size
    m = F.normalize(metric.float(), dim=-1)
    scores = m[:, ::2] @ m[:, 1::2].transpose(1, 2)            # (B, Na, Nb)
    scores[:, 0, :] = -math.inf                                  # never merge the class token
    best, best_idx = scores.max(-1)
    order = best.argsort(-1, descending=True)
    src = order[:, :r]
    keep = order[:, r:].sort(-1).values                          # class token stays first
    dst = best_idx.gather(1, src)
    d = x.shape[-1]
    xs, s = (x.float() * size), size
    xa, xb, sa, sb = xs[:, ::2], xs[:, 1::2], s[:, ::2], s[:, 1::2]
    xb = xb.scatter_add(1, dst[..., None].expand(-1, -1, d), xa.gather(1, src[..., None].expand(-1, -1, d)))
    sb = sb.scatter_add(1, dst[..., None], sa.gather(1, src[..., None]))
    xa, sa = xa.gather(1, keep[..., None].expand(-1, -1, d)), sa.gather(1, keep[..., None])
    s_new = torch.cat([sa, sb], 1)
    return (torch.cat([xa, xb], 1) / s_new).to(x.dtype), s_new


# ----------------------------------------------------------------------------- model
class EfficientVisual(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module, v: Variant):
        super().__init__()
        self.visual, self.v = visual, v
        self.blocks = list(visual.transformer.resblocks)[: len(visual.transformer.resblocks) - v.drop]
        self.q = QState(v)
        pos = visual.positional_embedding.detach()
        g0 = int(math.isqrt(pos.shape[0] - 1))
        g = v.res // visual.conv1.kernel_size[0]
        if g != g0:
            grid = pos[1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2).float()
            grid = F.interpolate(grid, size=(g, g), mode="bicubic", align_corners=False)
            pos = torch.cat([pos[:1], grid.permute(0, 2, 3, 1).reshape(g * g, -1).to(pos.dtype)])
        self.register_buffer("pos", pos, persistent=False)

    def names(self) -> list[str]:
        return [f"b{i}.{k}" for i in range(len(self.blocks)) for k in LINEARS]

    def weight_of(self, name: str) -> torch.nn.Parameter:
        i, k = name[1:].split(".")
        b = self.blocks[int(i)]
        return {"qkv": b.attn.in_proj_weight, "out": b.attn.out_proj.weight,
                "fc1": b.mlp.c_fc.weight, "fc2": b.mlp.c_proj.weight}[k]

    def _block(self, i, b, x, size):
        q = self.q
        a = b.attn
        bsz, n, c = x.shape
        qkv = F.linear(q(f"b{i}.qkv", b.ln_1(x)), a.in_proj_weight, a.in_proj_bias)
        qh, kh, vh = qkv.reshape(bsz, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)
        mask = None if size is None else size.log().to(qh.dtype)[:, None, None, :, 0]
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=a.scale)
        o = o.transpose(1, 2).reshape(bsz, n, c)
        x = x + b.ls_1(F.linear(q(f"b{i}.out", o), a.out_proj.weight, a.out_proj.bias))
        if self.v.tome_r:
            if size is None:
                size = torch.ones(bsz, n, 1, device=x.device, dtype=torch.float32)
            x, size = bipartite_merge(kh.mean(1), self.v.tome_r, x, size)
        h = F.linear(q(f"b{i}.fc1", b.ln_2(x)), b.mlp.c_fc.weight, b.mlp.c_fc.bias)
        h = F.linear(q(f"b{i}.fc2", b.mlp.gelu(h)), b.mlp.c_proj.weight, b.mlp.c_proj.bias)
        return x + b.ls_2(h), size

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        vis = self.visual
        x = vis.conv1(img).flatten(2).transpose(1, 2)
        cls = vis.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = vis.ln_pre(torch.cat([cls, x], 1) + self.pos.to(x.dtype))
        size = None
        for i, b in enumerate(self.blocks):
            x, size = self._block(i, b, x, size)
        x = vis.ln_post(x[:, 0])
        return x @ vis.proj

    # ------------------------------------------------------------------------- quant setup
    @torch.no_grad()
    def quantize_weights(self, names=None) -> None:
        if self.v.w_bits >= 16:
            return
        for n in names or self.names():
            if n not in self.v.fp16_layers:
                w = self.weight_of(n)
                w.copy_(fq_weight(w, self.v.w_bits, self.v.w_group))

    @torch.no_grad()
    def smooth(self, alpha: float) -> None:
        """SmoothQuant on qkv / fc1 (fold into LayerNorm) and out (fold into V rows).
        Needs self.q.chan_amax from a calib_channel pass."""
        c = self.visual.transformer.width
        for i, b in enumerate(self.blocks):
            for k, ln, w in (("qkv", b.ln_1, b.attn.in_proj_weight), ("fc1", b.ln_2, b.mlp.c_fc.weight)):
                name = f"b{i}.{k}"
                if name in self.v.fp16_layers:
                    continue
                s = self._factor(self.q.chan_amax[name], w, alpha)
                ln.weight.div_(s)
                ln.bias.div_(s)
                w.mul_(s)
            name = f"b{i}.out"
            if name not in self.v.fp16_layers:
                w = b.attn.out_proj.weight
                s = self._factor(self.q.chan_amax[name], w, alpha)
                b.attn.in_proj_weight[2 * c:].div_(s[:, None])
                b.attn.in_proj_bias[2 * c:].div_(s)
                w.mul_(s)

    @staticmethod
    def _factor(a_amax, w, alpha):
        w_amax = w.abs().amax(0).float().clamp_min(1e-5)
        return (a_amax.clamp_min(1e-5).pow(alpha) / w_amax.pow(1 - alpha)).clamp(1e-2, 1e2).to(w.dtype)


def preprocess_for(pre, res: int):
    """Stock open_clip eval transform with Resize/CenterCrop set to `res`."""
    out = []
    for t in pre.transforms:
        if isinstance(t, Tv.Resize):
            t = Tv.Resize(res, interpolation=t.interpolation, antialias=True)
        elif isinstance(t, Tv.CenterCrop):
            t = Tv.CenterCrop(res)
        out.append(t)
    return Tv.Compose(out)


@torch.no_grad()
def calibrate(ev: EfficientVisual, batches, mode: str) -> None:
    ev.q.mode = mode
    for x in batches:
        with torch.autocast("cuda", dtype=torch.float16):
            ev(x)
    ev.q.mode = "run"
    if mode == "calib_range":
        ev.q.finish_range_calibration()


def build(visual, v: Variant, calib_batches=None) -> EfficientVisual:
    """Wrap `visual` (modified in place when quantizing). calib_batches: list of input tensors
    on the GPU, needed for static activations and SmoothQuant."""
    ev = EfficientVisual(visual, v).eval()
    if v.a_bits < 16 and v.a_mode == "static":
        assert calib_batches, "static activation quantization needs calibration batches"
        if v.smooth > 0:
            calibrate(ev, calib_batches, "calib_channel")
            ev.smooth(v.smooth)
        ev.quantize_weights()
        calibrate(ev, calib_batches, "calib_range")
    else:
        ev.quantize_weights()
    return ev


def weight_bytes(visual, v: Variant) -> int:
    """Deployable weight size: quantized linears at w_bits (+ fp16 scales), everything else FP16."""
    n_blocks = len(visual.transformer.resblocks) - v.drop
    ev = EfficientVisual(visual, Variant(drop=v.drop))
    total = sum(p.numel() for p in visual.parameters())
    dropped = sum(p.numel() for b in list(visual.transformer.resblocks)[n_blocks:] for p in b.parameters())
    nbytes = (total - dropped) * 2
    if v.w_bits < 16:
        for n in ev.names():
            if n in v.fp16_layers:
                continue
            o, i = ev.weight_of(n).shape
            g = v.w_group if v.w_group and i % v.w_group == 0 else i
            nbytes += -o * i * 2 + o * i * v.w_bits // 8 + o * (i // g) * 2
    return nbytes

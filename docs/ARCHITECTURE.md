# Architecture

## Pipeline
```
camera (full fps) ──► MNv4-Conv-M + SSDLite, INT8, Tensor G4 NPU, 10–30 inf/s
                      boxes labelled genus/family; look-alike group ⇒ ⚠ caution badge
user taps capture ──► BioCLIP 2.5 ViT-H/14 image tower (≈632M params), GPU delegate,
                      INT8/INT4 weights + FP16 activations, ≤10 s
                      · embedding · W (linear probe, text-initialised) · /T · softmax
                      ──► top-5 species with relative %, reference thumbnails,
                          caution sign when any candidate is toxic/conditional,
                          info tabs (parts, conditions, sources). No safe/unsafe verdict.
```
Everything runs on the phone. The text tower never ships: label embeddings are computed offline.

## Hardware
| Role | Device | Notes |
|---|---|---|
| Training | RX 7900 XTX 24 GB (gfx1100), ROCm, pip wheels | >10 TB storage, holds all data |
| Primary target | Pixel 9 Pro: Tensor G4, 16 GB RAM, Mali-G715 GPU, Google TPU | |
| Fallback targets | Pixel 8/8 Pro (G3), Pixel 7 (G2), any arm64 with ≥8 GB | GPU→CPU delegates only |

## BioCLIP size budget (image tower only)
| Precision | Weights | Expected use |
|---|---|---|
| FP32 | ~2.5 GB | training/reference only |
| FP16 | ~1.26 GB | accuracy reference on-device |
| W8A16 | ~0.63 GB | **default** |
| W4A16 | ~0.32 GB | fallback for ≤8 GB phones |
| W8A8 partial | ~0.63 GB | NPU attempt; sensitive layers kept FP16 |

## Runtime fallback chain
- Detector: NPU (AOT-compiled) → GPU → CPU XNNPACK. Lower inf/s on fallback; passthrough stays at full fps.
- BioCLIP: GPU (LiteRT) → NPU (if AOT compile of ViT-H succeeds) → CPU XNNPACK (W4A16).
  Device-aware: the app picks the precision/runtime from a benchmarked table at first launch.

## Efficiency research axis
{runtime: LiteRT-GPU, LiteRT-NPU, ExecuTorch} × {precision: table above} × {ToMe r ∈ 0/8/16} × {drop last 0/2/4 blocks}
→ latency p50/p90, peak memory, top-1, DMR. Pareto plot is the headline figure.

## Distribution
APK ships with no models. First launch downloads a versioned bundle (sha256 manifest):
detector .tflite, BioCLIP .tflite (variant chosen per device), labels.json, linear head, T,
traits, thumbnails + attribution.

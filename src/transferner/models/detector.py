"""Stage-1 live detector.

SSDLite head on timm mobilenetv4_conv_medium features (pure conv -> NPU-friendly).
Classes = genera; family scores are a fixed genus->family sum at post-processing.
Alt backend: Ultralytics YOLO (comparison only, AGPL-3.0)."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

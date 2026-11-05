"""Post-training quantization.

Detector: full INT8 (weights+activations), representative dataset = 500 val images.
BioCLIP: w8a16, w4a16, w8a8_partial. Sensitive-layer search ranks layers by the
increase in look-alike confusion when quantized alone; top-k stay FP16."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

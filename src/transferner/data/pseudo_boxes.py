"""Step 08. Pseudo bounding boxes for the detector.

OWLv2 with generic plant-part prompts, then keeps boxes above min_box_score.
Each box inherits the image's genus label (one subject per image assumption);
images with zero boxes fall back to a full-frame box.
Writes data/manifests/boxes.parquet."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

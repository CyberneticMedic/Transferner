"""BioCLIP 2.5 Huge wrapper.

- load(): open_clip.create_model_and_transforms(hf_hub); returns image tower only for export.
- embed_text(species_df) -> [N,1024] normalized, template-averaged (configs/bioclip.yaml).
- ImageClassifier(encoder, W, logit_scale): W initialised from text embeddings (zero-shot)
  or a trained linear probe; same graph on-device either way.
- Token merging (ToMe) and block dropping hooks for the efficiency sweep."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

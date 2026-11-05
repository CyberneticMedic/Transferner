"""Step 00. Resolve external identifiers before any download.

- Pl@ntNet DOI -> GBIF datasetKey (GET /dataset/doi/{doi}).
- Verify GADM level-1 IDs for SC/NC/GA/TN.
- Record the ToL-200M catalog commit hash and GBIF snapshot info.
Writes data/manifests/sources.json."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

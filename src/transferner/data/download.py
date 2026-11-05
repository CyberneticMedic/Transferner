"""Step 05. Image fetcher (resumable, rate-limited, per-host concurrency).

Input: union of candidate manifests after per-species capping.
Fetches source_url, resizes to images.store_max_side, writes
images/<shard>/<uuid>.jpg, records sha1 + license + attribution.
Writes data/manifests/images.parquet."""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()

"""Step 06. Exact (sha1) and perceptual-hash (imagehash.phash) dedup within each species.

Near-duplicates (Hamming <= leakage.phash_hamming_max) collapse to one row. When a
cluster mixes a ToL image with a regional one, the ToL copy is kept (it is training
data anyway) and the cluster is flagged so the regional copy can never reach test.
Regional records whose gbifID appears as a ToL source_id are flagged in_tol as well.
Rewrites manifests/images.parquet with columns dup_of, in_tol, keep."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ..utils.config import load, data_root


def phash_int(h: pd.Series) -> np.ndarray:
    return np.array([int(x, 16) for x in h], dtype=np.uint64)


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def clusters(h: np.ndarray, max_d: int) -> np.ndarray:
    """Union-find over all pairs within Hamming distance max_d -> root index per row."""
    n = len(h)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        d = popcount64(h[i] ^ h[i + 1:])
        for j in np.nonzero(d <= max_d)[0] + i + 1:
            a, b = find(i), find(j)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def main() -> None:
    cfg = load("data")
    root = data_root()
    img = pd.read_parquet(root / "manifests/images.parquet")
    # rows earlier in this order win a cluster: ToL before regional, then best pick_rank
    img["_ord"] = (img.source == "gbif_regional").astype(int)
    img = img.sort_values(["class_idx", "_ord", "pick_rank"]).reset_index(drop=True)

    # exact duplicates (same bytes) anywhere, then near-duplicates within species
    first = img.groupby("sha1").uuid.transform("first")
    img["dup_of"] = np.where(first != img.uuid, first, None)
    img["near_tol"] = False
    max_d = cfg["leakage"]["phash_hamming_max"]
    for k, g in img[img.dup_of.isna()].groupby("class_idx"):
        if len(g) < 2:
            continue
        roots = clusters(phash_int(g.phash), max_d)
        idx = g.index.to_numpy()
        winner = idx[roots]
        dup = winner != idx
        img.loc[idx[dup], "dup_of"] = img.loc[winner[dup], "uuid"].to_numpy()
        has_tol = pd.Series(g.source.to_numpy() == "tol").groupby(roots).transform("any").to_numpy()
        img.loc[idx, "near_tol"] = has_tol

    tol_ids = set()
    if cfg["leakage"]["exclude_if_in_tol_catalog"]:
        reg_ids = img.loc[img.source == "gbif_regional", "gbif_id"].dropna().unique().tolist()
        if reg_ids:
            c = duckdb.connect()
            c.register("ids", pd.DataFrame({"source_id": reg_ids}))
            tol_ids = set(c.execute(f"""SELECT DISTINCT t.source_id FROM '{root / 'raw/tol/plantae.parquet'}' t
                                        JOIN ids USING (source_id)""").df().source_id)
    reg = img.source == "gbif_regional"
    img["in_tol"] = (img.source == "tol") | (reg & (img.gbif_id.isin(tol_ids) | img.near_tol))
    img["keep"] = img.dup_of.isna()
    img = img.drop(columns=["_ord", "near_tol"])
    img.to_parquet(root / "manifests/images.parquet", index=False)
    print(f"dedup: {len(img):,} rows -> {img.keep.sum():,} kept ({(~img.keep).sum():,} duplicates); "
          f"regional flagged in_tol: {(reg & img.in_tol).sum():,} (gbifID match {len(tol_ids)})")


if __name__ == "__main__":
    main()

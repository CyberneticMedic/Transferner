"""Step 05. Image fetcher (resumable, rate-limited, per-host concurrency).

Input: manifests/gbif_candidates.parquet (regional) U manifests/tol_candidates.parquet.
Queue order: toxic / look-alike / must-include species first, then by regional
abundance, so a partial run is already useful. Within a species: regional first,
then ToL by pick_rank.
Fetches each url (iNaturalist "original" is swapped for "large", 1024 px, since we
store at images.store_max_side anyway), resizes, writes images/<xx>/<uuid>.jpg and
records sha1, phash, size. Progress is checkpointed to cache/download/part-*.parquet,
so re-running resumes; --retry-failed retries earlier failures.
Writes manifests/images.parquet (successful images joined with candidate metadata)."""
from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.parse import urlparse

import imagehash
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from ..utils.config import load, data_root
from ..utils.http import session

Image.MAX_IMAGE_PIXELS = 200_000_000
HOST_LIMITS = {"inaturalist-open-data.s3.amazonaws.com": 24, "static.inaturalist.org": 8}
DEFAULT_HOST_LIMIT = 6
MAX_BYTES = 40 * 2**20
MIN_FREE_GB = 150
DEAD_HOST_AFTER = 60          # attempts with zero successes -> skip host this run
CHECKPOINT_EVERY = 2000

COLS = ["uuid", "class_idx", "source", "observation_id", "url", "license", "rights_holder",
        "publisher", "curation_tier", "is_specimen"]
OPT = ["data_source", "source_id", "basis", "img_type", "pick_rank", "gbif_id", "dataset_key",
       "event_date", "state", "organ", "occurrence_ref", "license_link"]


def fetch_url(url: str) -> str:
    if "inaturalist" in url and "/original." in url:
        return url.replace("/original.", "/large.")
    return url


def rel_path(uuid: str) -> str:
    return f"images/{hashlib.sha1(uuid.encode()).hexdigest()[:2]}/{uuid}.jpg"


def load_queue(root: Path) -> pd.DataFrame:
    sp = pd.read_parquet(root / "manifests/species.parquet")
    parts = []
    for name in ("gbif_candidates", "tol_candidates"):
        p = root / f"manifests/{name}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            parts.append(d[[c for c in COLS + OPT if c in d.columns]])
    q = pd.concat(parts, ignore_index=True).drop_duplicates("uuid")
    pri = sp.set_index("class_idx")
    q["priority"] = q.class_idx.map((pri.is_toxic | pri.is_lookalike | pri.must_include).astype(int))
    q["n_regional"] = q.class_idx.map(pri.n_regional)
    q["src_order"] = (q.source != "gbif_regional").astype(int)
    q["pick_rank"] = q.get("pick_rank", pd.Series(0, index=q.index)).fillna(0)
    # round-robin across species: rank-within-species first, so every species fills evenly
    q = q.sort_values(["priority", "class_idx", "src_order", "pick_rank"], ascending=[False, True, True, True])
    q["rr"] = q.groupby("class_idx").cumcount()
    return q.sort_values(["priority", "rr", "n_regional"], ascending=[False, True, False]).reset_index(drop=True)


class Fetcher:
    def __init__(self, root: Path, max_side: int, quality: int):
        self.root, self.max_side, self.quality = root, max_side, quality
        self.sems: dict[str, threading.Semaphore] = defaultdict(lambda: None)
        self.lock = threading.Lock()
        self.tries: dict[str, int] = defaultdict(int)
        self.oks: dict[str, int] = defaultdict(int)

    def sem(self, host: str) -> threading.Semaphore:
        with self.lock:
            if self.sems[host] is None:
                self.sems[host] = threading.Semaphore(HOST_LIMITS.get(host, DEFAULT_HOST_LIMIT))
            return self.sems[host]

    def __call__(self, uuid: str, url: str) -> dict:
        out = {"uuid": uuid, "status": "ok", "error": None, "path": rel_path(uuid)}
        dst = self.root / out["path"]
        u = fetch_url(url)
        host = urlparse(u).netloc
        if self.tries[host] >= DEAD_HOST_AFTER and self.oks[host] == 0:
            out.update(status="failed", error="HostDead: no successes this run")
            return out
        try:
            with self.sem(host):
                self.tries[host] += 1
                r = session("image").get(u, timeout=(10, 60), stream=True)
                if r.status_code in (403, 404) and u != url:          # some iNat photos lack "large"
                    r.close()
                    r = session("image").get(url, timeout=(10, 60), stream=True)
                try:
                    r.raise_for_status()
                    data = r.raw.read(MAX_BYTES + 1, decode_content=True)
                finally:
                    r.close()
            if len(data) > MAX_BYTES:
                raise ValueError("too large")
            im = Image.open(io.BytesIO(data))
            im = ImageOps.exif_transpose(im).convert("RGB")
            out["orig_w"], out["orig_h"] = im.size
            if min(im.size) < 64:
                raise ValueError(f"too small {im.size}")
            im.thumbnail((self.max_side, self.max_side), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=self.quality, optimize=True)
            tmp.replace(dst)
            out.update(w=im.size[0], h=im.size[1], sha1=hashlib.sha1(data).hexdigest(),
                       phash=str(imagehash.phash(im)))
            self.oks[host] += 1
        except Exception as e:                                          # noqa: BLE001 - record and move on
            out.update(status="failed", error=f"{type(e).__name__}: {str(e)[:200]}")
        return out


def merge_parts(root: Path) -> pd.DataFrame:
    parts = sorted((root / "cache/download").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["uuid", "status"])
    return pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("uuid", keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new attempts (0 = all)")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    cfg = load("data")["images"]
    root = data_root()
    (root / "cache/download").mkdir(parents=True, exist_ok=True)
    q = load_queue(root)
    done = merge_parts(root)
    skip = set(done.uuid) if not args.retry_failed else set(done.loc[done.status == "ok", "uuid"])
    todo = q[~q.uuid.isin(skip)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"queue {len(q):,} | done {len(skip):,} | to fetch {len(todo):,}")

    if not args.merge_only and len(todo):
        fetch = Fetcher(root, cfg["store_max_side"], cfg["jpeg_quality"])
        buf, part_id = [], int(time.time())
        bar = tqdm(total=len(todo), desc="download", smoothing=0.05)
        it = iter(zip(todo.uuid, todo.url))
        with ThreadPoolExecutor(args.workers) as ex:
            pending = set()
            for _ in range(args.workers * 4):
                try:
                    pending.add(ex.submit(fetch, *next(it)))
                except StopIteration:
                    break
            n_ok = 0
            while pending:
                fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in fin:
                    res = f.result()
                    buf.append(res)
                    n_ok += res["status"] == "ok"
                    bar.update()
                    try:
                        pending.add(ex.submit(fetch, *next(it)))
                    except StopIteration:
                        pass
                if len(buf) >= CHECKPOINT_EVERY:
                    pd.DataFrame(buf).to_parquet(root / f"cache/download/part-{part_id}.parquet")
                    part_id, buf = part_id + 1, []
                    bar.set_postfix(ok=f"{n_ok / max(bar.n, 1):.0%}")
                    if shutil.disk_usage(root).free < MIN_FREE_GB * 2**30:
                        print(f"\nfree space below {MIN_FREE_GB} GB; stopping (re-run to resume)")
                        for p in pending:
                            p.cancel()
                        break
        if buf:
            pd.DataFrame(buf).to_parquet(root / f"cache/download/part-{part_id}.parquet")
        bar.close()

    done = merge_parts(root)
    ok = done[done.status == "ok"].drop(columns=["status", "error"])
    img = q.drop(columns=["priority", "n_regional", "src_order", "rr"]).merge(ok, on="uuid", how="inner")
    img.to_parquet(root / "manifests/images.parquet", index=False)
    fails = done.loc[done.status == "failed", "error"].str.split(":").str[0].value_counts().head(8).to_dict()
    print(f"images.parquet: {len(img):,} images, {img.class_idx.nunique()} species | failures: {fails}")


if __name__ == "__main__":
    main()

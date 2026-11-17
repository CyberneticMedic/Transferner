"""Step 05. Image fetcher (resumable, parallel, per-host rate limits).

Input: manifests/gbif_candidates.parquet (regional) U manifests/tol_candidates.parquet.
Queue order: toxic / look-alike / must-include species first, then round-robin across
species, so a partial run is already useful. Within a species: regional first, then ToL
by pick_rank.

Parallelism: `--procs N` starts N worker processes, each owning hash(uuid) % N of the
queue (decode/resize/phash is CPU-bound, so one process is GIL-limited to ~1 core).
Inside a process a scheduler only dispatches an image when its host has a free slot,
so a slow host cannot starve the fast ones (no head-of-line blocking). Per-host limits
are totals across all processes (each gets limit / N), so small herbarium servers see
the same load whatever N is.

Fetches each url (iNaturalist "original" is swapped for "large", 1024 px, since we
store at images.store_max_side anyway), resizes, writes images/<xx>/<uuid>.jpg and
records sha1, phash, size. Progress is checkpointed to cache/download/part-*.parquet,
so re-running resumes; --retry-failed retries earlier failures.
Writes manifests/images.parquet (successful images joined with candidate metadata)."""
from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.parse import urlparse

import imagehash
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm import tqdm

from ..utils.config import load, data_root
from ..utils.http import session

Image.MAX_IMAGE_PIXELS = 200_000_000
# Total concurrent connections per host, summed over all processes.
HOST_LIMITS = {"inaturalist-open-data.s3.amazonaws.com": 192,   # AWS Open Data bucket, built for bulk use
               "static.inaturalist.org": 8}
DEFAULT_HOST_LIMIT = 6
MAX_BYTES = 40 * 2**20
MIN_FREE_GB = 150
DEAD_HOST_AFTER = 60          # attempts with zero successes -> skip host for the rest of this run
CHECKPOINT_EVERY = 2000

COLS = ["uuid", "class_idx", "source", "observation_id", "url", "license", "rights_holder",
        "publisher", "curation_tier", "is_specimen"]
OPT = ["data_source", "source_id", "basis", "img_type", "pick_rank", "gbif_id", "dataset_key",
       "event_date", "state", "organ", "occurrence_ref", "license_link"]
QUEUE_COLS = ["uuid", "class_idx", "source", "url", "pick_rank"]


def fetch_url(url: str) -> str:
    if "inaturalist" in url and "/original." in url:
        return url.replace("/original.", "/large.")
    return url


def rel_path(uuid: str) -> str:
    return f"images/{hashlib.sha1(uuid.encode()).hexdigest()[:2]}/{uuid}.jpg"


def shard_of(uuid: str, n: int) -> int:
    return int(hashlib.md5(uuid.encode()).hexdigest()[:8], 16) % n


def load_queue(root: Path, full: bool = True) -> pd.DataFrame:
    """Candidates in download order. full=False keeps only what the fetch loop needs (low RAM)."""
    sp = pd.read_parquet(root / "manifests/species.parquet")
    want = COLS + OPT if full else QUEUE_COLS
    parts = []
    for name in ("gbif_candidates", "tol_candidates"):
        p = root / f"manifests/{name}.parquet"
        if p.exists():
            have = set(pq.read_schema(p).names)
            parts.append(pd.read_parquet(p, columns=[c for c in want if c in have]))
    q = pd.concat(parts, ignore_index=True).drop_duplicates("uuid")
    pri = sp.set_index("class_idx")
    q["priority"] = q.class_idx.map((pri.is_toxic | pri.is_lookalike | pri.must_include).astype(int))
    q["n_regional"] = q.class_idx.map(pri.n_regional)
    q["src_order"] = (q.source != "gbif_regional").astype(int)
    q["pick_rank"] = q["pick_rank"].fillna(0) if "pick_rank" in q else 0
    # round-robin across species: rank-within-species first, so every species fills evenly
    q = q.sort_values(["priority", "class_idx", "src_order", "pick_rank"], ascending=[False, True, True, True])
    q["rr"] = q.groupby("class_idx").cumcount()
    return q.sort_values(["priority", "rr", "n_regional"], ascending=[False, True, False]).reset_index(drop=True)


class Fetcher:
    def __init__(self, root: Path, max_side: int, quality: int):
        self.root, self.max_side, self.quality = root, max_side, quality

    def __call__(self, uuid: str, url: str) -> dict:
        out = {"uuid": uuid, "status": "ok", "error": None, "path": rel_path(uuid)}
        dst = self.root / out["path"]
        u = fetch_url(url)
        try:
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
            tmp = dst.with_suffix(f".{os.getpid()}.tmp")
            im.save(tmp, "JPEG", quality=self.quality, optimize=True)
            tmp.replace(dst)
            out.update(w=im.size[0], h=im.size[1], sha1=hashlib.sha1(data).hexdigest(),
                       phash=str(imagehash.phash(im)))
        except Exception as e:                                          # noqa: BLE001 - record and move on
            out.update(status="failed", error=f"{type(e).__name__}: {str(e)[:200]}")
        return out


def merge_parts(root: Path, columns: list[str] | None = None) -> pd.DataFrame:
    parts = sorted((root / "cache/download").glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["uuid", "status"])
    return pd.concat([pd.read_parquet(p, columns=columns) for p in parts]).drop_duplicates("uuid", keep="last")


def run_worker(root: Path, todo: pd.DataFrame, workers: int, n_procs: int, label: str) -> None:
    cfg = load("data")["images"]
    fetch = Fetcher(root, cfg["store_max_side"], cfg["jpeg_quality"])
    hosts: dict[str, deque] = {}
    for uuid, url in zip(todo.uuid, todo.url):                    # keeps priority order within each host
        hosts.setdefault(urlparse(fetch_url(url)).netloc, deque()).append((uuid, url))
    limit = {h: max(1, math.ceil(HOST_LIMITS.get(h, DEFAULT_HOST_LIMIT) / n_procs)) for h in hosts}
    inflight, tries, oks = defaultdict(int), defaultdict(int), defaultdict(int)

    tag = f"{int(time.time())}-{os.getpid()}"                       # unique per process
    part = lambda n: root / f"cache/download/part-{tag}-{n:06d}.parquet"  # noqa: E731
    buf, n_part, n_ok, stop = [], 0, 0, False
    bar = tqdm(total=len(todo), desc=label, smoothing=0.02, mininterval=0.5 if sys.stderr.isatty() else 60)
    pending: set = set()

    def fill(ex):
        progressed = True
        while progressed and len(pending) < workers and hosts:
            progressed = False
            for h in list(hosts):
                q = hosts[h]
                if tries[h] >= DEAD_HOST_AFTER and oks[h] == 0:       # dead host: record and drop the rest
                    buf.extend({"uuid": u, "status": "failed", "error": "HostDead: no successes this run",
                                "path": rel_path(u)} for u, _ in q)
                    bar.update(len(q))
                    del hosts[h]
                    continue
                if inflight[h] < limit[h] and len(pending) < workers:
                    uuid, url = q.popleft()
                    f = ex.submit(fetch, uuid, url)
                    f.host = h
                    pending.add(f)
                    inflight[h] += 1
                    tries[h] += 1
                    progressed = True
                if not q:
                    del hosts[h]

    def refill_host(ex, h):
        q = hosts.get(h)
        if q and inflight[h] < limit[h] and not (tries[h] >= DEAD_HOST_AFTER and oks[h] == 0):
            uuid, url = q.popleft()
            f = ex.submit(fetch, uuid, url)
            f.host = h
            pending.add(f)
            inflight[h] += 1
            tries[h] += 1
            if not q:
                del hosts[h]

    last_scan = 0.0
    with ThreadPoolExecutor(workers) as ex:
        fill(ex)
        while pending:
            fin, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                pending.discard(f)
                inflight[f.host] -= 1
                res = f.result()
                oks[f.host] += res["status"] == "ok"
                n_ok += res["status"] == "ok"
                buf.append(res)
                bar.update()
                if not stop:
                    refill_host(ex, f.host)                # cheap: same host usually has more work
            now = time.monotonic()
            if not stop and (now - last_scan > 0.5 or not pending):  # full scan picks up other hosts
                fill(ex)
                last_scan = now
            if len(buf) >= CHECKPOINT_EVERY:
                pd.DataFrame(buf).to_parquet(part(n_part))
                n_part, buf = n_part + 1, []
                bar.set_postfix(ok=f"{n_ok / max(bar.n, 1):.0%}", hosts=len(hosts))
                if not stop and shutil.disk_usage(root).free < MIN_FREE_GB * 2**30:
                    print(f"\n[{label}] free space below {MIN_FREE_GB} GB; draining and stopping")
                    stop = True
    if buf:
        pd.DataFrame(buf).to_parquet(part(n_part))
    bar.close()


def write_manifest(root: Path) -> None:
    done = merge_parts(root)
    ok = done[done.status == "ok"].drop(columns=["status", "error"])
    q = load_queue(root, full=True)
    img = q.drop(columns=["priority", "n_regional", "src_order", "rr"]).merge(ok, on="uuid", how="inner")
    img.to_parquet(root / "manifests/images.parquet", index=False)
    fails = done.loc[done.status == "failed", "error"].str.split(":").str[0].value_counts().head(8).to_dict()
    print(f"images.parquet: {len(img):,} images, {img.class_idx.nunique()} species "
          f"(queue {len(q):,}) | failures: {fails}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=1, help="worker processes (each owns 1/N of the queue)")
    ap.add_argument("--workers", type=int, default=64, help="threads per process")
    ap.add_argument("--shard", default=None, help="internal: i/N, set by --procs")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new attempts (0 = all)")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--only-source", choices=["tol", "gbif_regional"])
    args = ap.parse_args()

    root = data_root()
    (root / "cache/download").mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        write_manifest(root)
        return

    if args.procs > 1 and args.shard is None:                        # parent: fan out, wait, merge
        base = [sys.executable, "-m", "transferner.data.download", "--workers", str(args.workers)]
        for flag, val in (("--only-source", args.only_source), ("--limit", args.limit or None)):
            if val:
                base += [flag, str(val)]
        if args.retry_failed:
            base.append("--retry-failed")
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        tag = args.only_source or "all"
        procs = []
        for i in range(args.procs):
            log = open(logs / f"download_{tag}_{i}of{args.procs}.log", "a")
            procs.append(subprocess.Popen(base + ["--shard", f"{i}/{args.procs}"], stdout=log, stderr=log))
        print(f"started {args.procs} shards: pids {[p.pid for p in procs]}; logs in {logs}", flush=True)
        codes = [p.wait() for p in procs]
        print(f"shards exited {codes}", flush=True)
        write_manifest(root)
        return

    i, n = map(int, args.shard.split("/")) if args.shard else (0, 1)
    q = load_queue(root, full=False)
    done = merge_parts(root, columns=["uuid", "status"])
    skip = set(done.uuid) if not args.retry_failed else set(done.loc[done.status == "ok", "uuid"])
    del done
    todo = q[~q.uuid.isin(skip)]
    if args.only_source:
        todo = todo[todo.source == args.only_source]
    if n > 1:
        todo = todo[todo.uuid.map(lambda u: shard_of(u, n) == i)]
    if args.limit:
        todo = todo.head(args.limit // n if n > 1 else args.limit)
    todo = todo[["uuid", "url"]].reset_index(drop=True)
    del q
    print(f"shard {i}/{n}: to fetch {len(todo):,} (already done {len(skip):,})", flush=True)
    if len(todo):
        run_worker(root, todo, args.workers, n, f"dl{i}/{n}")
    if n == 1:
        write_manifest(root)


if __name__ == "__main__":
    main()

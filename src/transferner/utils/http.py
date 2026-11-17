"""Shared HTTP session: retries, backoff, a generic User-Agent, and an on-disk JSON cache.

The User-Agent deliberately carries no personal contact details.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import data_root

UA = "Transferner-research/0.1 (academic plant-identification project)"
GBIF = "https://api.gbif.org/v1"

_local = threading.local()


def session(kind: str = "api") -> requests.Session:
    """One session per thread and kind (requests.Session is not guaranteed thread-safe).

    "api": patient retries for GBIF/HF metadata. "image": fail fast; the downloader
    records failures and can retry them later with --retry-failed."""
    attr = f"s_{kind}"
    s = getattr(_local, attr, None)
    if s is None:
        s = requests.Session()
        if kind == "api":
            retry = Retry(total=8, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                          allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
        else:
            retry = Retry(total=2, connect=1, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
                          allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers["User-Agent"] = UA
        setattr(_local, attr, s)
    return s


def _cache_path(url: str, params: dict | None) -> Path:
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()
    return data_root() / "cache" / "http" / h[:2] / f"{h}.json"


def get_json(url: str, params: dict | None = None, cache: bool = True, timeout: float = 60):
    """GET -> parsed JSON. Cached on disk forever when cache=True (re-run with cache=False to refresh)."""
    p = _cache_path(url, params)
    if cache and p.exists():
        return json.loads(p.read_text())
    r = session().get(url, params=params, timeout=timeout)
    if r.status_code == 404:
        data = None
    else:
        r.raise_for_status()
        data = r.json()
    if cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)
    return data

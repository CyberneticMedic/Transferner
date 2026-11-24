"""Publish a built APK on the home bundle server for in-app updates (D-042).

Reads versionCode/versionName from the APK (aapt2), checks the signing certificate against the sideload key,
copies the APK to $TRANSFERNER_DATA/bundle/transferner-<versionName>.apk, writes app.json (version, size,
SHA-256, notes; atomically, after the APK is in place), points index.html at it, and removes older APKs.
  source /models/ml/android/env.sh
  python scripts/publish_apk.py [--apk app/app/build/outputs/apk/release/app-release.apk] [--notes "..."]"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

SIDELOAD_CERT_SHA256 = "35a2f1fd67c352f124563bc58008b2799f64f8062677e7e8e10ece3ab7ff380e"


def tool(name: str) -> str:
    bt = Path(os.environ["ANDROID_HOME"]) / "build-tools"
    return str(sorted(bt.iterdir())[-1] / name)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--apk", default=str(repo / "app/app/build/outputs/apk/release/app-release.apk"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    apk = Path(args.apk)
    badging = subprocess.run([tool("aapt2"), "dump", "badging", str(apk)], capture_output=True, text=True, check=True).stdout
    code = int(re.search(r"versionCode='(\d+)'", badging).group(1))
    name = re.search(r"versionName='([^']+)'", badging).group(1)
    certs = subprocess.run([tool("apksigner"), "verify", "--print-certs", str(apk)], capture_output=True, text=True, check=True).stdout
    if SIDELOAD_CERT_SHA256 not in certs:
        raise SystemExit("APK is not signed with the sideload key: installed apps would refuse it as an update")
    base = Path(os.environ["TRANSFERNER_DATA"]) / "bundle"
    current = json.loads((base / "app.json").read_text()) if (base / "app.json").exists() else None
    if current and current["versionCode"] >= code:
        raise SystemExit(f"server already offers versionCode {current['versionCode']} >= {code}; bump versionCode")
    dst = base / f"transferner-{name}.apk"
    shutil.copyfile(apk, dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    rel = {"versionCode": code, "versionName": name, "apk": dst.name, "bytes": dst.stat().st_size, "sha256": sha, "notes": args.notes}
    tmp = base / "app.json.tmp"
    tmp.write_text(json.dumps(rel))
    tmp.replace(base / "app.json")
    idx = base / "index.html"
    if idx.exists():
        s = idx.read_text()
        s = re.sub(r"transferner-[0-9][^\"]*\.apk", dst.name, s)
        s = re.sub(r"Transferner [0-9][0-9.]*<", f"Transferner {name}<", s)
        idx.write_text(s)
    for old in base.glob("transferner-*.apk"):
        if old.name != dst.name:
            old.unlink()
    print(f"published {dst.name}: versionCode {code}, {rel['bytes'] / 2**20:.1f} MB, sha256 {sha[:12]}…")


if __name__ == "__main__":
    main()

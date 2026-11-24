"""Home-network server for the app prototype (D-034/D-036): serves $TRANSFERNER_DATA/bundle/ (APK, index page,
model bundle) and accepts the app's observation-log uploads.

  GET  /...                               static files from the bundle directory
  POST /upload/<device>/<file>            body = file bytes; saved to $TRANSFERNER_DATA/field_logs/<device>/<file>

Names are restricted to [A-Za-z0-9._-] (no paths), files to 25 MB, and an existing file is never overwritten
(a new name gets a numeric suffix), except observations.csv, which the app re-sends whole. Plain HTTP on the
home network only, like the bundle itself. Run through scripts/serve_bundle.sh."""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
MAX_BYTES = 25 * 1024 * 1024


class Handler(SimpleHTTPRequestHandler):
    uploads: Path = Path(".")

    def do_POST(self):  # noqa: N802 (http.server naming)
        parts = self.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "upload" or not all(NAME.match(p) for p in parts[1:]) or ".." in self.path:
            self.send_error(400, "expected /upload/<device>/<file>")
            return
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > MAX_BYTES:
            self.send_error(413, "empty or too large")
            return
        d = self.uploads / parts[1]
        d.mkdir(parents=True, exist_ok=True)
        dst = d / parts[2]
        if dst.exists() and parts[2] != "observations.csv":
            stem, suf, k = dst.stem, dst.suffix, 1
            while dst.exists():
                dst = d / f"{stem}.{k}{suf}"
                k += 1
        data = self.rfile.read(n)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        self.send_response(201)
        self.send_header("Content-Length", "0")
        self.end_headers()
        self.log_message("upload %s (%d bytes)", dst.relative_to(self.uploads), n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    root = Path(os.environ["TRANSFERNER_DATA"])
    Handler.uploads = root / "field_logs"
    Handler.uploads.mkdir(exist_ok=True)
    ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=str(root / "bundle"))).serve_forever()


if __name__ == "__main__":
    main()

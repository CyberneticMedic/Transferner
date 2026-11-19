"""Append-only CSV run logging. Every experiment writes results/runs/<run_id>/.

Layout per run:
  config.yaml   frozen copy of every config used
  metrics.csv   step/epoch-level metrics (long format: step,split,metric,value)
  summary.csv   one row of final metrics, appended to results/tables/<table>.csv
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .config import REPO


class RunLog:
    def __init__(self, name: str):
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
        self.dir = REPO / "results" / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._f = open(self.dir / "metrics.csv", "a", newline="")
        self._w = csv.writer(self._f)
        if self._f.tell() == 0:
            self._w.writerow(["step", "split", "metric", "value"])

    def log(self, step: int, split: str, **metrics: float) -> None:
        for k, v in metrics.items():
            self._w.writerow([step, split, k, v])
        self._f.flush()

    def summary(self, table: str, **row) -> None:
        """Append one row (with run_id) to results/tables/<table>.csv."""
        path = REPO / "results" / "tables" / f"{table}.csv"
        row = {"run_id": self.run_id, **row}
        old_rows, fields, rewrite = [], list(row), not path.exists()
        if path.exists():
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                old_fields = r.fieldnames or []
                fields = old_fields + [k for k in row if k not in old_fields]
                if fields != old_fields:          # new columns: rewrite with the widened header
                    old_rows, rewrite = list(r), True
        if rewrite:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(old_rows)
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    def close(self) -> None:
        self._f.close()

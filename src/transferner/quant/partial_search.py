"""Partial W8A8: smallest number k of FP16 layers that meets the accuracy target on val.

For k in --ks (ascending): static W8A8 with the k most sensitive layers (from the
sensitivity CSV) kept in FP16, evaluated on val only. Stops at the first k whose regional
val top-1 >= target x the FP16 reference's (artifacts/efficiency/fp16/result.json), then
runs that k on test. Every attempt is a row in results/tables/efficiency.csv."""
from __future__ import annotations

import argparse
import json

from ..eval import efficiency
from ..utils.config import data_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", default="quant_sensitivity.csv", help="file in results/tables")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 4, 8, 16, 32, 64])
    ap.add_argument("--target", type=float, default=0.98, help="fraction of FP16 regional-val top-1 to keep")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--a-pct", type=float, default=99.99)
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--prefix", default="w8a8p")
    args = ap.parse_args()
    ref = json.loads((data_root() / "artifacts/efficiency/fp16/result.json").read_text())
    goal = args.target * ref["val_regional_top1"]
    print(f"target: regional val top-1 >= {goal:.4f} ({args.target:.0%} of FP16 {ref['val_regional_top1']:.4f})")

    def variant(k, val_only):
        return efficiency.parser().parse_args([
            "--name", f"{args.prefix}_s{args.smooth}_k{k}", "--w-bits", str(args.w_bits), "--a-bits", "8",
            "--a-mode", "static", "--smooth", str(args.smooth), "--a-pct", str(args.a_pct),
            "--fp16-from", args.sens, "--fp16-k", str(k), "--note", "partial search" + (" (val)" if val_only else "")]
            + (["--val-only"] if val_only else []))

    for k in sorted(args.ks):
        row = efficiency.run(variant(k, True))
        ok = row["val_regional_top1"] >= goal
        print(f"k={k}: regional val {row['val_regional_top1']:.4f} -> {'meets' if ok else 'misses'} target", flush=True)
        if ok:
            efficiency.run(variant(k, False))
            return
    print("no k met the target; widen --ks, raise --smooth, or change --a-pct")


if __name__ == "__main__":
    main()

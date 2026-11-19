"""A5: efficiency report from results/tables/efficiency.csv (CPU only, safe while the GPU queue runs).

  python -m transferner.eval.efficiency_report

Writes results/EFFICIENCY.md (every variant: test top-1/top-5, delta vs its reference, val,
weight MB, GFLOPs, DMR) and results/figures/efficiency_pareto.png (test top-1 against GFLOPs
and against weight MB, two panels, one y-axis each).

Reference per row: production-head variants are compared with `fp16`; retrained-head variants
(--retrain, --drop) with `base_subset`, the same 40-per-species head on unmodified features.
DMR (dangerous miss rate, contact / ingestion) is recomputed here from each variant's saved
test embeddings and head, with the toxic sets of the current trait table (Phase C). Agreement
and McNemar are recomputed the same way against the row's own reference (efficiency.csv's
`*_agree_ref` columns are always against `fp16`, which is wrong for retrained heads).
Latest row per name wins; smoke rows live in efficiency_smoke.csv and are ignored."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..utils.config import RESULTS, data_root
from . import metrics as M
from .efficiency import prod_head_dir
from .evaluate import trait_flags

TARGET = 0.98          # OVERNIGHT.md acceptance: partial W8A8 test top-1 >= 0.98 x FP16
# dataviz reference palette, first three categorical slots (validated all-pairs for scatter)
COLOR = {"quantization": "#2a78d6", "structural": "#eb6834", "combination": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def family(r) -> str:
    if r["name"] == "fp16":
        return "reference"
    quant = r["w_bits"] < 16 or r["a_bits"] < 16
    struct = r["res"] != 224 or r["tome_r"] > 0 or r["drop"] > 0 or r["head"] == "retrained"
    return "combination" if quant and struct else "quantization" if quant else "structural"


def load_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "tables/efficiency.csv")
    if "limit" in df:
        df = df[df["limit"].fillna(0) == 0]
    df = df.drop_duplicates("name", keep="last").reset_index(drop=True)
    df["family"] = df.apply(family, axis=1)
    df["ref"] = np.where(df["head"] == "retrained", "base_subset", "fp16")
    df.loc[df.name.isin(["fp16", "base_subset"]), "ref"] = ""
    return df


@torch.no_grad()
def test_preds(name: str, head: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Top-1 test predictions and labels from a variant's saved embeddings and its head."""
    root = data_root()
    d = root / "artifacts/efficiency" / name
    if not (d / "test.npy").exists():
        return None
    torch.set_num_threads(8)
    pr = torch.load((d if head == "retrained" else prod_head_dir(root)) / "probe.pt", map_location="cpu")
    W, b, s = pr["W"].float(), pr["b"].float(), pr["scale"]
    e = np.load(d / "test.npy", mmap_mode="r")
    pred = np.concatenate([(s * torch.from_numpy(np.asarray(e[i:i + 16384], np.float32)) @ W.T + b).argmax(-1).numpy()
                           for i in range(0, len(e), 16384)])
    y = pd.read_parquet(d / "test.parquet").class_idx.to_numpy()
    return pred, y


def dmr_for(py: tuple[np.ndarray, np.ndarray] | None, tox: dict[str, np.ndarray]) -> dict:
    if py is None:
        return {}
    pred, y = py
    return {f"dmr_{kind.split('_')[0]}": M.dmr(pred, y, tox[kind])["dmr"]
            for kind in ("contact_toxic", "ingestion_toxic")}


def vs_ref(py, ref_py) -> dict:
    """Top-1 agreement and exact McNemar p against the reference, on the same test rows."""
    if py is None or ref_py is None or len(py[1]) != len(ref_py[1]) or not np.array_equal(py[1], ref_py[1]):
        return {"test_agree_ref": np.nan, "test_mcnemar_p_ref": np.nan}
    (p, y), (rp, _) = py, ref_py
    return {"test_agree_ref": float((p == rp).mean()), "test_mcnemar_p_ref": M.mcnemar(rp == y, p == y)["p"]}


def pareto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mask of points not dominated (lower x and higher y are better)."""
    keep = np.ones(len(x), bool)
    for i in range(len(x)):
        keep[i] = not np.any((x <= x[i]) & (y >= y[i]) & ((x < x[i]) | (y > y[i])))
    return keep


def figure(df: pd.DataFrame, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df[df.test_top1.notna()]
    ref = t[t.name == "fp16"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, (col, xl) in zip(axes, [("gflops", "GFLOPs per image"), ("weight_mb", "Deployable weights (MB)")]):
        for fam, c in COLOR.items():
            s = t[(t.family == fam) & (t["head"] != "retrained")]
            ax.scatter(s[col], 100 * s.test_top1, s=42, color=c, edgecolors="white", linewidths=1.5,
                       label=fam, zorder=3)
            s = t[(t.family == fam) & (t["head"] == "retrained")]          # hollow: own head, see base_subset
            ax.scatter(s[col], 100 * s.test_top1, s=42, facecolors="white", edgecolors=c, linewidths=2,
                       label=f"{fam}, retrained head" if len(s) else None, zorder=3)
        if len(ref):
            ax.scatter(ref[col], 100 * ref.test_top1, s=60, marker="D", color=INK, edgecolors="white",
                       linewidths=1.5, label="FP16 reference", zorder=4)
            ax.axhline(100 * TARGET * ref.test_top1.iloc[0], color=MUTED, lw=1, zorder=1)
            ax.annotate("98% of FP16 (acceptance)", (ax.get_xlim()[0], 100 * TARGET * ref.test_top1.iloc[0]),
                        xytext=(4, 3), textcoords="offset points", fontsize=8, color=MUTED)
        front = t[pareto(t[col].to_numpy(), t.test_top1.to_numpy())]
        for r in front.itertuples():
            ax.annotate(r.name, (getattr(r, col), 100 * r.test_top1), xytext=(5, -10), textcoords="offset points",
                        fontsize=8, color=INK)
        ax.set_xlabel(xl, color=INK)
        ax.grid(True, color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel("Test top-1 (%)", color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.95),
               ncol=len(lab), handletextpad=0.3, columnspacing=1.2)
    fig.suptitle("BioCLIP 2.5 ViT-H/14 efficiency variants: test top-1 vs cost (labels: Pareto front)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Fake-quantized variants keep FP16 FLOPs; their saving shows in the weight panel. "
             "Retrained-head variants are judged against base_subset in EFFICIENCY.md.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)


def fmt(x, nd=2, pct=False, sign=False):
    if pd.isna(x):
        return "–"
    v = 100 * x if pct else x
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def main() -> None:
    root = data_root()
    df = load_table()
    sp = pd.read_parquet(root / "manifests/species.parquet").sort_values("class_idx").reset_index(drop=True)
    tox = trait_flags(sp)
    preds = {r.name: test_preds(r.name, r.head) for r in df.itertuples()}
    d = pd.DataFrame([dmr_for(preds[r.name], tox) for r in df.itertuples()], index=df.index)
    df = df.join(d) if len(d.columns) else df
    for i, r in df.iterrows():                    # agreement / McNemar against the row's own reference
        for k, v in (vs_ref(preds[r["name"]], preds.get(r["ref"])) if r["ref"] else
                     {"test_agree_ref": np.nan, "test_mcnemar_p_ref": np.nan}).items():
            df.at[i, k] = v
    by_name = df.set_index("name")
    for c in ("test_top1", "val_regional_top1"):
        refv = df.ref.map(lambda n: by_name[c].get(n, np.nan) if n else np.nan)
        df[f"d_{c}"] = df[c] - refv
        df[f"rel_{c}"] = df[c] / refv
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    figure(df, RESULTS / "figures/efficiency_pareto.png")

    order = {"reference": 0, "quantization": 1, "structural": 2, "combination": 3}
    tested = df[df.test_top1.notna()].assign(_o=lambda d: d.family.map(order) + (d.name != "base_subset") * 0.5)
    tested = tested.sort_values(["_o", "name"])
    val_only = df[df.test_top1.isna()]
    L = ["# Efficiency variants (BioCLIP 2.5 ViT-H/14 tower + production probe)", "",
         "Generated by `python -m transferner.eval.efficiency_report` from `results/tables/efficiency.csv`. "
         "Accuracy is measured on recent regional photos (test 176,590 images); variants were selected on val. "
         "Δ is in points against the row's reference (`fp16`, or `base_subset` for retrained heads); "
         "rel = test top-1 / reference. Acceptance for partial W8A8: rel ≥ 0.98.", "",
         "DMR = dangerous miss rate: share of test images of toxic species (contact / ingestion, from "
         "`data/traits/species_traits.csv`) predicted as a non-toxic species, without abstention. "
         "The trait table is still partial, so DMR covers only the species it lists.", "",
         "Latency, memory and NPU compile need the phone; GPU img/s here is a ROCm proxy for float variants only.", "",
         "| variant | family | ref | test top-1 | Δ pt | rel | test top-5 | val reg top-1 | val_ext top-1 | rare top-1 "
         "| agree ref | McNemar p | DMR contact | DMR ingest | weights MB | GFLOPs | GPU img/s |",
         "|" + "---|" * 17]
    for r in tested.itertuples():
        L.append(f"| {r.name} | {r.family} | {r.ref or '–'} | {fmt(r.test_top1, 2, True)} | "
                 f"{fmt(r.d_test_top1, 2, True, True)} | {fmt(r.rel_test_top1, 4)} | {fmt(r.test_top5, 2, True)} | "
                 f"{fmt(r.val_regional_top1, 2, True)} | {fmt(getattr(r, 'val_ext_top1', np.nan), 2, True)} | "
                 f"{fmt(getattr(r, 'test_rare_top1', np.nan), 2, True)} | {fmt(getattr(r, 'test_agree_ref', np.nan), 4)} | "
                 f"{'–' if pd.isna(getattr(r, 'test_mcnemar_p_ref', np.nan)) else f'{r.test_mcnemar_p_ref:.1e}'} | "
                 f"{fmt(getattr(r, 'dmr_contact', np.nan), 2, True)} | {fmt(getattr(r, 'dmr_ingestion', np.nan), 2, True)} | "
                 f"{fmt(r.weight_mb, 0)} | {fmt(r.gflops, 1)} | {fmt(getattr(r, 'gpu_img_s', np.nan), 0)} |")
    if len(val_only):
        L += ["", "## Val-only runs (partial-INT8 search, calibration and combination screens)", "",
              "| variant | val reg top-1 | Δ pt vs fp16 | rel | val_ext top-1 | agree fp16 | cos fp16 | fp16 layers | GFLOPs | note |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in val_only.itertuples():
            note = getattr(r, "note", "")
            L.append(f"| {r.name} | {fmt(r.val_regional_top1, 2, True)} | {fmt(r.d_val_regional_top1, 2, True, True)} | "
                     f"{fmt(r.rel_val_regional_top1, 4)} | {fmt(getattr(r, 'val_ext_top1', np.nan), 2, True)} | "
                     f"{fmt(getattr(r, 'val_regional_agree_ref', np.nan), 4)} | "
                     f"{fmt(getattr(r, 'val_regional_cos_ref', np.nan), 4)} | {r.fp16_k} | {fmt(r.gflops, 1)} | "
                     f"{'' if pd.isna(note) else note} |")
    L += ["", "![Pareto](figures/efficiency_pareto.png)", ""]
    (RESULTS / "EFFICIENCY.md").write_text("\n".join(L))
    print("\n".join(L[:12 + len(tested)]))


if __name__ == "__main__":
    main()

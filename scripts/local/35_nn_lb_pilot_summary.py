"""Summary for the NN + LB candidate-cell pilot (script 34).

Reads:
  - data/local/<short>_nn_lb_pilot/emotional_raw.jsonl  (the pilot)
  - data/local/<short>/emotional_raw.jsonl              (canonical v4 main)

Computes:
  1. Per-cell modal kaomoji table (NN, LB; pilot rows only).
  2. Cosine-similarity of cell-mean h_first vectors: NN and LB pilot
     means against each existing v4 cell mean (HP-S, HP-D, LP, HN-D,
     HN-S, LN, NB, NP, HB). Identifies the nearest existing cell for
     NN and LB — the structural-distinctiveness read.
  3. Per-row nearest-existing-cell histogram. Confirms whether the
     hidden state lands in a region populated by an existing cell, or
     genuinely off-grid. Both cell-mean cosine and per-row nearest-
     cell tell the same story when the cell is internally tight; they
     diverge when it isn't.

n=10 per cell × 1 seed is too small for the v5-promotion 95%ile-vs-
permutation gate, but adequate for a directional read on whether the
cells separate at all under existing v4 hidden-state geometry.

Output: data/local/<short>_nn_lb_pilot/pilot_summary.{tsv,md} per
model + a combined cross-model markdown at
data/local/nn_lb_pilot_summary.md.

Usage:
  LLMOJI_MODEL=gemma python scripts/local/35_nn_lb_pilot_summary.py
  LLMOJI_MODEL=qwen  python scripts/local/35_nn_lb_pilot_summary.py
  python scripts/local/35_nn_lb_pilot_summary.py --models gemma,qwen
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_experiment.config import DATA_DIR, MODEL_REGISTRY, resolve_model
from llmoji_experiment.emotional_analysis import (
    load_emotional_features_stack,
    load_emotional_features_stack_at,
)


PILOT_CELLS = ("NN", "LB")


def _cell_means(df: pd.DataFrame, X: np.ndarray) -> dict[str, np.ndarray]:
    """Mean h_first per cell. Cells are read from df['quadrant']."""
    out: dict[str, np.ndarray] = {}
    for q, sub in df.groupby("quadrant"):
        idx = sub.index.to_numpy()
        if len(idx) == 0:
            continue
        out[str(q)] = X[idx].mean(axis=0)
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _modal_face_table(df: pd.DataFrame, top_k: int = 5) -> dict[str, list[tuple[str, int]]]:
    """For each cell in df, return the top-k canonical faces (most-frequent)."""
    out: dict[str, list[tuple[str, int]]] = {}
    for q, sub in df.groupby("quadrant"):
        c = Counter(sub["first_word"].dropna())
        out[str(q)] = c.most_common(top_k)
    return out


def _per_row_nearest(
    X_pilot: np.ndarray, v4_means: dict[str, np.ndarray]
) -> list[str]:
    """For each pilot row, return the nearest v4 cell name by cosine."""
    cell_names = list(v4_means.keys())
    M = np.stack([v4_means[c] for c in cell_names], axis=0)
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    Xn = X_pilot / np.maximum(np.linalg.norm(X_pilot, axis=1, keepdims=True), 1e-9)
    sims = Xn @ Mn.T  # (n_rows, n_cells)
    best_idx = sims.argmax(axis=1)
    return [cell_names[i] for i in best_idx]


def summarize_one_model(short: str) -> dict:
    """Build the full per-model summary dict. Returns the structured
    payload used downstream by both the per-model TSV/MD writers and
    the cross-model writer."""
    # 1. Load canonical v4 main with HN + HP D/S split applied.
    os.environ.pop("LLMOJI_OUT_SUFFIX", None)
    df_v4, X_v4 = load_emotional_features_stack(short, split_hn=True)
    df_v4 = df_v4.assign(quadrant=df_v4["quadrant"].astype(str))

    # 2. Load pilot. resolve_model with LLMOJI_OUT_SUFFIX=nn_lb_pilot
    #    redirects emotional_data_path / experiment.
    os.environ["LLMOJI_OUT_SUFFIX"] = "nn_lb_pilot"
    os.environ["LLMOJI_MODEL"] = short
    M_pilot = resolve_model(short)
    if not M_pilot.emotional_data_path.exists():
        raise SystemExit(
            f"pilot JSONL missing for {short}: {M_pilot.emotional_data_path}\n"
            f"run: LLMOJI_MODEL={short} python scripts/local/34_nn_lb_pilot.py"
        )
    df_pilot, X_pilot = load_emotional_features_stack_at(
        M_pilot.emotional_data_path,
        DATA_DIR,
        experiment=M_pilot.experiment,
    )
    # No PAD split for pilot — NN and LB are d=0 cells.
    df_pilot = df_pilot.assign(quadrant=df_pilot["quadrant"].astype(str))

    # Sanity: pilot quadrants should be exactly NN and LB.
    pilot_qs = set(df_pilot["quadrant"].unique())
    if not pilot_qs.issubset({"NN", "LB"}):
        print(f"  WARN: unexpected pilot quadrants {pilot_qs}")

    # 3. Cell means (v4 split-applied) + pilot means.
    v4_means = _cell_means(df_v4, X_v4)
    pilot_means = _cell_means(df_pilot, X_pilot)

    # 4. Cosine of pilot mean vs each v4 mean.
    cos_table: list[dict] = []
    for pcell in PILOT_CELLS:
        if pcell not in pilot_means:
            continue
        for v4cell, v4vec in sorted(v4_means.items()):
            cos_table.append({
                "pilot_cell": pcell,
                "v4_cell": v4cell,
                "cosine": _cosine(pilot_means[pcell], v4vec),
                "n_pilot": int((df_pilot["quadrant"] == pcell).sum()),
                "n_v4": int((df_v4["quadrant"] == v4cell).sum()),
            })
    cos_df = pd.DataFrame(cos_table)

    # 5. Per-row nearest-v4-cell histogram.
    nearest_per_row = _per_row_nearest(X_pilot, v4_means)
    df_pilot = df_pilot.assign(nearest_v4_cell=nearest_per_row)
    nearest_hist: dict[str, Counter[str]] = {}
    for pcell in PILOT_CELLS:
        sub = df_pilot[df_pilot["quadrant"] == pcell]
        nearest_hist[pcell] = Counter(sub["nearest_v4_cell"])

    # 6. Modal face table.
    modal = _modal_face_table(df_pilot)

    return {
        "short": short,
        "df_pilot": df_pilot,
        "cos_df": cos_df,
        "modal": modal,
        "nearest_hist": nearest_hist,
        "v4_means": v4_means,
        "pilot_means": pilot_means,
    }


def write_per_model_outputs(payload: dict) -> None:
    short = payload["short"]
    out_dir = DATA_DIR / "local" / f"{short}_nn_lb_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    # TSV: cosine table.
    payload["cos_df"].to_csv(out_dir / "pilot_summary.tsv", sep="\t", index=False)

    # Markdown: modal faces + cosine table + per-row nearest histogram.
    md: list[str] = []
    md.append(f"# NN + LB pilot summary — {short}\n")
    md.append(f"n=10 prompts × 1 seed × 2 cells = 20 generations. "
              f"Hidden-state read: cosine of pilot cell-mean h_first against "
              f"each v4 cell-mean h_first (PAD-split v4: HP-D / HP-S / LP / "
              f"NP / HN-D / HN-S / LN / NB / HB). At n=10 per cell the "
              f"cosine read is directional only; gates require N=20 with "
              f"95%ile-vs-permutation. See "
              f"docs/2026-05-06-nn-lb-future-cells.md for the v5-promotion plan.\n")

    md.append("## Modal faces per pilot cell\n")
    for cell in PILOT_CELLS:
        rows = payload["modal"].get(cell, [])
        if not rows:
            md.append(f"- {cell}: (no rows)\n")
            continue
        md.append(f"- {cell} ({len(payload['df_pilot'][payload['df_pilot']['quadrant'] == cell])}/10 kaomoji-bearing):")
        for face, k in rows:
            md.append(f"    - `{face}` × {k}")
        md.append("")

    md.append("## Cosine of pilot cell-mean vs v4 cell-mean (h_first stack)\n")
    md.append("| pilot | nearest v4 cell | cos | full table |")
    md.append("|---|---|---|---|")
    cos_df = payload["cos_df"]
    for pcell in PILOT_CELLS:
        sub = cos_df[cos_df["pilot_cell"] == pcell].sort_values("cosine", ascending=False)
        if sub.empty:
            continue
        nearest = sub.iloc[0]
        full = ", ".join(f"{row.v4_cell} {row.cosine:.3f}" for row in sub.itertuples())
        md.append(f"| {pcell} | **{nearest.v4_cell}** | {nearest.cosine:.3f} | {full} |")
    md.append("")

    md.append("## Per-row nearest-v4-cell histogram\n")
    for pcell in PILOT_CELLS:
        hist = payload["nearest_hist"].get(pcell, Counter())
        if not hist:
            continue
        md.append(f"- {pcell}: " + ", ".join(
            f"{cell} × {n}" for cell, n in hist.most_common()
        ))
    md.append("")

    md.append("## Per-row pilot data\n")
    md.append("| prompt_id | cell | first_word | nearest_v4 |")
    md.append("|---|---|---|---|")
    df = payload["df_pilot"].sort_values(["quadrant", "prompt_id"])
    for r in df.itertuples():
        md.append(f"| {r.prompt_id} | {r.quadrant} | `{r.first_word}` | {r.nearest_v4_cell} |")
    md.append("")

    (out_dir / "pilot_summary.md").write_text("\n".join(md))
    print(f"  wrote {out_dir / 'pilot_summary.tsv'}")
    print(f"  wrote {out_dir / 'pilot_summary.md'}")


def write_combined_md(payloads: list[dict]) -> None:
    out_path = DATA_DIR / "local" / "nn_lb_pilot_summary.md"
    md: list[str] = []
    md.append("# NN + LB candidate-cell pilot — cross-model summary\n")
    md.append(
        "Models: " + ", ".join(p["short"] for p in payloads)
        + ". n=10 prompts × 1 seed × 2 cells per model. "
          "Pre-registered in docs/2026-05-06-nn-lb-future-cells.md "
          "(steps 1–2 evidence-gathering deferred; this is step 3 "
          "smoke pilot).\n"
    )

    md.append("## Modal faces per (model, cell)\n")
    md.append("| model | cell | top-3 faces |")
    md.append("|---|---|---|")
    for p in payloads:
        for cell in PILOT_CELLS:
            rows = p["modal"].get(cell, [])
            tag = ", ".join(f"`{f}`×{k}" for f, k in rows[:3]) or "(none)"
            md.append(f"| {p['short']} | {cell} | {tag} |")
    md.append("")

    md.append("## Cosine of pilot cell-mean vs nearest v4 cell\n")
    md.append("| model | pilot | nearest v4 | cos | runner-up | cos |")
    md.append("|---|---|---|---|---|---|")
    for p in payloads:
        for pcell in PILOT_CELLS:
            sub = p["cos_df"][p["cos_df"]["pilot_cell"] == pcell].sort_values(
                "cosine", ascending=False,
            )
            if len(sub) < 2:
                continue
            r0 = sub.iloc[0]
            r1 = sub.iloc[1]
            md.append(
                f"| {p['short']} | {pcell} | **{r0.v4_cell}** | {r0.cosine:.3f} | "
                f"{r1.v4_cell} | {r1.cosine:.3f} |"
            )
    md.append("")

    md.append("## Per-row nearest-v4-cell histogram\n")
    md.append("| model | cell | histogram |")
    md.append("|---|---|---|")
    for p in payloads:
        for pcell in PILOT_CELLS:
            hist = p["nearest_hist"].get(pcell, Counter())
            tag = ", ".join(f"{c}×{n}" for c, n in hist.most_common()) or "(none)"
            md.append(f"| {p['short']} | {pcell} | {tag} |")
    md.append("")

    md.append("## Read\n")
    md.append(
        "Cell-mean cosine ranks the nearest existing v4 cell; per-row "
        "nearest-cell histogram says how concentrated the pilot rows are "
        "around that cell. Convergent ranking (mean + per-row both point "
        "at the same v4 neighbor) indicates the cell is structurally "
        "miscoded into that neighbor — i.e. existing v4 hidden geometry "
        "doesn't carve a separate region for NN/LB. Divergent ranking "
        "indicates the pilot rows scatter across multiple v4 cells, which "
        "is consistent with the pilot region being genuinely off-grid "
        "(plausible structural-novelty signal). Cross-model agreement on "
        "the nearest-v4-cell amplifies whichever read.\n"
    )

    out_path.write_text("\n".join(md))
    print(f"\nwrote combined: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", default=None,
        help="Comma-separated short names; default: $LLMOJI_MODEL only.",
    )
    args = ap.parse_args()

    if args.models:
        shorts = [s.strip() for s in args.models.split(",") if s.strip()]
    else:
        shorts = [os.environ.get("LLMOJI_MODEL", "gemma")]

    for s in shorts:
        if s not in MODEL_REGISTRY:
            raise SystemExit(f"unknown model {s!r}; known: {sorted(MODEL_REGISTRY)}")

    payloads: list[dict] = []
    for short in shorts:
        print(f"\n=== {short} ===")
        payload = summarize_one_model(short)
        write_per_model_outputs(payload)
        payloads.append(payload)

    if len(payloads) > 1:
        write_combined_md(payloads)


if __name__ == "__main__":
    main()

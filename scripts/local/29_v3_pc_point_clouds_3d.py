# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false
"""v3 PC point clouds in 3D, interactive HTML.

Top-3 PCs of the h_first layer-stack rendered as a per-quadrant 3D
scatter, in native PC1/PC2/PC3 axes. One marker per row, colored by
Russell quadrant (HN bisected).

Previously this script also fit an orthogonal Procrustes rotation that
aligned the canonical probes (happy.sad / angry.calm /
fearful.unflinching) onto the canonical x/y/z axes, then drew probe
arrows + rotated PC arrows over the scatter. That whole rotation +
arrow apparatus was retired 2026-05-07 — the scatter alone reads more
cleanly without the extra trace clutter, and the Procrustes residual-
angle numbers are already captured in `findings.md`.

Per-model HTMLs at:
  figures/local/<short>/fig_v3_pc_point_clouds_3d.html
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from llmoji_experiment.config import MODEL_REGISTRY, resolve_model
from llmoji_experiment.emotional_analysis import (
    ALL_CELLS_ORDER,
    LB_LABEL,
    LB_QUADRANT,
    QUADRANT_COLORS,
    load_emotional_features_stack,
)


def _plot_one_model(
    short: str,
    pc_scores: np.ndarray,
    explained_var: np.ndarray,
    quadrants: np.ndarray,
    out_path: Path,
) -> None:
    """3D scatter of the top-3 PCs in native axes, colored by HN-split
    quadrant + OA-1 (off-axis bliss-attractor cell, when present in
    the dataset). No probe / PC arrows; point clouds only."""
    span = float(np.percentile(np.abs(pc_scores), 99)) * 1.05
    if span <= 0:
        span = 1.0

    traces: list = []
    for q in ALL_CELLS_ORDER:
        mask = quadrants == q
        if not mask.any():
            continue
        label = LB_LABEL if q == LB_QUADRANT else q
        traces.append(go.Scatter3d(
            x=pc_scores[mask, 0],
            y=pc_scores[mask, 1],
            z=pc_scores[mask, 2],
            mode="markers",
            marker=dict(
                size=4, opacity=0.7,
                color=QUADRANT_COLORS.get(q, "#666"),
                line=dict(width=0),
            ),
            name=f"{label} (n={int(mask.sum())})",
            legendgroup=f"q_{q}",
            hovertemplate=f"{label}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<br>PC3=%{{z:.2f}}<extra></extra>",
        ))

    title = (
        f"{short}: h_first layer-stack, top-3 PCs<br>"
        f"<span style='font-size:11px'>"
        f"native PCA axes; markers colored by Russell quadrant (HN bisected)"
        f"</span>"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title=f"PC1 ({explained_var[0]*100:.1f}%)", range=[-span, span]),
            yaxis=dict(title=f"PC2 ({explained_var[1]*100:.1f}%)", range=[-span, span]),
            zaxis=dict(title=f"PC3 ({explained_var[2]*100:.1f}%)", range=[-span, span]),
            aspectmode="cube",
        ),
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, b=10, t=60),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _per_model(short: str, *, pool_lb_from: str | None = None) -> bool:
    M = resolve_model(short)  # honors LLMOJI_OUT_SUFFIX for active model
    if not M.emotional_data_path.exists():
        print(f"  [{short}] no v3 data; skipping")
        return False

    print(f"\n{short}  (h_first, layer-stack)")
    df, X = load_emotional_features_stack(
        short, which="h_first", split_hn=True,
        pool_lb_from=pool_lb_from,
    )
    if len(df) == 0:
        print(f"  [{short}] no kaomoji-bearing rows")
        return False

    Xc = X - X.mean(axis=0)
    pca = PCA(n_components=3)
    pc_scores = pca.fit_transform(Xc)
    explained_var = pca.explained_variance_ratio_
    print(f"  PCA explained variance ratio: "
          f"PC1={explained_var[0]:.3f}  PC2={explained_var[1]:.3f}  "
          f"PC3={explained_var[2]:.3f}  (sum={explained_var.sum():.3f})")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = M.figures_dir / "fig_v3_pc_point_clouds_3d.html"
    quadrants = df["quadrant"].astype(str).to_numpy()
    _plot_one_model(short, pc_scores, explained_var, quadrants, out_path)
    print(f"  wrote {out_path}")
    return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=None,
        help=("Short name of one model to render. Default: iterate every "
              "key in MODEL_REGISTRY (the canonical full-registry sweep). "
              "Pass a single short like 'gemma' for the self-event chain, "
              "where only the active model has fresh data."),
    )
    parser.add_argument(
        "--pool-lb-from", default=None, metavar="SUFFIX",
        help=("LLMOJI_OUT_SUFFIX value of a sibling dataset whose OA-1 "
              "rows should be pooled into this PCA fit. Used for "
              "rendering mirror-frame point clouds with OA-1 dots from "
              "the self-event capture (typical: --pool-lb-from "
              "self_event). Layer-intersects automatically when capture "
              "depths differ. No-op when the suffixed dataset has no "
              "OA rows."),
    )
    args = parser.parse_args()

    if args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"unknown model {args.model!r}; "
                  f"known: {sorted(MODEL_REGISTRY)}")
            sys.exit(1)
        shorts: tuple[str, ...] = (args.model,)
    else:
        shorts = tuple(MODEL_REGISTRY)

    any_written = False
    for short in shorts:
        if _per_model(short, pool_lb_from=args.pool_lb_from):
            any_written = True

    if not any_written:
        print("no models produced output")


if __name__ == "__main__":
    main()

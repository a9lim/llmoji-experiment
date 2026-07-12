# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""Quadrant scatter in PCA(h_first|h_last|h_mean) space, no probe overlay.

Companion to ``22b_saklas_probes_in_pca.py``. Strips out the probe
loading + arrow overlay; just fits PCA on layer-stacked hidden states
(any of h_first / h_last / h_mean) and scatters the rows by 9-cell
quadrant. The point: see whether the cluster structure looks
different at parse-time (h_first) vs expression-time (h_last). Under
the hypothesis that v3 centroids encode "user-affect-perception"
rather than "self-affect-expression", h_last should have a
qualitatively different cluster geometry — closer to the model's own
generated state, weaker conditioning on the user-side prompt affect.

2D-only: the 3D HTML companion to this script was retired 2026-05-09
because it duplicated ``29_v3_pc_point_clouds_3d.py``'s output (same
data, same 9-cell + OA-1 colors, same top-3 PCs). For the interactive
3D view use 29's ``fig_v3_pc_point_clouds_3d.html``.

Outputs to ``figures/local/<short>/``:

* ``fig_data_pca_scatter_<which>.png``  — PC1×PC2 colored by quadrant

Usage::

    LLMOJI_MODEL=gemma python scripts/local/22d_data_pca_scatter.py
    LLMOJI_MODEL=gemma python scripts/local/22d_data_pca_scatter.py --which h_last
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from llmoji_experiment.config import DATA_DIR, current_model
from llmoji_experiment.emotional_analysis import (
    _use_cjk_font,
    apply_pad_split,
    is_kaomoji_candidate,
    pool_lb_into,
)
from llmoji_experiment.hidden_state_analysis import load_hidden_features_all_layers
from llmoji_experiment.quadrants import (
    ALL_CELLS_ORDER,
    LB_LABEL,
    LB_QUADRANT,
    QUADRANT_COLORS,
)
from llmoji.taxonomy import canonicalize_kaomoji


def _plot_2d(coords, quadrants, var, out_path: Path, short_name: str, which: str) -> None:
    import matplotlib.pyplot as plt

    _use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.4, 7.0))

    # Iterate ALL_CELLS_ORDER (= QUADRANT_ORDER_SPLIT + OA-1) so the
    # off-axis bliss-attractor cell renders as black dots alongside the
    # canonical 9-cell rainbow when the dataset contains OA rows.
    # Cells with zero rows are skipped — keeps the legend clean.
    for q in ALL_CELLS_ORDER:
        mask = quadrants == q
        if not np.any(mask):
            continue
        label = LB_LABEL if q == LB_QUADRANT else q
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=10, alpha=0.55,
            color=QUADRANT_COLORS.get(q, "#888888"),
            label=f"{label} (n={int(mask.sum())})",
            edgecolors="none",
        )

    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(
        f"{which} layer-stack PCA(2) — {short_name}\n"
        f"v4 9-cell scatter; cumulative PC1+PC2 = {(var[0]+var[1])*100:.1f}%"
    )
    ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--which", choices=("h_first", "h_last", "h_mean"), default="h_first",
        help="Which per-layer aggregate to load (default: h_first).",
    )
    parser.add_argument(
        "--n-components", type=int, default=8,
        help="Number of PCs to fit + report (default: 8).",
    )
    parser.add_argument(
        "--pool-lb-from", default=None, metavar="SUFFIX",
        help=("LLMOJI_OUT_SUFFIX value of a sibling dataset whose OA-1 "
              "rows should be pooled into this PCA fit. Used for "
              "rendering mirror-frame figures with OA-1 dots from the "
              "self-event capture (typical: --pool-lb-from self_event)."
              " Layer-intersects automatically when capture depths "
              "differ. No-op when the suffixed dataset has no OA rows."),
    )
    args = parser.parse_args()

    M = current_model()
    print(f"model: {M.short_name}  (id={M.model_id})  which={args.which}")

    if not M.emotional_data_path.exists():
        print(f"no v3 data at {M.emotional_data_path}")
        sys.exit(1)

    # Per-aggregate cache file. The shared loader honors LLMOJI_WHICH
    # only when "h_mean" is in the cache path, so we name explicitly
    # here to keep h_first / h_last / h_mean caches independent. Use
    # M.experiment (suffix-aware) rather than M.short_name (suffix-
    # blind) so LLMOJI_OUT_SUFFIX runs cache against their own data
    # rather than silently loading the canonical mirror cache —
    # validation is row-count-based (cache ≥ jsonl ⇒ valid), so a
    # short suffixed jsonl against a long mirror cache will load the
    # mirror cache and produce mirror figures in the suffixed dir.
    cache_path = (DATA_DIR / "local" / "cache"
                  / f"{M.experiment}_{args.which}_all_layers.npz")
    df, X3, layer_idxs = load_hidden_features_all_layers(
        M.emotional_data_path, DATA_DIR, M.experiment,
        which=args.which, cache_path=cache_path,
    )
    if len(df) == 0:
        print("no rows after sidecar load; aborting")
        sys.exit(1)

    df = df.assign(
        quadrant=df["prompt_id"].str[:2].str.upper(),
        first_word_raw=df["first_word"],
        first_word=df["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    mask = np.asarray([
        isinstance(s, str) and is_kaomoji_candidate(s) for s in df["first_word"]
    ])
    df = df.loc[mask].reset_index(drop=True)
    X3 = X3[mask]

    # 9-cell split (matches the rest of the v4 chain).
    df, X3 = apply_pad_split(df, X3)

    # Optional OA-1 pooling from a sibling suffixed dataset. After this,
    # X3 may have OA rows appended and a layer-intersected depth.
    if args.pool_lb_from:
        df, X3, layer_idxs = pool_lb_into(
            df, X3, layer_idxs,
            ref_short=M.short_name,
            lb_suffix=args.pool_lb_from,
            which=args.which,
        )

    n_rows, n_layers, hidden_dim = X3.shape
    print(f"  {n_rows} rows × {n_layers} layers × {hidden_dim} hidden_dim, "
          f"layers {layer_idxs[0]}..{layer_idxs[-1]}")
    X = X3.reshape(n_rows, n_layers * hidden_dim)

    n_components = min(args.n_components, X.shape[0], X.shape[1])
    print(f"\nfitting PCA(n_components={n_components})...")
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print("  explained-variance spectrum:")
    for k, v in enumerate(var, 1):
        print(f"    PC{k}: {v*100:6.2f}%  (cumulative {var[:k].sum()*100:5.2f}%)")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    quadrants = df["quadrant"].to_numpy()

    out_2d = M.figures_dir / f"fig_data_pca_scatter_{args.which}.png"
    _plot_2d(coords, quadrants, var, out_2d, M.short_name, args.which)
    print(f"\nwrote {out_2d}")


if __name__ == "__main__":
    main()

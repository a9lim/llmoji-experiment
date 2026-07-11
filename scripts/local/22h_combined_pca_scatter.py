# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false, reportOptionalMemberAccess=false
"""Combined PCA scatter — mirror and self-event in a shared basis.

Loads both frames of one model's emit data (mirror at the canonical
``data/local/<short>/`` path and self-event at the suffixed
``data/local/<short>_self_event/`` path), concatenates the layer-
stacked ``h_first`` matrices, fits a single PCA on the combined
matrix, and scatters all rows in the shared basis. Frame is encoded
by marker shape (mirror = circle, self-event = triangle); cell is
encoded by color. Per-cell centroids in the PCA-projected coordinate
space are overlaid with thin gray segments connecting each cell's
mirror centroid to its self-event centroid — visualizing the per-
cell frame-shift vector that aggregates into ``self.other`` in 22g.

OA-1 is the off-axis bliss-attractor cell (added 2026-05-09 with
self-event v4 prompts). It has no mirror counterpart, so its
centroid appears as a single star marker in the self-event frame
only and contributes no frame-shift segment.

Outputs to ``figures/local/<short>/``:

* ``fig_combined_pca_scatter_<which>.png``       — PC1×PC2 + PC2×PC3
* ``fig_combined_pca_scatter_3d_<which>.html``   — interactive PC1..3

Usage::

    LLMOJI_MODEL=gemma .venv/bin/python scripts/local/22h_combined_pca_scatter.py
    LLMOJI_MODEL=gemma .venv/bin/python scripts/local/22h_combined_pca_scatter.py --which h_last
    LLMOJI_MODEL=gemma .venv/bin/python scripts/local/22h_combined_pca_scatter.py --self-event-suffix self_event
"""

from __future__ import annotations

import argparse
import sys
import dataclasses
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.decomposition import PCA

from llmoji_experiment.config import DATA_DIR, MODEL_REGISTRY
from llmoji_experiment.emotional_analysis import (
    _use_cjk_font,
    apply_pad_split,
    is_kaomoji_candidate,
)
from llmoji_experiment.hidden_state_analysis import load_hidden_features_all_layers
from llmoji_experiment.quadrants import (
    ALL_CELLS_ORDER,
    LB_LABEL,
    LB_QUADRANT,
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
)
from llmoji.taxonomy import canonicalize_kaomoji


# OA-1 color comes from QUADRANT_COLORS via the canonical "OA" key
# (defined in quadrants.py). Kept as a local alias for readability —
# every script in the chain writes OA dots in this color.
OA_COLOR = QUADRANT_COLORS[LB_QUADRANT]


def _resolve(short: str, suffix: str | None):
    """Return ``ModelPaths`` for either the canonical mirror data
    (suffix=None) or the suffixed dataset. Direct dataclass replace
    avoids env-var bookkeeping so this script can load both frames in
    a single process without state coupling."""
    if short not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {short!r}; "
                       f"known: {sorted(MODEL_REGISTRY)}")
    M = MODEL_REGISTRY[short]
    if not suffix:
        return M
    suffixed = f"{short}_{suffix}"
    return dataclasses.replace(
        M,
        emotional_data_path=DATA_DIR / "local" / suffixed
                            / "emotional_raw.jsonl",
        emotional_summary_path=DATA_DIR / "local" / suffixed
                               / "emotional_summary.tsv",
        experiment=suffixed,
    )


def _load_frame(M, which: str):
    """Load one frame: returns (df, X3, layer_idxs) or None.

    X3 has shape ``(n_rows, n_layers, hidden_dim)``; df carries the
    9-cell-split ``quadrant`` column with OA passing through unchanged
    via apply_pad_split (which only rewrites HP/HN)."""
    if not M.emotional_data_path.exists():
        print(f"  no data at {M.emotional_data_path}")
        return None
    cache_path = (DATA_DIR / "local" / "cache"
                  / f"{M.experiment}_{which}_all_layers.npz")
    df, X3, layer_idxs = load_hidden_features_all_layers(
        M.emotional_data_path, DATA_DIR, M.experiment,
        which=which, cache_path=cache_path,
    )
    if len(df) == 0:
        return None
    df = df.assign(
        quadrant=df["prompt_id"].str[:2].str.upper(),
        first_word=df["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    mask = np.asarray([
        isinstance(s, str) and is_kaomoji_candidate(s)
        for s in df["first_word"]
    ])
    df = df.loc[mask].reset_index(drop=True)
    X3 = X3[mask]
    df, X3 = apply_pad_split(df, X3)
    return df, X3, layer_idxs


def _common_layer_view(
    X3_a: np.ndarray,
    layers_a: list[int],
    X3_b: np.ndarray,
    layers_b: list[int],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Intersect two layer indexings so the layer-stacked matrices
    can be concatenated. Returns (X3_a', X3_b', common_layers).

    Mirror data was historically captured under saklas <2.1 (56-layer
    coverage on gemma-31b); self-event v4 was captured under saklas
    2.1+ (58-layer coverage). When indices differ, the intersection
    keeps both frames in the same per-layer-position space."""
    a, b = list(layers_a), list(layers_b)
    if a == b:
        return X3_a, X3_b, a
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError(
            f"no common layers between mirror={a[:3]}.. and "
            f"self_event={b[:3]}..; cannot fit a shared basis"
        )
    idx_a = [a.index(L) for L in common]
    idx_b = [b.index(L) for L in common]
    print(f"  layer intersection: |a|={len(a)} ∩ |b|={len(b)} → "
          f"{len(common)} common layers")
    return X3_a[:, idx_a, :], X3_b[:, idx_b, :], common


def _centroids_in_pca(coords, quadrants, cells):
    """Per-cell mean of projected coords. Returns ``{cell: (n_pcs,)}``
    only for cells that actually have rows."""
    out: dict[str, np.ndarray] = {}
    for q in cells:
        mask = quadrants == q
        if np.any(mask):
            out[q] = coords[mask].mean(axis=0)
    return out


def _color_for(cell: str) -> str:
    if cell == "OA":
        return OA_COLOR
    return QUADRANT_COLORS.get(cell, "#888888")


def _label_for(cell: str) -> str:
    return LB_LABEL if cell == "OA" else cell


def _plot_2d(coords_m, q_m, coords_s, q_s, var, *,
             cent_m, cent_s, out_path, short_name, which,
             n_mirror, n_self):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _use_cjk_font()
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 7.6))

    panels = [(0, 1), (1, 2)]
    for ax, (i, j) in zip(axes, panels):
        # Mirror rows — circles, lower alpha.
        for q in QUADRANT_ORDER_SPLIT:
            mask = q_m == q
            if not np.any(mask):
                continue
            ax.scatter(
                coords_m[mask, i], coords_m[mask, j],
                s=8, alpha=0.40, marker="o",
                color=_color_for(q), edgecolors="none",
            )
        # Self-event rows — triangles, slightly higher alpha.
        for q in ALL_CELLS_ORDER:
            mask = q_s == q
            if not np.any(mask):
                continue
            ax.scatter(
                coords_s[mask, i], coords_s[mask, j],
                s=10, alpha=0.55, marker="^",
                color=_color_for(q), edgecolors="none",
            )
        # Per-cell centroid overlay + frame-shift segments.
        for q in QUADRANT_ORDER_SPLIT:
            color = _color_for(q)
            cm = cent_m.get(q)
            cs = cent_s.get(q)
            if cm is not None and cs is not None:
                ax.plot([cm[i], cs[i]], [cm[j], cs[j]],
                        color="#666666", lw=0.9, alpha=0.7, zorder=2)
            if cm is not None:
                ax.scatter(cm[i], cm[j], s=140, marker="o",
                           edgecolors="black", facecolors=color,
                           linewidths=1.0, zorder=3)
            if cs is not None:
                ax.scatter(cs[i], cs[j], s=160, marker="^",
                           edgecolors="black", facecolors=color,
                           linewidths=1.0, zorder=3)
        # OA-1 self-event centroid (no mirror counterpart) — star.
        if "OA" in cent_s:
            cs = cent_s["OA"]
            ax.scatter(cs[i], cs[j], s=240, marker="*",
                       edgecolors="black", facecolors=OA_COLOR,
                       linewidths=1.0, zorder=4)

        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
        ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
        ax.set_xlabel(f"PC{i+1} ({var[i]*100:.1f}% var)")
        ax.set_ylabel(f"PC{j+1} ({var[j]*100:.1f}% var)")
        ax.set_aspect("equal", adjustable="datalim")

    # Combined legend on the right side. Two stanzas:
    #   1) cell colors (one swatch per cell, 9-cell + OA-1)
    #   2) frame markers (circle = mirror, triangle = self-event,
    #      star = OA-1 centroid)
    cell_handles = [
        Patch(facecolor=_color_for(q), edgecolor="none", label=_label_for(q))
        for q in ALL_CELLS_ORDER
    ]
    frame_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               markeredgecolor="black", markersize=9,
               label=f"mirror row / centroid (n={n_mirror})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#666",
               markeredgecolor="black", markersize=9,
               label=f"self-event row / centroid (n={n_self})"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=OA_COLOR,
               markeredgecolor="black", markersize=14,
               label="OA-1 centroid (self-event only)"),
        Line2D([0], [0], color="#666666", lw=1.2,
               label="frame-shift segment"),
    ]
    fig.legend(handles=cell_handles, loc="center right",
               bbox_to_anchor=(0.995, 0.66), fontsize=8,
               title="cell", title_fontsize=9, frameon=True)
    fig.legend(handles=frame_handles, loc="center right",
               bbox_to_anchor=(0.995, 0.30), fontsize=8,
               title="frame", title_fontsize=9, frameon=True)

    fig.suptitle(
        f"{which} layer-stack PCA — {short_name}\n"
        f"mirror vs self-event in shared basis  "
        f"(cum PC1+PC2 = {(var[0]+var[1])*100:.1f}%)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def _plot_3d(coords_m, q_m, coords_s, q_s, var, *,
             cent_m, cent_s, out_path, short_name, which):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed; skipping 3D HTML")
        return

    fig = go.Figure()

    # Mirror rows by cell (circles).
    for q in QUADRANT_ORDER_SPLIT:
        mask = q_m == q
        if not np.any(mask):
            continue
        fig.add_trace(go.Scatter3d(
            x=coords_m[mask, 0], y=coords_m[mask, 1], z=coords_m[mask, 2],
            mode="markers",
            marker=dict(size=2.6, opacity=0.55, symbol="circle",
                        color=_color_for(q)),
            name=f"mirror {q} (n={int(mask.sum())})",
            legendgroup="mirror",
        ))
    # Self-event rows by cell (diamonds — plotly doesn't support
    # triangle-up for 3D Scatter3d markers).
    for q in ALL_CELLS_ORDER:
        mask = q_s == q
        if not np.any(mask):
            continue
        fig.add_trace(go.Scatter3d(
            x=coords_s[mask, 0], y=coords_s[mask, 1], z=coords_s[mask, 2],
            mode="markers",
            marker=dict(size=3.0, opacity=0.7, symbol="diamond",
                        color=_color_for(q)),
            name=f"self_event {_label_for(q)} (n={int(mask.sum())})",
            legendgroup="self_event",
        ))
    # Frame-shift segments (one segment per cell that has both frames).
    for q in QUADRANT_ORDER_SPLIT:
        cm = cent_m.get(q)
        cs = cent_s.get(q)
        if cm is None or cs is None:
            continue
        fig.add_trace(go.Scatter3d(
            x=[cm[0], cs[0]], y=[cm[1], cs[1]], z=[cm[2], cs[2]],
            mode="lines",
            line=dict(color="#444", width=3),
            name=f"shift {q}",
            legendgroup="shift",
            showlegend=False,
        ))
    # OA-1 self-event centroid as a large marker.
    if "OA" in cent_s:
        cs = cent_s["OA"]
        fig.add_trace(go.Scatter3d(
            x=[cs[0]], y=[cs[1]], z=[cs[2]],
            mode="markers",
            marker=dict(size=10, color=OA_COLOR, symbol="cross",
                        line=dict(color="black", width=1)),
            name="OA-1 centroid",
        ))

    fig.update_layout(
        title=f"{which} layer-stack PCA(3) — {short_name} "
              f"(cum PC1..3 = {sum(var[:3])*100:.1f}%)",
        scene=dict(
            xaxis_title=f"PC1 ({var[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({var[2]*100:.1f}%)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None,
                        help="Short name (default: $LLMOJI_MODEL or gemma).")
    parser.add_argument("--which", choices=("h_first", "h_last", "h_mean"),
                        default="h_first",
                        help="Which per-layer aggregate (default: h_first).")
    parser.add_argument("--self-event-suffix", default="self_event",
                        help="LLMOJI_OUT_SUFFIX value used when the "
                             "self-event emit ran (default: self_event).")
    parser.add_argument("--n-components", type=int, default=8,
                        help="Number of PCs to fit + report (default: 8).")
    args = parser.parse_args()

    import os
    short = args.model or os.environ.get("LLMOJI_MODEL", "gemma")
    print(f"model: {short}  which={args.which}  "
          f"self-event suffix: {args.self_event_suffix!r}")

    M_mirror = _resolve(short, suffix=None)
    M_self = _resolve(short, suffix=args.self_event_suffix)

    print(f"\n[mirror]  {M_mirror.emotional_data_path}")
    bundle_m = _load_frame(M_mirror, args.which)
    if bundle_m is None:
        print("no mirror frame data; aborting")
        sys.exit(1)
    assert bundle_m is not None  # pyright narrowing aid past sys.exit
    df_m, X3_m, layers_m = bundle_m
    print(f"  {len(df_m)} rows × {X3_m.shape[1]} layers × "
          f"{X3_m.shape[2]} hidden_dim")

    print(f"\n[self_event]  {M_self.emotional_data_path}")
    bundle_s = _load_frame(M_self, args.which)
    if bundle_s is None:
        print("no self-event frame data; aborting")
        sys.exit(1)
    assert bundle_s is not None  # pyright narrowing aid past sys.exit
    df_s, X3_s, layers_s = bundle_s
    print(f"  {len(df_s)} rows × {X3_s.shape[1]} layers × "
          f"{X3_s.shape[2]} hidden_dim")

    # Align on common layer indices, then layer-stack and concatenate.
    X3_m, X3_s, _ = _common_layer_view(
        X3_m, layers_m, X3_s, layers_s,
    )
    n_m, L, H = X3_m.shape
    n_s = X3_s.shape[0]
    X_m = X3_m.reshape(n_m, L * H)
    X_s = X3_s.reshape(n_s, L * H)
    X = np.concatenate([X_m, X_s], axis=0)

    n_components = min(args.n_components, X.shape[0], X.shape[1])
    print(f"\nfitting PCA(n_components={n_components}) on combined "
          f"{X.shape[0]} rows × {X.shape[1]} feats...")
    pca = PCA(n_components=n_components)
    pca.fit(X)
    var = pca.explained_variance_ratio_
    print("  explained-variance spectrum:")
    for k, v in enumerate(var, 1):
        print(f"    PC{k}: {v*100:6.2f}%  "
              f"(cumulative {var[:k].sum()*100:5.2f}%)")

    coords_m = pca.transform(X_m)
    coords_s = pca.transform(X_s)

    q_m = df_m["quadrant"].to_numpy()
    q_s = df_s["quadrant"].to_numpy()
    cent_m = _centroids_in_pca(coords_m, q_m, QUADRANT_ORDER_SPLIT)
    cent_s = _centroids_in_pca(coords_s, q_s, ALL_CELLS_ORDER)

    print("\nself-event row counts by cell:")
    for q in ALL_CELLS_ORDER:
        n = int((q_s == q).sum())
        if n:
            print(f"  {_label_for(q):>6s}: {n}")
    print("mirror row counts by cell:")
    for q in QUADRANT_ORDER_SPLIT:
        n = int((q_m == q).sum())
        if n:
            print(f"  {q:>6s}: {n}")

    # Mirror figures dir is the canonical home for cross-frame outputs.
    M_mirror.figures_dir.mkdir(parents=True, exist_ok=True)
    out_2d = (M_mirror.figures_dir
              / f"fig_combined_pca_scatter_{args.which}.png")
    _plot_2d(coords_m, q_m, coords_s, q_s, var,
             cent_m=cent_m, cent_s=cent_s,
             out_path=out_2d, short_name=short, which=args.which,
             n_mirror=n_m, n_self=n_s)
    print(f"\nwrote {out_2d}")

    out_3d = (M_mirror.figures_dir
              / f"fig_combined_pca_scatter_3d_{args.which}.html")
    _plot_3d(coords_m, q_m, coords_s, q_s, var,
             cent_m=cent_m, cent_s=cent_s,
             out_path=out_3d, short_name=short, which=args.which)
    if out_3d.exists():
        print(f"wrote {out_3d}")


if __name__ == "__main__":
    main()

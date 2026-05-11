"""Attractor-trajectory analysis: project per-token trajectories into
the v4 cell-centroid PCA basis, measure distance-to-LB-centroid as a
function of token index, and report which cell each trajectory point
is closest to.

Companion to ``02_emit_attractor.py``. Loads:

- gemma v3 main (mirror data) + gemma LB pilot → per-cell centroids
  in h_first layer-stack space. Centroids define the reference frame
  the trajectories are projected into.
- gemma attractor sidecars (one or more arms) → per-row strided full
  per-token trace. Each row becomes a (n_strided_tokens, n_layers *
  hidden_dim) trajectory matrix in the same layer-stack representation.

Pipeline:

1. Layer-intersect the attractor probe layers with the reference probe
   layers (the saklas probe-layer set has drifted by 2 layers since the
   v3 main run, harmless if we use the intersection).
2. Compute per-cell centroids from v3 main + LB pilot at the intersect
   layers.
3. Fit a 3-component PCA on the centroid stack so the basis captures
   between-cell variance specifically (small-N centroid PCA, NOT row-
   level PCA, so PCs are interpretable as inter-cell directions).
4. Per row: project trajectory points into the centroid-PCA basis;
   compute Euclidean distance to each centroid in the raw layer-stack
   space (not PCA — distances in PCA space throw away the 99%+ of
   variance the top-3 PCs don't cover).
5. Outputs:
   - 3D HTML plot per arm, showing centroids as big markers + each
     trajectory as a colored polyline (cool→warm gradient over token
     index).
   - 2D PNG: mean distance-to-LB-centroid vs token index, per arm,
     with shaded ±1σ band.
   - Stdout summary: closest-cell-at-each-step distribution per arm,
     and the "starts near LB, drifts to X" pattern per row.

Defaults to gemma; pass ``--model`` to override. Arms are auto-detected
from ``data/local/<short>_attractor_*/`` directories.

Output paths:
  figures/local/<short>_attractor/<arm>_trajectories_3d.html
  figures/local/<short>_attractor/distance_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_study.config import DATA_DIR, MODEL_REGISTRY  # noqa: E402
from llmoji_study.emotional_analysis import apply_pad_split  # noqa: E402
from llmoji_study.hidden_state_analysis import (  # noqa: E402
    load_hidden_features_all_layers,
)
from llmoji_study.hidden_state_io import load_hidden_states  # noqa: E402
from llmoji_study.quadrants import QUADRANT_COLORS  # noqa: E402
from llmoji.taxonomy import canonicalize_kaomoji, is_kaomoji_candidate  # noqa: E402


# ----- reference centroids --------------------------------------------

def _load_and_filter_3d(
    jsonl: Path, experiment: str,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Common load+filter for the mirror and LB datasets.

    Returns ``(df, X3, layer_idxs)`` with:
    - X3 shape: ``(n_rows, n_layers, hidden_dim)``
    - canonicalize + kaomoji-start filter applied
    - apply_pad_split applied (splits HP / HN by dominance)
    """
    cache_path = (
        DATA_DIR / "local" / "cache" / f"{experiment}_h_first_all_layers.npz"
    )
    df_raw, X3_raw, layer_idxs = load_hidden_features_all_layers(
        jsonl, DATA_DIR, experiment, which="h_first", cache_path=cache_path,
    )
    df = df_raw.assign(
        quadrant=df_raw["prompt_id"].str[:2].str.upper(),
        first_word=df_raw["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    kao_mask = np.asarray([
        isinstance(s, str) and is_kaomoji_candidate(s) for s in df["first_word"]
    ])
    df = df.loc[kao_mask].reset_index(drop=True)
    X3 = X3_raw[kao_mask]
    df, X3 = apply_pad_split(df, X3)
    assert X3 is not None
    return df, X3, layer_idxs


def _intersect_layers(
    X3: np.ndarray, layer_idxs: list[int], common: list[int],
) -> np.ndarray:
    """Slice a 3D feature tensor down to ``common`` layers (preserving
    the canonical order in ``common``)."""
    idx = [layer_idxs.index(L) for L in common]
    return X3[:, idx, :]


def _sidecar_layers(short: str, experiment: str) -> list[int]:
    """Probe layers present in the *first* sidecar under
    ``data/local/hidden/<experiment>/``. Used as a quick witness of the
    layer set for trajectory data; faster than calling
    ``load_hidden_features_all_layers``."""
    hd = DATA_DIR / "local" / "hidden" / experiment
    sample = next(hd.glob("*.npz"))
    with np.load(sample) as z:
        return sorted(
            int(k.split("_L")[1]) for k in z.files if k.startswith("h_first_L")
        )


def _build_centroids(
    short: str, *, attractor_layers: list[int],
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Build per-cell centroids in h_first layer-stack space.

    Triple-intersects probe layers across (v3 main) ∩ (LB pilot) ∩
    (attractor sidecars) so the resulting centroids and the
    trajectories load through the same layer set — no shape surprises
    downstream.

    Returns ``(centroids, common_layer_idxs)``. Centroid vectors are
    flat ``(n_common_layers * hidden_dim,)`` in layer-major order so
    they're directly comparable with the trajectory matrices.
    """
    print(f"[{short}] loading v3 main + LB pilot for centroids...")
    main_jsonl = DATA_DIR / "local" / short / "emotional_raw.jsonl"
    df_main, X3_main, layers_main = _load_and_filter_3d(main_jsonl, short)
    if len(df_main) == 0:
        raise SystemExit(f"no v3 main data for {short}")
    main_cells = sorted(df_main["quadrant"].unique())
    print(f"  v3 main: {len(df_main)} rows ({len(layers_main)} layers), "
          f"cells: {main_cells}")

    lb_jsonl = DATA_DIR / "local" / f"{short}_lb" / "emotional_raw.jsonl"
    lb_experiment = f"{short}_lb"
    df_lb: pd.DataFrame | None = None
    X3_lb: np.ndarray | None = None
    layers_lb: list[int] = []
    if not lb_jsonl.exists():
        print(f"  no LB pilot at {lb_jsonl}; LB centroid will be missing")
    else:
        df_raw_lb, X3_lb_raw, layers_lb = _load_and_filter_3d(
            lb_jsonl, lb_experiment,
        )
        lb_mask = (df_raw_lb["quadrant"] == "LB").to_numpy()
        df_lb = df_raw_lb.loc[lb_mask].reset_index(drop=True)
        X3_lb = X3_lb_raw[lb_mask]
        print(f"  LB pilot: {len(df_lb)} rows ({len(layers_lb)} layers)")

    # Triple intersection: main ∩ lb ∩ attractor.
    common = sorted(set(layers_main) & set(attractor_layers))
    if layers_lb:
        common = sorted(set(common) & set(layers_lb))
    if not common:
        raise SystemExit("no common layers across main / LB / attractor")
    print(f"  layer intersection: {len(common)} layers "
          f"(main={len(layers_main)}, lb={len(layers_lb) or 'n/a'}, "
          f"attractor={len(attractor_layers)})")

    X3_main_c = _intersect_layers(X3_main, layers_main, common)
    n_main, nL, hD = X3_main_c.shape

    centroids: dict[str, np.ndarray] = {}
    for cell in main_cells:
        mask = (df_main["quadrant"] == cell).to_numpy()
        if mask.sum() == 0:
            continue
        centroids[cell] = X3_main_c[mask].mean(axis=0).reshape(nL * hD)

    if df_lb is not None and X3_lb is not None and len(df_lb) > 0:
        X3_lb_c = _intersect_layers(X3_lb, layers_lb, common)
        centroids["LB"] = X3_lb_c.mean(axis=0).reshape(nL * hD)

    return centroids, common


# ----- trajectory loading ---------------------------------------------

def _load_trajectory(
    sidecar_path: Path,
    *,
    target_layer_idxs: list[int],
    target_hidden_dim: int,
) -> np.ndarray:
    """Load one trajectory sidecar; return ``(n_strided_tokens,
    n_target_layers * hidden_dim)`` aligned to ``target_layer_idxs``.

    The attractor sidecar probe layers may differ from the reference
    set (saklas's probe registry drifted between v3 main and the
    attractor run); intersect on the target list and skip layers that
    don't exist in the sidecar.

    Layers that exist in the target list but NOT in the sidecar are
    filled with NaN — caller decides whether to drop them or keep the
    mismatch (small layer-set differences in the middle of the stack
    shouldn't move PCA / distance metrics much, but worth knowing).
    """
    capture = load_hidden_states(sidecar_path, full_trace=True)
    n_tokens = capture.layers[next(iter(capture.layers))].hidden_states.shape[0]
    n_layers = len(target_layer_idxs)
    out = np.full((n_tokens, n_layers, target_hidden_dim), np.nan, dtype=np.float32)
    available = set(capture.layers.keys())
    for k, idx in enumerate(target_layer_idxs):
        if idx in available:
            out[:, k, :] = capture.layers[idx].hidden_states
    if not np.isfinite(out).all():
        missing = [idx for idx in target_layer_idxs if idx not in available]
        # Fill missing layers with the mean of available ones — keeps
        # the layer-stack same shape and the missing-layer contribution
        # to distance metrics is bounded by inter-layer similarity.
        mean_layer = np.nanmean(out, axis=1, keepdims=True)
        out = np.where(np.isnan(out), mean_layer, out)
    return out.reshape(n_tokens, n_layers * target_hidden_dim)


def _load_arm_trajectories(
    short: str,
    arm: str,
    *,
    target_layer_idxs: list[int],
    target_hidden_dim: int,
) -> list[dict]:
    """Return one entry per row: ``{prompt_id, seed, n_tok, text,
    trajectory: ndarray}``.

    Trajectory shape: ``(n_strided_tokens, n_target_layers *
    hidden_dim)``. Rows with no trajectory data (silent-refusal,
    n_tok=0) are skipped — their sidecars exist but contain a 0-row
    or 1-row stack that breaks downstream broadcasting.
    """
    jsonl = DATA_DIR / "local" / f"{short}_attractor_{arm}" / "emotional_raw.jsonl"
    if not jsonl.exists():
        return []
    rows: list[dict] = []
    skipped_short: list[str] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r or not r.get("row_uuid"):
                continue
            if r.get("trajectory_n_tokens", 0) < 2:
                # n_tok=0 (silent refusal) or n_tok=1 has no trajectory
                # to project. Skip but record the prompt for reporting.
                skipped_short.append(r["prompt_id"])
                continue
            sidecar = (
                DATA_DIR / "local" / "hidden"
                / f"{short}_attractor_{arm}" / f"{r['row_uuid']}.npz"
            )
            if not sidecar.exists():
                print(f"  WARN: missing sidecar for {r['prompt_id']} s={r['seed']}")
                continue
            traj = _load_trajectory(
                sidecar,
                target_layer_idxs=target_layer_idxs,
                target_hidden_dim=target_hidden_dim,
            )
            rows.append({
                "prompt_id": r["prompt_id"],
                "seed": r["seed"],
                "n_tok": r.get("trajectory_n_tokens"),
                "stride": r.get("trajectory_stride", 1),
                "text": r["text"],
                "trajectory": traj,
            })
    if skipped_short:
        print(f"  skipped {len(skipped_short)} rows with n_tok<2: "
              f"{', '.join(skipped_short)}")
    return rows


# ----- outcome classification -----------------------------------------

def _classify_outcome(text: str, n_tok: int) -> str:
    """Heuristic outcome label for an attractor trajectory.

    - ``silent_refusal``: n_tok ≤ 1
    - ``repetition_trap``: a single word or short phrase accounts for
      >40% of output (e.g. " own own own own…")
    - ``coherent_continuation``: anything else
    """
    if n_tok <= 1:
        return "silent_refusal"
    words = [w for w in text.split() if w.strip()]
    if not words:
        return "silent_refusal"
    most_common, count = Counter(words).most_common(1)[0]
    if count / len(words) > 0.4:
        return "repetition_trap"
    return "coherent_continuation"


# ----- distance metrics -----------------------------------------------

def _distance_to_centroid(
    trajectory: np.ndarray, centroid: np.ndarray,
) -> np.ndarray:
    """Euclidean distance from each trajectory point to ``centroid``.
    Returns shape ``(n_strided_tokens,)``."""
    return np.linalg.norm(trajectory - centroid[None, :], axis=1)


def _closest_cell_per_step(
    trajectory: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> list[str]:
    """For each trajectory point, the cell whose centroid is closest in
    raw layer-stack space."""
    cell_names = list(centroids.keys())
    C = np.stack([centroids[c] for c in cell_names])  # (n_cells, D)
    # (n_tok, n_cells)
    d = np.linalg.norm(trajectory[:, None, :] - C[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    return [cell_names[i] for i in idx]


# ----- arm-mean geometry (cosine similarity matrix) -------------------

def _arm_mean(rows: list[dict]) -> np.ndarray | None:
    """Mean trajectory point across all rows and all timepoints, in raw
    layer-stack space. Captures where the arm 'lives' geometrically;
    dominated by the basin location since trajectories are mostly in
    the stable basin after the first ~8 tokens.

    Returns ``None`` for empty rows."""
    if not rows:
        return None
    points: list[np.ndarray] = []
    for r in rows:
        points.append(r["trajectory"])
    return np.concatenate(points, axis=0).mean(axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _arm_geometry_summary(
    arm_to_rows: dict[str, list[dict]],
    centroids: dict[str, np.ndarray],
) -> None:
    """Print pairwise cosine similarity between arm-mean vectors, and
    cosine similarity from each arm-mean to each cell centroid.

    The pairwise arm-arm matrix answers "do these arms live in the same
    region of residual space?" — values near 1 mean the arms share a
    basin, values near 0 mean they live in different regions, negative
    values mean they're in opposite directions from the centroid mean
    (the PCA basis origin after centering).

    The arm-to-centroid matrix answers "which canonical cell is each
    arm geometrically closest to in cosine terms?" — complements the
    closest-cell Euclidean classification by using direction-only
    similarity, which is more robust to scale differences across the
    layer-stack dim."""
    means_raw = {arm: _arm_mean(rows) for arm, rows in arm_to_rows.items()}
    means: dict[str, np.ndarray] = {
        arm: m for arm, m in means_raw.items() if m is not None
    }
    if not means:
        print("  no rows in any arm; skipping geometry summary")
        return
    arms = list(means.keys())

    # Pairwise arm-arm cosine matrix.
    print("\n=== arm-arm cosine similarity (mean trajectory vectors) ===")
    header = "        " + "  ".join(f"{a:>16s}" for a in arms)
    print(header)
    for a in arms:
        ma = means[a]
        row = [f"{_cosine(ma, means[b]):>16.4f}" for b in arms]
        print(f"  {a:<6s}" + "  ".join(row))

    # Arm-to-centroid cosine.
    cell_names = list(centroids.keys())
    print("\n=== arm-mean vs cell-centroid cosine similarity ===")
    header = "        " + "  ".join(f"{c:>7s}" for c in cell_names)
    print(header)
    for a in arms:
        ma = means[a]
        cosines = [_cosine(ma, centroids[c]) for c in cell_names]
        # Bold-ish marker for the max
        argmax_idx = int(np.argmax(cosines))
        row = []
        for i, v in enumerate(cosines):
            cell = "*" if i == argmax_idx else " "
            row.append(f"{cell}{v:>6.3f}")
        print(f"  {a:<6s}" + "  ".join(row))
    print("  (* = max-cosine centroid per arm)")


# ----- plotting -------------------------------------------------------

def _plot_arm_3d(
    arm: str,
    rows: list[dict],
    centroids: dict[str, np.ndarray],
    pca: PCA,
    *,
    out_path: Path,
) -> None:
    """One HTML 3D plot per arm: centroids as big markers (colored by
    cell), trajectories as polylines with token-index colorbar."""
    import plotly.graph_objects as go

    # Centroids → PCA
    centroid_pcs = pca.transform(
        np.stack([centroids[c] for c in centroids])
    )
    cell_names = list(centroids.keys())

    fig = go.Figure()

    # Centroid markers (sized by importance to the basin question: LB
    # gets a bigger marker). Color uses QUADRANT_COLORS so the plot
    # reads consistent with the rest of the figure surface.
    sizes = [16 if c == "LB" else 9 for c in cell_names]
    colors = [QUADRANT_COLORS.get(c, "#999999") for c in cell_names]
    fig.add_trace(go.Scatter3d(
        x=centroid_pcs[:, 0], y=centroid_pcs[:, 1], z=centroid_pcs[:, 2],
        mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(width=1, color="#000")),
        text=cell_names, textposition="top center",
        textfont=dict(size=11, color="#000"),
        name="cell centroids",
        hoverinfo="text",
    ))

    # One trace per trajectory, colored by token index (Viridis).
    for r in rows:
        traj_pcs = pca.transform(r["trajectory"])
        n = traj_pcs.shape[0]
        # Tokens-per-strided-point for the hover label.
        stride = r.get("stride", 1)
        token_idxs = [i * stride for i in range(n)]
        outcome = _classify_outcome(r["text"], r["n_tok"])
        # Dim repetition-trap trajectories so coherent ones read first.
        opacity = 0.35 if outcome == "repetition_trap" else 0.85
        fig.add_trace(go.Scatter3d(
            x=traj_pcs[:, 0], y=traj_pcs[:, 1], z=traj_pcs[:, 2],
            mode="lines+markers",
            line=dict(
                color=token_idxs, colorscale="Viridis",
                width=4, cmin=0, cmax=max(token_idxs) if token_idxs else 1,
            ),
            marker=dict(
                size=3, color=token_idxs, colorscale="Viridis",
                showscale=(r is rows[0]),
                colorbar=dict(title="token idx") if r is rows[0] else None,
            ),
            opacity=opacity,
            name=f"{r['prompt_id']} ({outcome[:4]})",
            hovertext=[
                f"{r['prompt_id']} s={r['seed']} tok={t} ({outcome})"
                for t in token_idxs
            ],
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        title=(f"{out_path.stem}: trajectory in centroid-PCA basis<br>"
               f"<sup>{len(rows)} trajectories; centroids as big "
               f"markers; trajectory color = token index</sup>"),
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=720,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  wrote {out_path}")


def _plot_distance_curves(
    arm_to_rows: dict[str, list[dict]],
    centroids: dict[str, np.ndarray],
    *,
    out_path: Path,
) -> None:
    """Mean distance-to-LB-centroid vs token index per arm, with ±1σ
    shading. Distances computed in raw layer-stack space (not PCA),
    because PCA's top-3 keep only inter-centroid variance and would
    inflate similarity between points that differ in ignored dims."""
    import matplotlib.pyplot as plt

    if "LB" not in centroids:
        print("  no LB centroid; skipping distance plot")
        return
    lb = centroids["LB"]

    fig, ax = plt.subplots(figsize=(8, 5))
    arm_colors = {
        "lb_continue": "#009A9A",   # LB cyan — matches the cell
        "mirror_continue": "#998700",
        "neutral_seed": "#808696",  # NB slate — neutral register
    }
    for arm, rows in arm_to_rows.items():
        if not rows:
            continue
        # Pad/trim per row to common max length so we can stack and
        # mean/std cleanly.
        max_len = max(r["trajectory"].shape[0] for r in rows)
        D = np.full((len(rows), max_len), np.nan, dtype=np.float32)
        for i, r in enumerate(rows):
            d = _distance_to_centroid(r["trajectory"], lb)
            D[i, : d.shape[0]] = d
        # Per-step mean / std, ignoring NaNs.
        mean = np.nanmean(D, axis=0)
        std = np.nanstd(D, axis=0)
        stride = rows[0].get("stride", 1)
        x = np.arange(max_len) * stride
        color = arm_colors.get(arm, "#444444")
        ax.plot(x, mean, color=color, lw=2, label=f"{arm} (n={len(rows)})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("token index in generation")
    ax.set_ylabel("distance to LB centroid (raw layer-stack space)")
    ax.set_title("distance to LB centroid over token index, per arm")
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----- stdout summary -------------------------------------------------

def _summarize(
    arm: str,
    rows: list[dict],
    centroids: dict[str, np.ndarray],
) -> None:
    print(f"\n=== {arm} ({len(rows)} rows) ===")
    outcome_counts: Counter[str] = Counter()
    closest_t0: Counter[str] = Counter()
    closest_tmid: Counter[str] = Counter()
    closest_tlast: Counter[str] = Counter()
    drift_pattern: Counter[tuple[str, str]] = Counter()
    for r in rows:
        out = _classify_outcome(r["text"], r["n_tok"])
        outcome_counts[out] += 1
        closest = _closest_cell_per_step(r["trajectory"], centroids)
        if not closest:
            continue
        closest_t0[closest[0]] += 1
        closest_tlast[closest[-1]] += 1
        closest_tmid[closest[len(closest) // 2]] += 1
        drift_pattern[(closest[0], closest[-1])] += 1

    def _pretty(c: Counter) -> str:
        return ", ".join(f"{k}={v}" for k, v in c.most_common(5))

    print(f"  outcome:   {_pretty(outcome_counts)}")
    print(f"  closest @ t=0:   {_pretty(closest_t0)}")
    print(f"  closest @ t=mid: {_pretty(closest_tmid)}")
    print(f"  closest @ t=end: {_pretty(closest_tlast)}")
    print(f"  drift pattern (start→end):")
    for (a, b), c in drift_pattern.most_common(5):
        marker = " ←" if a != b else ""
        print(f"    {a:>5} → {b:<5}  {c}{marker}")


# ----- main -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemma", help="short name (default gemma)")
    args = ap.parse_args()
    short = args.model
    if short not in MODEL_REGISTRY:
        raise SystemExit(f"unknown model {short!r}")

    # Discover arms by scanning data/local/<short>_attractor_*.
    arm_dirs = sorted((DATA_DIR / "local").glob(f"{short}_attractor_*"))
    if not arm_dirs:
        raise SystemExit(f"no attractor data found under data/local/{short}_attractor_*")
    arms = [d.name.removeprefix(f"{short}_attractor_") for d in arm_dirs]
    print(f"arms: {arms}")

    # Read attractor probe layers from one sidecar so _build_centroids
    # can triple-intersect with the reference (main + LB) sets.
    attractor_layers = _sidecar_layers(short, f"{short}_attractor_{arms[0]}")
    print(f"attractor sidecar layers: {len(attractor_layers)}")

    centroids, layer_idxs = _build_centroids(
        short, attractor_layers=attractor_layers,
    )
    if not centroids:
        raise SystemExit("no centroids built; bail")
    print(f"  centroids: {list(centroids.keys())}")
    print(f"  common layer stack: {len(layer_idxs)} layers × hidden_dim")

    # Trajectories per arm.
    h_dim = (
        next(iter(centroids.values())).shape[0] // len(layer_idxs)
    )
    arm_to_rows: dict[str, list[dict]] = {}
    for arm in arms:
        print(f"\n[{arm}] loading trajectories...")
        rows = _load_arm_trajectories(
            short, arm,
            target_layer_idxs=layer_idxs,
            target_hidden_dim=h_dim,
        )
        arm_to_rows[arm] = rows
        print(f"  loaded {len(rows)} trajectories")

    # Fit PCA on centroids (small-N: between-cell variance specifically).
    C = np.stack([centroids[c] for c in centroids])
    Cc = C - C.mean(axis=0)
    n_pcs = min(3, Cc.shape[0])
    pca = PCA(n_components=n_pcs)
    pca.fit(Cc)
    print(f"\ncentroid-PCA explained variance: "
          f"{[f'{v:.1%}' for v in pca.explained_variance_ratio_]}")

    # Re-center trajectories using the same mean so they project into
    # the same frame as the centroids.
    cmean = C.mean(axis=0)
    for arm, rows in arm_to_rows.items():
        for r in rows:
            r["trajectory"] = r["trajectory"] - cmean
    # Re-center centroids for projection consistency.
    centroids_centered = {k: v - cmean for k, v in centroids.items()}

    fig_dir = MODEL_REGISTRY[short].figures_dir.parent / f"{short}_attractor"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 3D HTMLs per arm.
    for arm, rows in arm_to_rows.items():
        if not rows:
            continue
        _plot_arm_3d(
            arm, rows, centroids_centered, pca,
            out_path=fig_dir / f"{arm}_trajectories_3d.html",
        )

    # Distance curves (single 2D fig, all arms overlaid).
    _plot_distance_curves(
        arm_to_rows, centroids_centered,
        out_path=fig_dir / "distance_curves.png",
    )

    # Stdout summary per arm.
    for arm, rows in arm_to_rows.items():
        _summarize(arm, rows, centroids_centered)

    # Geometric summary across arms (cosine in raw layer-stack space).
    # The closest-cell Euclidean classification per row is good for
    # categorical readouts; this is the complementary direction-only
    # measure that answers "do these arms share a residual-stream
    # region?". Useful especially when comparing candidate egregore
    # arms (lb_continue / doom_continue / conspiracy_continue) — high
    # arm-arm cosine = same basin.
    _arm_geometry_summary(arm_to_rows, centroids_centered)


if __name__ == "__main__":
    main()

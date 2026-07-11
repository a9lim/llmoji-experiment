# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportReturnType=false
"""3D PC × probe rotation chart with NN + LB pilot rows + centroids overlaid.

Self-contained successor to the retired ``29_v3_pc_probe_rotation_3d.py``
(replaced 2026-05-07 by the arrow-less ``29_v3_pc_point_clouds_3d.py``).
This script keeps the rotated-frame rendering local to the NN + LB
pilot question: top-3 PCs of the v4 main h_first layer-stack, rotated
so the canonical probes (happy.sad / angry.calm / fearful.unflinching)
align with the x/y/z axes via orthogonal Procrustes — then overlays
the NN + LB pilot rows + cell centroids in the *same rotated frame*.
Pilot rows are projected through the v4-fit PCA basis so they sit
alongside v4 rows in the same subspace.

What you see:
  • v4 rows (small, faded): per-quadrant scatter from the canonical
    v4 main, exactly as in script 29 but at lower opacity to let the
    pilot points read clearly.
  • v4 cell centroids (medium, opaque): mean-of-rows per v4 cell,
    same color as the rows, with white outline.
  • NN + LB pilot rows (diamond, bright): the 10×1 pilot generations
    in distinct hot-pink (NN) / teal (LB) colors — should fall in or
    near the v4 cells they're being miscoded into per script 35's
    nearest-cell histogram.
  • NN + LB pilot centroids (diamond, large, black outline): the
    pilot cell-mean in rotated PC space. Distance from existing v4
    centroids is the structural-distinctiveness read.
  • Probe arrows (black) and rotated-PC arrows (pink): same as
    script 29.

Per-model HTMLs at:
  figures/local/<short>/fig_nn_lb_pilot_3d.html

Requires the pilot to have been run first:
  LLMOJI_MODEL=gemma python scripts/local/34_nn_lb_pilot.py
  LLMOJI_MODEL=qwen  python scripts/local/34_nn_lb_pilot.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from llmoji_experiment.config import (
    DATA_DIR,
    FIGURES_DIR,
    MODEL_REGISTRY,
    PROBES,
    resolve_model,
)
from llmoji_experiment.emotional_analysis import (
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
    load_emotional_features_stack,
    load_emotional_features_stack_at,
    load_rows,
)


# Pilot cells use marker colors distinct from the v4 palette so they
# pop out of the scatter. Matching diamond markers per pilot cell so
# they're visually disambiguated from the v4 circles.
PILOT_COLORS: dict[str, str] = {
    "NN": "#E91E63",  # hot pink — warm/negative-leaning, distinct from
                      # all v4 hues (HN-D red, HN-S purple, HP-D magenta)
    "LB": "#00ACC1",  # teal — cool/low-arousal, distinct from LN blue
                      # and LP green
}


def _attach_probe_scores(df: pd.DataFrame, jsonl_path: Path) -> pd.DataFrame:
    raw = load_rows(str(jsonl_path))
    keep_cols = ["row_uuid"] + [f"t0_{p}" for p in PROBES if f"t0_{p}" in raw.columns]
    raw = raw[keep_cols]
    return df.merge(raw, on="row_uuid", how="left")


def _probe_directions_in_pc_subspace(
    pc_scores: np.ndarray, probe_scores: np.ndarray,
) -> np.ndarray:
    n_probes = probe_scores.shape[1]
    D = np.zeros((n_probes, 3), dtype=np.float64)
    for k in range(n_probes):
        for j in range(3):
            x = probe_scores[:, k]
            y = pc_scores[:, j]
            mask = ~np.isnan(x)
            if mask.sum() < 3:
                continue
            xm = x[mask] - x[mask].mean()
            ym = y[mask] - y[mask].mean()
            denom = float(np.linalg.norm(xm) * np.linalg.norm(ym))
            if denom <= 0:
                continue
            D[k, j] = float(xm @ ym) / denom
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return D / norms


def _orthogonal_procrustes(D: np.ndarray) -> np.ndarray:
    P = D.T
    U, _, Vt = np.linalg.svd(P)
    return Vt.T @ U.T


def _build_arrow(
    origin: tuple[float, float, float],
    end: tuple[float, float, float],
    *, color: str, name: str, width: int = 4,
) -> list[go.Scatter3d]:
    line = go.Scatter3d(
        x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
        mode="lines", line=dict(color=color, width=width),
        name=name, showlegend=True, legendgroup=name, hoverinfo="name",
    )
    direction = np.array(end) - np.array(origin)
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        return [line]
    head = go.Cone(
        x=[end[0]], y=[end[1]], z=[end[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        sizemode="absolute", sizeref=norm * 0.18,
        anchor="tip", showscale=False,
        colorscale=[[0, color], [1, color]],
        name=name, showlegend=False, legendgroup=name, hoverinfo="skip",
    )
    return [line, head]


def _cell_centroids(
    rotated: np.ndarray, quadrants: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for q in np.unique(quadrants):
        mask = quadrants == q
        if not mask.any():
            continue
        out[str(q)] = rotated[mask].mean(axis=0)
    return out


def _plot_with_pilot(
    short: str,
    *,
    rotated_v4: np.ndarray,
    quadrants_v4: np.ndarray,
    rotated_pilot: np.ndarray,
    quadrants_pilot: np.ndarray,
    pilot_first_words: np.ndarray,
    pilot_prompt_ids: np.ndarray,
    R: np.ndarray,
    D: np.ndarray,
    explained_var: np.ndarray,
    out_path: Path,
    arrow_scale: float = 1.0,
) -> None:
    """Render: v4 rows (small + faded) + v4 centroids (medium) +
    pilot rows (diamond, bright) + pilot centroids (large diamond,
    black outline) + probe/PC arrows."""
    span_v4 = float(np.percentile(np.abs(rotated_v4), 99)) * 1.05
    span_pilot = (
        float(np.percentile(np.abs(rotated_pilot), 99)) * 1.05
        if len(rotated_pilot) else 0.0
    )
    span = max(span_v4, span_pilot, 1.0)

    traces: list = []

    # 1. v4 rows: small + faded.
    for q in QUADRANT_ORDER_SPLIT:
        mask = quadrants_v4 == q
        if not mask.any():
            continue
        traces.append(go.Scatter3d(
            x=rotated_v4[mask, 0],
            y=rotated_v4[mask, 1],
            z=rotated_v4[mask, 2],
            mode="markers",
            marker=dict(
                size=3, opacity=0.32,
                color=QUADRANT_COLORS.get(q, "#666"),
                line=dict(width=0),
            ),
            name=f"{q} rows (n={int(mask.sum())})",
            legendgroup=f"v4_{q}",
            hovertemplate=(
                f"{q}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>"
            ),
        ))

    # 2. v4 cell centroids: medium, white outline. Same color per cell.
    v4_centroids = _cell_centroids(rotated_v4, quadrants_v4)
    for q in QUADRANT_ORDER_SPLIT:
        if q not in v4_centroids:
            continue
        c = v4_centroids[q]
        traces.append(go.Scatter3d(
            x=[float(c[0])], y=[float(c[1])], z=[float(c[2])],
            mode="markers",
            marker=dict(
                size=10, opacity=1.0, symbol="circle",
                color=QUADRANT_COLORS.get(q, "#666"),
                line=dict(width=2, color="#FFFFFF"),
            ),
            name=f"{q} centroid",
            legendgroup=f"v4_{q}",
            showlegend=False,
            hovertemplate=(
                f"<b>{q} centroid</b><br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>"
            ),
        ))

    # 3. Pilot rows: diamond, bright color per pilot cell.
    for cell in ("NN", "LB"):
        mask = quadrants_pilot == cell
        if not mask.any():
            continue
        # Hover shows prompt_id + first_word for the row identification.
        custom = np.column_stack([
            pilot_prompt_ids[mask],
            pilot_first_words[mask],
        ])
        traces.append(go.Scatter3d(
            x=rotated_pilot[mask, 0],
            y=rotated_pilot[mask, 1],
            z=rotated_pilot[mask, 2],
            mode="markers",
            marker=dict(
                size=6, opacity=0.92, symbol="diamond",
                color=PILOT_COLORS[cell],
                line=dict(width=1, color="#000000"),
            ),
            name=f"{cell} pilot rows (n={int(mask.sum())})",
            legendgroup=f"pilot_{cell}",
            customdata=custom,
            hovertemplate=(
                f"<b>{cell}</b> %{{customdata[0]}}<br>"
                f"face: %{{customdata[1]}}<br>"
                f"x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>"
            ),
        ))

    # 4. Pilot centroids: large diamond, black outline.
    pilot_centroids = _cell_centroids(rotated_pilot, quadrants_pilot)
    for cell in ("NN", "LB"):
        if cell not in pilot_centroids:
            continue
        c = pilot_centroids[cell]
        traces.append(go.Scatter3d(
            x=[float(c[0])], y=[float(c[1])], z=[float(c[2])],
            mode="markers",
            marker=dict(
                size=16, opacity=1.0, symbol="diamond",
                color=PILOT_COLORS[cell],
                line=dict(width=3, color="#000000"),
            ),
            name=f"{cell} centroid",
            legendgroup=f"pilot_{cell}",
            hovertemplate=(
                f"<b>{cell} centroid</b><br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>"
            ),
        ))

    # 5. Probe + PC arrows (same as script 29).
    arrow_len = span * 0.85 * arrow_scale
    probe_axis_color = "#000000"
    probe_names = ["happy.sad", "angry.calm", "fearful.unflinching"]
    rotated_probes = D @ R.T
    for k, p_name in enumerate(probe_names):
        end = tuple((rotated_probes[k] * arrow_len).tolist())
        traces.extend(_build_arrow(
            (0.0, 0.0, 0.0), end,  # type: ignore[arg-type]
            color=probe_axis_color, name=f"probe: {p_name}", width=6,
        ))

    pc_color = "#cc4477"
    pc_unit = R.T
    for j in range(3):
        end_pc = tuple((pc_unit[j] * arrow_len * 0.7).tolist())
        traces.extend(_build_arrow(
            (0.0, 0.0, 0.0), end_pc,  # type: ignore[arg-type]
            color=pc_color,
            name=f"PC{j+1} ({explained_var[j]*100:.1f}%)",
            width=4,
        ))

    title = (
        f"{short}: v4 + NN/LB pilot, h_first layer-stack, "
        f"top-3 PCs rotated so probes ≈ canonical axes<br>"
        f"<span style='font-size:11px'>"
        f"x = happy.sad, y = angry.calm, z = fearful.unflinching; "
        f"v4 rows (small circles, faded) + cell centroids (large circles); "
        f"pilot rows + centroids as diamonds (NN pink, LB teal); "
        f"pink arrows = rotated PCs"
        f"</span>"
    )
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="probe: happy.sad", range=[-span, span]),
            yaxis=dict(title="probe: angry.calm", range=[-span, span]),
            zaxis=dict(title="probe: fearful.unflinching", range=[-span, span]),
            aspectmode="cube",
        ),
        legend=dict(font=dict(size=9)),
        margin=dict(l=10, r=10, b=10, t=70),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _per_model(short: str) -> dict | None:
    # 1. Load v4 main (suffix-free).
    os.environ.pop("LLMOJI_OUT_SUFFIX", None)
    M = resolve_model(short)
    if not M.emotional_data_path.exists():
        print(f"  [{short}] no v4 data; skipping")
        return None
    print(f"\n{short}  (h_first, layer-stack)")
    df_v4, X_v4 = load_emotional_features_stack(
        short, which="h_first", split_hn=True,
    )
    if len(df_v4) == 0:
        print(f"  [{short}] no kaomoji-bearing rows")
        return None
    df_v4 = _attach_probe_scores(df_v4, M.emotional_data_path)
    probe_cols = [f"t0_{p}" for p in PROBES if f"t0_{p}" in df_v4.columns]
    if len(probe_cols) < 3:
        print(f"  [{short}] missing probe columns; skipping")
        return None

    # 2. Fit PCA on v4 only (consistent with 29's behavior).
    mean_v4 = X_v4.mean(axis=0)
    Xc = X_v4 - mean_v4
    pca = PCA(n_components=3)
    pc_scores_v4 = pca.fit_transform(Xc)
    explained_var = pca.explained_variance_ratio_
    print(
        f"  PCA explained variance ratio: "
        f"PC1={explained_var[0]:.3f}  PC2={explained_var[1]:.3f}  "
        f"PC3={explained_var[2]:.3f}  (sum={explained_var.sum():.3f})"
    )

    # 3. Procrustes align probes to canonical axes.
    probe_scores = df_v4[probe_cols].to_numpy(dtype=float)
    D = _probe_directions_in_pc_subspace(pc_scores_v4, probe_scores)
    R = _orthogonal_procrustes(D)

    # 4. Load pilot, project into v4 PCA basis (same mean), apply rotation.
    os.environ["LLMOJI_OUT_SUFFIX"] = "nn_lb_pilot"
    os.environ["LLMOJI_MODEL"] = short
    M_pilot = resolve_model(short)
    if not M_pilot.emotional_data_path.exists():
        print(
            f"  [{short}] pilot JSONL missing; run scripts/local/34_nn_lb_pilot.py"
            f" first. Plotting v4 only."
        )
        df_pilot = pd.DataFrame()
        rotated_pilot = np.zeros((0, 3))
        quadrants_pilot = np.array([], dtype=str)
        pilot_first_words = np.array([], dtype=str)
        pilot_prompt_ids = np.array([], dtype=str)
    else:
        df_pilot, X_pilot = load_emotional_features_stack_at(
            M_pilot.emotional_data_path,
            DATA_DIR,
            experiment=M_pilot.experiment,
        )
        if len(df_pilot) == 0:
            print(f"  [{short}] pilot has no kaomoji-bearing rows")
            rotated_pilot = np.zeros((0, 3))
            quadrants_pilot = np.array([], dtype=str)
            pilot_first_words = np.array([], dtype=str)
            pilot_prompt_ids = np.array([], dtype=str)
        else:
            # Project pilot data through v4-fit PCA: subtract v4 mean,
            # then dot with the v4 PCA components. This is equivalent to
            # pca.transform(X_pilot - mean_v4) but we use the explicit
            # v4 mean so v4 and pilot share the centering.
            Xc_pilot = X_pilot - mean_v4
            pc_scores_pilot = pca.transform(Xc_pilot)
            rotated_pilot = pc_scores_pilot @ R.T
            quadrants_pilot = df_pilot["quadrant"].astype(str).to_numpy()
            pilot_first_words = df_pilot["first_word"].astype(str).to_numpy()
            pilot_prompt_ids = df_pilot["prompt_id"].astype(str).to_numpy()
            print(
                f"  pilot rows: {len(df_pilot)} "
                f"(NN={(quadrants_pilot=='NN').sum()}, "
                f"LB={(quadrants_pilot=='LB').sum()})"
            )

    # 5. Rotate v4 + plot.
    rotated_v4 = pc_scores_v4 @ R.T
    quadrants_v4 = df_v4["quadrant"].astype(str).to_numpy()

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = M.figures_dir / "fig_nn_lb_pilot_3d.html"
    _plot_with_pilot(
        short,
        rotated_v4=rotated_v4,
        quadrants_v4=quadrants_v4,
        rotated_pilot=rotated_pilot,
        quadrants_pilot=quadrants_pilot,
        pilot_first_words=pilot_first_words,
        pilot_prompt_ids=pilot_prompt_ids,
        R=R,
        D=D,
        explained_var=explained_var,
        out_path=out_path,
    )
    print(f"  wrote {out_path}")
    return {"short": short, "out_path": out_path}


def main() -> None:
    out_dir = FIGURES_DIR / "local"
    out_dir.mkdir(parents=True, exist_ok=True)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", default="gemma,qwen",
        help="Comma-separated short names (default: gemma,qwen — "
             "the pilot lineup).",
    )
    args = ap.parse_args()
    shorts = [s.strip() for s in args.models.split(",") if s.strip()]
    for s in shorts:
        if s not in MODEL_REGISTRY:
            raise SystemExit(
                f"unknown model {s!r}; known: {sorted(MODEL_REGISTRY)}"
            )

    written: list[Path] = []
    for short in shorts:
        result = _per_model(short)
        if result:
            written.append(result["out_path"])

    if written:
        print(f"\nwrote {len(written)} HTML(s):")
        for p in written:
            print(f"  {p}")


if __name__ == "__main__":
    main()

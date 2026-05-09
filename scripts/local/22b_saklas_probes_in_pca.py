# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""Saklas probe directions in v3 PCA space.

Companion to ``23_v3_pca3plus.py`` (which plots PC × probe-score
correlations). This script visualizes the saklas probe DIRECTION
VECTORS themselves: it fits PCA on the layer-stacked h_first
activations, then projects each probe's per-layer baked direction
into the same PCA basis and draws it as an arrow from the origin.

Why: the saklas v2.1 overhaul changed the default extractor from
contrastive-PCA to difference-of-means (DiM), with per-layer share
allocation in the Mahalanobis metric (whitened against the model's
neutral-activation covariance) instead of v1.x Euclidean magnitude.
This is a sanity-check plot — do the new probe directions still align
with the v3 PCA structure on gemma-4-31b-it (PC1 ≈ valence, PC2/PC3
carrying arousal + dominance)?

Workflow:

* Bootstraps a ``SaklasSession`` with ``probes=PROBE_CATEGORIES``
  for the active model. This forces extraction + cache write of the
  current-default-method profile under
  ``~/.saklas/vectors/default/<probe>/<safe_model_id>.safetensors``
  if not already present. Pass ``--no-bootstrap`` to skip if you've
  already extracted (saves ~5–15 min of model load).
* Loads each probe in ``PROBES`` from disk via ``saklas.load_profile``;
  zero-pads to the data's full layer set; flattens to layer-stack
  shape so the probe lives in the same space as ``h_first`` rows.
* Fits ``PCA(n_components=8)`` on layer-stacked ``h_first`` (matches
  ``23_v3_pca3plus.py``); projects probes via ``probe @ components_.T``.
* 2D PNG: PC1×PC2 scatter colored by quadrant + probe arrows from
  origin, scaled to data spread. 3D HTML: PC1×PC2×PC3 plotly with
  the same overlay (rotation reveals when a probe points more into
  PC3 than the 2D panel suggests).

Outputs to ``figures/local/<short>/``:

* ``fig_saklas_probes_in_pca.png`` — 2D matplotlib panel
* ``fig_saklas_probes_in_pca_3d.html`` — 3D plotly
* ``saklas_probes_in_pca.tsv`` — probe × PC{1..8} loadings + cosine

Method note (probe-as-direction vs probe-as-point):
PCA centers data before fitting; ``components_`` are unit eigenvectors
of the centered Gram matrix. A probe is a *direction* (displacement),
so we don't subtract the data mean from it — we just rotate it into
the PC basis via inner-product with each component. The "loading" of
a probe on PC k is therefore ``<probe, components_[k]>`` (in stacked-
fp32, in raw / unwhitened model-space units, since that's what the
data are in).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from llmoji_study.config import (
    PROBE_CATEGORIES,
    PROBES,
    current_model,
)
from llmoji_study.emotional_analysis import _use_cjk_font
from llmoji_study.hidden_state_analysis import load_hidden_features_all_layers
from llmoji_study.quadrants import QUADRANT_COLORS, QUADRANT_ORDER
from llmoji.taxonomy import canonicalize_kaomoji


def _saklas_profile_path(
    model_id: str,
    probe: str,
    *,
    method: str = "dim",
    namespace: str = "default",
) -> Path:
    """Resolve the on-disk profile path for ``probe`` under ``model_id``.

    ``namespace`` selects the vectors-tree directory: ``default`` for
    the saklas-bundled affect/agentic/etc. packs, ``llmoji`` for our
    centroid-derived probes (see ``22c_register_centroid_probes.py``),
    or ``local`` for ad-hoc user-extracted concepts. ``method`` only
    affects the saklas-internal filename suffix (``""`` for DiM,
    ``_pca`` for legacy PCA); centroid profiles always live at the
    DiM-equivalent un-suffixed name regardless of how the original
    saklas-extracted version was baked.

    Mirrors ``saklas.io.paths`` without importing it eagerly so the
    surface stays minimal.
    """
    from saklas.io.paths import tensor_filename, vectors_dir
    fname = tensor_filename(model_id, method=method)
    return vectors_dir() / namespace / probe / fname


def _bootstrap_extract(model_id: str, *, force_rebake: bool, method: str) -> None:
    """Boot a SaklasSession that auto-extracts the affect probes.

    Idempotent: saklas's extraction pipeline checks the cache and skips
    if a valid profile already exists. ``force_rebake=True`` deletes
    the existing profile files for ``PROBES`` first so the next session
    init re-extracts under the current saklas defaults — useful after
    a saklas overhaul (v2.1 DiM/Mahalanobis bake) when the on-disk
    profile predates the change.
    """
    from saklas import SaklasSession

    if force_rebake:
        for p in PROBES:
            path = _saklas_profile_path(model_id, p, method=method)
            sidecar = path.with_suffix(".json")
            for f in (path, sidecar):
                if f.exists():
                    print(f"  rebake: removing {f}")
                    f.unlink()

    print(f"  saklas bootstrap (probes={PROBE_CATEGORIES}, method={method})...")
    # ``probes=PROBE_CATEGORIES`` triggers extract-and-cache for every
    # bundled affect probe. The session itself is discarded immediately
    # — we only want the side effect of populating the vectors cache.
    sess_kwargs = {"probes": PROBE_CATEGORIES, "device": "auto"}
    if method == "pca":
        sess_kwargs["extraction_method"] = "pca"
    with SaklasSession.from_pretrained(model_id, **sess_kwargs):
        pass
    print("  bootstrap complete.")


def _load_probe_layer_stack(
    model_id: str,
    probe: str,
    layer_idxs: list[int],
    hidden_dim: int,
    *,
    method: str = "dim",
    namespace: str = "default",
) -> tuple[np.ndarray, list[int]]:
    """Load ``probe`` as a flat layer-stacked vector aligned to ``layer_idxs``.

    Returns ``(stacked_vec, covered_layers)`` where ``stacked_vec`` has
    shape ``(len(layer_idxs) * hidden_dim,)`` with zeros at any layer
    the probe does not cover. ``covered_layers`` is the subset of
    ``layer_idxs`` actually present in the saklas profile.
    """
    from saklas.core.vectors import load_profile

    path = _saklas_profile_path(model_id, probe, method=method, namespace=namespace)
    if not path.exists():
        raise FileNotFoundError(
            f"no saklas profile at {path}; rerun without --no-bootstrap"
        )
    profile, _meta = load_profile(str(path))
    # profile: dict[int, torch.Tensor[hidden_dim]]
    stacked = np.zeros((len(layer_idxs), hidden_dim), dtype=np.float32)
    covered: list[int] = []
    for i, L in enumerate(layer_idxs):
        if L in profile:
            v = profile[L].detach().cpu().to(dtype=__import__("torch").float32).numpy()
            if v.shape != (hidden_dim,):
                raise ValueError(
                    f"probe {probe!r} layer {L}: tensor shape {v.shape} "
                    f"!= expected ({hidden_dim},)"
                )
            stacked[i] = v
            covered.append(L)
    return stacked.reshape(-1), covered


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _arrow_scale(coords: np.ndarray, probe_pc: np.ndarray) -> float:
    """Choose an arrow scale so probe arrows reach ~80% of the data spread.

    ``coords`` are the PC1×PC2 scatter coordinates; ``probe_pc`` is
    ``(n_probes, 2)`` of probe loadings on PC1/PC2. Scales by the ratio
    of the 95th percentile of ``|coords|`` to the largest ``|probe_pc|``.
    Falls back to 1.0 if either is degenerate.
    """
    data_extent = float(np.percentile(np.abs(coords), 95))
    probe_extent = float(np.max(np.abs(probe_pc))) if probe_pc.size else 0.0
    if probe_extent < 1e-12 or data_extent < 1e-12:
        return 1.0
    return 0.8 * data_extent / probe_extent


def _plot_2d(
    coords: np.ndarray,
    quadrants: np.ndarray,
    probe_names: list[str],
    probe_pc: np.ndarray,  # (n_probes, n_components)
    var: np.ndarray,
    out_path: Path,
    short_name: str,
    method: str,
    which: str = "h_first",
    namespace: str = "default",
) -> None:
    import matplotlib.pyplot as plt

    _use_cjk_font()
    fig, ax = plt.subplots(figsize=(8.4, 7.0))

    # Scatter, colored by quadrant. Plot in QUADRANT_ORDER so legend is
    # consistent across figures.
    for q in QUADRANT_ORDER:
        mask = quadrants == q
        if not np.any(mask):
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=10, alpha=0.55,
            color=QUADRANT_COLORS.get(q, "#888888"),
            label=f"{q} (n={int(mask.sum())})",
            edgecolors="none",
        )

    # Probe arrows from origin.
    scale = _arrow_scale(coords[:, :2], probe_pc[:, :2])
    for name, pc in zip(probe_names, probe_pc[:, :2]):
        ax.annotate(
            "", xy=(pc[0] * scale, pc[1] * scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="black",
                            lw=1.6, alpha=0.95),
        )
        # Position the label slightly past the arrow tip.
        ax.text(
            pc[0] * scale * 1.08, pc[1] * scale * 1.08,
            name,
            fontsize=9, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white", ec="black", lw=0.5, alpha=0.85),
        )

    ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.6, zorder=0)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(
        f"saklas probes in PCA space — {short_name} "
        f"(method={method}, namespace={namespace})\n"
        f"data: {which} layer-stack; arrows: probe direction projected onto PC1×PC2 "
        f"(scale ×{scale:.3g})"
    )
    ax.legend(fontsize=8, loc="best", frameon=True, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_3d(
    coords: np.ndarray,
    quadrants: np.ndarray,
    probe_names: list[str],
    probe_pc: np.ndarray,
    var: np.ndarray,
    out_path: Path,
    short_name: str,
    method: str,
    which: str = "h_first",
    namespace: str = "default",
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  plotly not installed; skipping 3D HTML "
              "(install saklas[notebook] or plotly directly)")
        return

    fig = go.Figure()

    for q in QUADRANT_ORDER:
        mask = quadrants == q
        if not np.any(mask):
            continue
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
            mode="markers",
            marker=dict(size=3.2, opacity=0.7,
                        color=QUADRANT_COLORS.get(q, "#888888")),
            name=f"{q} (n={int(mask.sum())})",
        ))

    scale = _arrow_scale(coords[:, :3], probe_pc[:, :3])
    for name, pc in zip(probe_names, probe_pc[:, :3]):
        fig.add_trace(go.Scatter3d(
            x=[0, pc[0] * scale],
            y=[0, pc[1] * scale],
            z=[0, pc[2] * scale],
            mode="lines+markers+text",
            line=dict(color="black", width=6),
            marker=dict(size=[0, 6], color="black"),
            text=["", name],
            textposition="top center",
            textfont=dict(size=12, color="black"),
            name=f"probe: {name}",
            showlegend=True,
        ))

    fig.update_layout(
        title=(f"saklas probes in PCA(3) — {short_name} "
               f"(method={method}, namespace={namespace}, data={which}; "
               f"arrow scale ×{scale:.3g})"),
        scene=dict(
            xaxis_title=f"PC1 ({var[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({var[2]*100:.1f}%)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _kaomoji_filter_idx(first_words) -> np.ndarray:
    from llmoji_study.emotional_analysis import is_kaomoji_candidate
    return np.asarray([
        isinstance(s, str) and is_kaomoji_candidate(s) for s in first_words
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip the SaklasSession bootstrap; assume profiles are "
             "already cached on disk. Errors out if any are missing.",
    )
    parser.add_argument(
        "--rebake", action="store_true",
        help="Delete existing on-disk profiles for PROBES before "
             "bootstrap, forcing fresh extraction under current saklas "
             "defaults (DiM + Mahalanobis bake). Use after a saklas "
             "overhaul. No-op when combined with --no-bootstrap.",
    )
    parser.add_argument(
        "--method", choices=("dim", "pca"), default="dim",
        help="Saklas extraction method to load (default: dim, current). "
             "Pass 'pca' to load the legacy contrastive-PCA tensors "
             "alongside the canonical DiM file (must already be "
             "extracted at the legacy path).",
    )
    parser.add_argument(
        "--n-components", type=int, default=8,
        help="Number of PCs to fit + report (default: 8). 2D and 3D "
             "panels always use PC1/PC2 (and PC3 for 3D).",
    )
    parser.add_argument(
        "--namespace", default="default",
        help=("Saklas vectors namespace to load probes from (default: "
              "'default' — the bundled affect/agentic packs). Pass "
              "'llmoji' to load the centroid-derived probes registered "
              "by 22c_register_centroid_probes.py."),
    )
    parser.add_argument(
        "--probes", default="",
        help=("Comma-separated probe names to load + plot. Default "
              "(empty): use config.PROBES (the canonical 3-probe set "
              "used by the v3 chain). When --namespace=llmoji is "
              "passed, this should typically be a centroid set like "
              "'HP.LN,HP.LP,HND.HNS' (the 3 axis probes) or a list of "
              "the q_<QUAD> unipolar probes."),
    )
    parser.add_argument(
        "--out-tag", default="",
        help=("Filename suffix appended after the method tag, so "
              "comparison panels (e.g. centroid-axes vs saklas-affect) "
              "don't overwrite each other. Example: --out-tag axes "
              "→ fig_saklas_probes_in_pca_dim_axes.png."),
    )
    parser.add_argument(
        "--which", choices=("h_first", "h_last", "h_mean"), default="h_first",
        help=("Per-layer hidden-state aggregate to load + run PCA on "
              "(default: h_first). h_last shows the post-generation "
              "expression state, useful for read-vs-express disambig "
              "(see 2026-05-09 self-event pilot). All-layers cache is "
              "keyed by which, so different aggregates don't collide."),
    )
    args = parser.parse_args()

    # Resolve probe list. Empty --probes falls through to config.PROBES.
    probes_to_plot: list[str] = (
        [p.strip() for p in args.probes.split(",") if p.strip()]
        if args.probes else list(PROBES)
    )

    M = current_model()
    print(f"model: {M.short_name}  (id={M.model_id})")

    if not M.emotional_data_path.exists():
        print(f"no v3 data at {M.emotional_data_path}")
        sys.exit(1)

    # 1. Bootstrap saklas (extract + cache probes if missing).
    # Only valid for the bundled-affect path (namespace=default + canonical
    # PROBES). For custom probe sets or alternate namespaces (e.g. our
    # llmoji centroid probes) the profiles must already exist on disk —
    # bootstrap is a saklas-side extraction operation and doesn't create
    # arbitrary user-defined probes.
    bootstrap_ok = (
        args.namespace == "default"
        and probes_to_plot == list(PROBES)
        and not args.no_bootstrap
    )
    if bootstrap_ok:
        _bootstrap_extract(
            M.model_id,
            force_rebake=args.rebake,
            method=args.method,
        )
    else:
        print(f"  skipping saklas session (namespace={args.namespace!r}, "
              f"probes={probes_to_plot}); expecting profiles on disk")

    # 2. Load layer-stack data at the chosen aggregate.
    print(f"loading hidden-state features ({args.which}, layer-stack)...")
    from llmoji_study.config import DATA_DIR
    # Cache key tracks both the experiment slug (so LLMOJI_OUT_SUFFIX'd
    # runs cache separately) and the aggregate (so h_first / h_last
    # don't clobber each other).
    cache_path = (DATA_DIR / "local" / "cache"
                  / f"{M.experiment}_{args.which}_all_layers.npz")
    df, X3, layer_idxs = load_hidden_features_all_layers(
        M.emotional_data_path, DATA_DIR, M.experiment,
        which=args.which, cache_path=cache_path,
    )
    if len(df) == 0:
        print("no rows after sidecar load; aborting")
        sys.exit(1)

    # Apply the standard kaomoji-start row filter to match the rest of
    # the v3 analysis chain.
    df = df.assign(
        quadrant=df["prompt_id"].str[:2].str.upper(),
        first_word_raw=df["first_word"],
        first_word=df["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    mask = _kaomoji_filter_idx(df["first_word"])
    df = df.loc[mask].reset_index(drop=True)
    X3 = X3[mask]

    n_rows, n_layers, hidden_dim = X3.shape
    print(f"  {n_rows} rows × {n_layers} layers × {hidden_dim} hidden_dim")
    print(f"  layers covered: [{layer_idxs[0]} .. {layer_idxs[-1]}] "
          f"({len(layer_idxs)} captured)")
    X = X3.reshape(n_rows, n_layers * hidden_dim)

    # 3. Load probes as flat layer-stack vectors aligned to layer_idxs.
    print(f"\nloading saklas probes ({probes_to_plot}) "
          f"from namespace={args.namespace!r}...")
    probe_vecs: list[np.ndarray] = []
    coverages: list[list[int]] = []
    for p in probes_to_plot:
        v, cov = _load_probe_layer_stack(
            M.model_id, p, layer_idxs, hidden_dim,
            method=args.method, namespace=args.namespace,
        )
        probe_vecs.append(v)
        coverages.append(cov)
        cov_pct = 100.0 * len(cov) / max(1, len(layer_idxs))
        cov_range = (f"layers {cov[0]}..{cov[-1]}" if cov else "no layers")
        print(f"  {p}: {len(cov)}/{len(layer_idxs)} layers covered "
              f"({cov_pct:.1f}%) — {cov_range}; "
              f"||v||₂ = {float(np.linalg.norm(v)):.4f}")
    probe_mat = np.stack(probe_vecs)  # (n_probes, total_dim)

    # 4. Fit PCA + project both data and probes.
    n_components = min(args.n_components, X.shape[0], X.shape[1])
    print(f"\nfitting PCA(n_components={n_components}) on layer-stack X...")
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print("  explained-variance spectrum:")
    for k, v in enumerate(var, 1):
        print(f"    PC{k}: {v*100:6.2f}%  (cumulative {var[:k].sum()*100:5.2f}%)")

    # Probe loadings on each PC: <probe, components_[k]>.
    probe_pc = probe_mat @ pca.components_.T  # (n_probes, n_components)
    # Cosine of probe with each PC, for interpretability.
    probe_norm = np.linalg.norm(probe_mat, axis=1, keepdims=True)
    pc_norm = np.linalg.norm(pca.components_, axis=1, keepdims=True).T  # (1, n_pc)
    cos = probe_pc / np.maximum(probe_norm * pc_norm, 1e-12)

    print("\nprobe × PC loading (raw):")
    print(f"  {'probe':<22} " + " ".join(f"PC{k+1:>3}".rjust(9) for k in range(n_components)))
    for name, row in zip(probes_to_plot, probe_pc):
        cells = " ".join(f"{v:+9.4f}" for v in row)
        print(f"  {name:<22} {cells}")
    print("\nprobe × PC cosine:")
    print(f"  {'probe':<22} " + " ".join(f"PC{k+1:>3}".rjust(9) for k in range(n_components)))
    for name, row in zip(probes_to_plot, cos):
        cells = " ".join(f"{v:+9.4f}" for v in row)
        print(f"  {name:<22} {cells}")

    # 5. Plots + TSV.
    M.figures_dir.mkdir(parents=True, exist_ok=True)
    quadrants = df["quadrant"].to_numpy()

    # Method-tagged filenames so a DiM run and a PCA run can coexist
    # in the same figures dir without overwriting each other. --out-tag
    # appends an extra suffix so e.g. centroid-axes vs saklas-affect
    # don't collide either. --which is also included so h_first / h_last
    # comparison plots don't clobber each other; the bare default
    # ``h_first`` is omitted to keep canonical filenames stable.
    extra = f"_{args.out_tag}" if args.out_tag else ""
    which_suffix = f"_{args.which}" if args.which != "h_first" else ""
    suffix = f"_{args.method}{which_suffix}{extra}"
    out_2d = M.figures_dir / f"fig_saklas_probes_in_pca{suffix}.png"
    _plot_2d(coords, quadrants, list(probes_to_plot), probe_pc, var,
             out_2d, M.short_name, args.method,
             which=args.which, namespace=args.namespace)
    print(f"\nwrote {out_2d}")

    out_3d = M.figures_dir / f"fig_saklas_probes_in_pca_3d{suffix}.html"
    _plot_3d(coords, quadrants, list(probes_to_plot), probe_pc, var,
             out_3d, M.short_name, args.method,
             which=args.which, namespace=args.namespace)
    if out_3d.exists():
        print(f"wrote {out_3d}")

    rows = []
    for name, vec, cov, pc_row, cos_row in zip(
        probes_to_plot, probe_vecs, coverages, probe_pc, cos,
    ):
        row = {
            "probe": name,
            "namespace": args.namespace,
            "extract_method": args.method,
            "covered_layers": len(cov),
            "covered_layer_min": cov[0] if cov else None,
            "covered_layer_max": cov[-1] if cov else None,
            "vec_norm": float(np.linalg.norm(vec)),
        }
        for k in range(n_components):
            row[f"pc{k+1}_loading"] = float(pc_row[k])
            row[f"pc{k+1}_cosine"] = float(cos_row[k])
        rows.append(row)
    tsv_path = M.figures_dir / f"saklas_probes_in_pca{suffix}.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()

# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""Extract `self.other` — the meta-axis of self-frame vs other-frame.

For each of the 9 v4 cells, compute::

    delta[c] = q_<c>(self_event) - q_<c>(mirror)

then average over cells::

    self.other = mean_c delta[c]

The headline question: is there a coherent "self-vs-other frame" direction
in gemma's hidden state that's independent of the specific affect? If
yes, the per-cell deltas should be mutually aligned (high pairwise
cosine), and their mean carries the frame swap as a single steerable
direction. If no, the deltas point in different directions for
different affects, and the frame swap is affect-conditional.

Saved as a saklas profile under ``~/.saklas/vectors/llmoji/self.other/
<safe_model_id>.safetensors`` — saklas's bipolar naming convention
applies: +α steers toward `self`, -α toward `other`.

Reports:

* Per-cell delta norms — how much the centroid moved per cell
* Per-cell delta pairwise cosines — coherence of the self.other direction
* cos(self.other, axis) for valence / arousal / dominance from both
  namespaces — does the frame axis have any specific affect-axis
  component, or is it orthogonal to all of them
* The resulting self.other profile gets a pack.json so saklas's UI
  can find and steer with it like any other concept
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch

from llmoji_experiment.config import current_model
from llmoji_experiment.emotional_analysis import _use_cjk_font


_QUADS = ("hpd", "hps", "lp", "np", "hnd", "hns", "ln", "nb", "hb")
_AXES = ("hp.ln", "hp.lp", "hnd.hns")


def _load_profile(namespace: str, concept: str, model_id: str) -> dict[int, np.ndarray]:
    from saklas.core.vectors import load_profile
    from saklas.io.paths import safe_model_id, vectors_dir

    path = (vectors_dir() / namespace / concept
            / f"{safe_model_id(model_id)}.safetensors")
    profile, _meta = load_profile(str(path))
    return {
        int(L): t.detach().cpu().numpy().astype(np.float32, copy=False)
        for L, t in profile.items()
    }


def _stack_shared(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Layer-stack on the intersection of layers, in sorted order."""
    shared = sorted(set(a) & set(b))
    return np.concatenate([a[L] - b[L] for L in shared]), shared


def _stack_layers(a: dict[int, np.ndarray], layers: list[int]) -> np.ndarray:
    return np.concatenate([a[L] for L in layers])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def _save_profile(
    profile: dict[int, np.ndarray],
    *,
    concept: str,
    model_id: str,
    namespace: str,
    method: str,
    components: dict,
) -> Path:
    from saklas.core.vectors import save_profile
    from saklas.io.paths import safe_model_id, vectors_dir

    concept_dir = vectors_dir() / namespace / concept
    concept_dir.mkdir(parents=True, exist_ok=True)
    path = concept_dir / f"{safe_model_id(model_id)}.safetensors"

    profile_t = {
        int(L): torch.from_numpy(v.astype(np.float32, copy=False).copy())
        for L, v in profile.items()
    }
    save_profile(profile_t, str(path), {
        "method": method,
        "components": components,
    })
    return path


def _write_pack_json(namespace: str, concept: str, description: str) -> None:
    from saklas.io.packs import synthesize_pack_metadata
    from saklas.io.paths import vectors_dir

    concept_dir = vectors_dir() / namespace / concept
    pack = synthesize_pack_metadata(
        name=concept,
        description=description,
        tags=["llmoji-experiment", "centroid", "meta-axis", "v4-9cell"],
        version="1.0.0",
        license="CC-BY-4.0",
        recommended_alpha=0.3,
        source="llmoji-experiment/scripts/local/22g_self_other_axis.py",
        pack_dir=concept_dir,
    )
    pack.write(concept_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror-ns", default="llmoji")
    parser.add_argument("--self-event-ns", default="llmoji_self_event")
    parser.add_argument(
        "--out-namespace", default="llmoji",
        help="Where to write the resulting self.other profile (default: "
             "llmoji — sits alongside the mirror centroids and inherits "
             "the bipolar `self.other` naming convention).",
    )
    args = parser.parse_args()

    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"mirror namespace: {args.mirror_ns}")
    print(f"self-event namespace: {args.self_event_ns}")

    # 1. Load each cell from both namespaces and compute the delta.
    mirror_unipolar: dict[str, dict[int, np.ndarray]] = {}
    self_unipolar: dict[str, dict[int, np.ndarray]] = {}
    for q in _QUADS:
        mirror_unipolar[q] = _load_profile(args.mirror_ns, f"q_{q}", M.model_id)
        self_unipolar[q] = _load_profile(args.self_event_ns, f"q_{q}", M.model_id)

    # Determine the layer set common to *every* cell+namespace pairing.
    shared_set: set[int] | None = None
    for q in _QUADS:
        layers_q = set(mirror_unipolar[q]) & set(self_unipolar[q])
        shared_set = layers_q if shared_set is None else (shared_set & layers_q)
    shared = sorted(shared_set or [])
    print(f"shared layers across all 9 cells × 2 namespaces: {len(shared)} "
          f"({shared[0]}..{shared[-1]})")

    # 2. Per-cell delta = self_event centroid − mirror centroid, layer-stacked.
    deltas: dict[str, np.ndarray] = {}
    delta_per_layer: dict[str, dict[int, np.ndarray]] = {}
    for q in _QUADS:
        per_layer = {L: self_unipolar[q][L] - mirror_unipolar[q][L] for L in shared}
        delta_per_layer[q] = per_layer
        deltas[q] = np.concatenate([per_layer[L] for L in shared])

    print("\nper-cell delta norms (||self_event[c] − mirror[c]||₂):")
    for q in _QUADS:
        print(f"  q_{q:<4s}  ‖Δ‖ = {float(np.linalg.norm(deltas[q])):8.2f}")

    # 3. Pairwise cosines of the per-cell deltas — coherence diagnostic.
    print("\nper-cell delta pairwise cosines (high → coherent self.other axis):")
    print(f"  {'':>5}  " + "  ".join(f"{q:>6}" for q in _QUADS))
    cos_mat = np.zeros((len(_QUADS), len(_QUADS)), dtype=np.float32)
    for i, qi in enumerate(_QUADS):
        cells = []
        for j, qj in enumerate(_QUADS):
            c = _cosine(deltas[qi], deltas[qj])
            cos_mat[i, j] = c
            cells.append(f"{c:+6.3f}")
        print(f"  {qi:>5}  " + "  ".join(cells))

    # 4. Average delta = self.other axis.
    self_other_per_layer = {
        L: np.mean([delta_per_layer[q][L] for q in _QUADS], axis=0)
        for L in shared
    }
    self_other_stack = np.concatenate([self_other_per_layer[L] for L in shared])
    so_norm = float(np.linalg.norm(self_other_stack))
    print(f"\nself.other ‖v‖ (layer-stack on {len(shared)} shared layers) = {so_norm:.2f}")

    # Coherence: how aligned is the mean with each per-cell delta?
    print("\ncoherence: cos(self.other, delta[c]) per cell:")
    coherences: list[float] = []
    for q in _QUADS:
        c = _cosine(self_other_stack, deltas[q])
        coherences.append(c)
        print(f"  q_{q:<4s}  cos = {c:+.4f}")
    print(f"  mean coherence = {np.mean(coherences):+.4f} "
          f"(>+0.5 = coherent meta-axis; near 0 = affect-conditional)")

    # 5. Save the self.other profile.
    out_path = _save_profile(
        self_other_per_layer,
        concept="self.other",
        model_id=M.model_id,
        namespace=args.out_namespace,
        method="centroid_meta_axis",
        components={
            "construction": "mean over 9 v4 cells of "
                            "(self_event[q_<c>] - mirror[q_<c>])",
            "cells": list(_QUADS),
            "n_layers": len(shared),
            "layer_min": shared[0],
            "layer_max": shared[-1],
            "mean_coherence": float(np.mean(coherences)),
        },
    )
    print(f"\nwrote {out_path}")
    _write_pack_json(
        args.out_namespace, "self.other",
        description=(
            "Self-vs-other frame meta-axis. Constructed as the mean over "
            "9 v4 cells of (self_event_centroid − mirror_centroid) — i.e. "
            "the average shift in gemma's hidden-state direction when "
            "the affective situation is reframed from third-party "
            "(user describes someone else) to first-party (user "
            "describes the model itself), holding affect cell fixed. "
            "Constructed from llmoji-experiment/data/local/gemma + "
            "gemma_self_event v2 emit data on 2026-05-09."
        ),
    )
    print(f"wrote pack.json under {out_path.parent}")

    # 6. Cosines vs the canonical affect axes from both namespaces.
    print("\ncos(self.other, affect-axis) — does self.other carry affect content?")
    print(f"  {'axis':<10s}  {'mirror':>8s}  {'self_event':>11s}")
    rows = []
    for axis in _AXES:
        try:
            mirror_axis = _load_profile(args.mirror_ns, axis, M.model_id)
            mirror_axis_stack = _stack_layers(mirror_axis, shared)
            cos_m = _cosine(self_other_stack, mirror_axis_stack)
        except FileNotFoundError:
            cos_m = float("nan")
        try:
            self_axis = _load_profile(args.self_event_ns, axis, M.model_id)
            self_axis_stack = _stack_layers(self_axis, shared)
            cos_s = _cosine(self_other_stack, self_axis_stack)
        except FileNotFoundError:
            cos_s = float("nan")
        print(f"  {axis:<10s}  {cos_m:+8.4f}  {cos_s:+11.4f}")
        rows.append({
            "axis": axis,
            "cos_mirror": cos_m,
            "cos_self_event": cos_s,
        })

    # 7. Diagnostic plot — per-cell delta cosine heatmap.
    M.figures_dir.mkdir(parents=True, exist_ok=True)
    out_fig = M.figures_dir / "fig_self_other_axis_cos.png"
    _plot_delta_cosine_heatmap(cos_mat, list(_QUADS), out_fig, M.short_name,
                                np.mean(coherences))
    print(f"\nwrote {out_fig}")

    # TSV summary.
    summary_rows = []
    for i, q in enumerate(_QUADS):
        summary_rows.append({
            "cell": q,
            "delta_norm": float(np.linalg.norm(deltas[q])),
            "coherence_with_mean": coherences[i],
        })
    tsv_path = M.figures_dir / "self_other_axis_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"wrote {tsv_path}")


def _plot_delta_cosine_heatmap(
    cos_mat: np.ndarray,
    cells: list[str],
    out_path: Path,
    short_name: str,
    mean_coherence: float,
) -> None:
    import matplotlib.pyplot as plt

    _use_cjk_font()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"Δ q_{c}" for c in cells], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels([f"Δ q_{c}" for c in cells], fontsize=9)
    for i in range(len(cells)):
        for j in range(len(cells)):
            v = cos_mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.5 else "#222")
    cb = fig.colorbar(im, ax=ax, shrink=0.8,
                      label="cos(Δ q_i, Δ q_j)")
    cb.ax.tick_params(labelsize=8)
    ax.set_title(
        f"self-other delta coherence — {short_name}\n"
        f"Δ q_c = q_c(self_event) − q_c(mirror); high cells = "
        f"coherent self.other axis. mean coherence = {mean_coherence:+.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

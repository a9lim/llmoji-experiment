# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""Mirror-vs-self-event centroid comparison: the read-vs-express test.

Loads both the canonical mirror centroids
(``~/.saklas/vectors/llmoji/<concept>/<safe_id>.safetensors``, built
from v3/v4 user-disclosure prompts) and the 2026-05-09 self-event
centroids (``~/.saklas/vectors/llmoji_self_event/...``, from
second-person status updates addressed to the model). Computes
per-layer cosine similarity for each shared concept, restricted to
the layer indices both profiles cover (mirror was emitted under an
older saklas with 56-layer capture; self-event under v2.1 with 58).

Output: ``figures/local/<short>/fig_mirror_vs_self_event_cos.png`` —
heatmap of per-layer cosine, concept × layer; and a
``mirror_vs_self_event_cos.tsv`` summary with mean / min / max cosine
per concept.

The headline question this answers:

  Does the model have a unified affect representation regardless of
  who the affect is "about" (mirror ≈ self_event), or are mirror and
  self-event affect representations decomposable into separate
  steerable directions (mirror ⟂ self_event)?

If mean cos > ~0.5 across cells → unified representation; the v3
finding then becomes "the model represents conversational affect
robustly across speaker frames."

If mean cos near 0 → orthogonal representations; we have a
decomposable read-vs-express axis and the steering pathology a9
identified is fixed by re-extracting on the self-event set.

If mean cos near +1 on positive cells but lower on negative cells (or
vice versa) → asymmetric — the model has separate self/other
representations only for some affect types. That's the most
interesting middle-ground.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji_study.config import current_model
from llmoji_study.emotional_analysis import _use_cjk_font


# Concepts shared between the mirror and self-event namespaces. Both
# 22c runs write the same 20 concept names, so the lists are identical
# by construction; declared explicitly here so the analysis surface is
# self-documenting.
_UNIPOLAR = ("q_hpd", "q_hps", "q_lp", "q_np",
             "q_hnd", "q_hns", "q_ln", "q_nb", "q_hb")
_VS_NB = ("hpd.nb", "hps.nb", "lp.nb", "np.nb",
          "hnd.nb", "hns.nb", "ln.nb", "hb.nb")
_AXES = ("hp.ln", "hp.lp", "hnd.hns")
_ALL_CONCEPTS = _UNIPOLAR + _VS_NB + _AXES


def _load_profile(namespace: str, concept: str, model_id: str) -> dict[int, np.ndarray]:
    """Load a saklas profile and convert tensors to numpy fp32."""
    from saklas.core.vectors import load_profile
    from saklas.io.paths import safe_model_id, vectors_dir

    path = vectors_dir() / namespace / concept / f"{safe_model_id(model_id)}.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"missing {namespace}/{concept} for {model_id}: {path}")
    profile, _meta = load_profile(str(path))
    return {
        int(L): t.detach().cpu().numpy().astype(np.float32, copy=False)
        for L, t in profile.items()
    }


def _per_layer_cosine(
    mirror: dict[int, np.ndarray],
    self_event: dict[int, np.ndarray],
) -> dict[int, float]:
    """Cosine of mirror[L] vs self_event[L] for every layer L in both."""
    shared = sorted(set(mirror) & set(self_event))
    out: dict[int, float] = {}
    for L in shared:
        m = mirror[L]
        s = self_event[L]
        nm = float(np.linalg.norm(m))
        ns = float(np.linalg.norm(s))
        if nm < 1e-12 or ns < 1e-12:
            out[L] = float("nan")
            continue
        out[L] = float(m @ s / (nm * ns))
    return out


def _stack_cosine(
    mirror: dict[int, np.ndarray],
    self_event: dict[int, np.ndarray],
) -> float:
    """Cosine of the layer-stacked vectors restricted to shared layers."""
    shared = sorted(set(mirror) & set(self_event))
    if not shared:
        return float("nan")
    m_stack = np.concatenate([mirror[L] for L in shared])
    s_stack = np.concatenate([self_event[L] for L in shared])
    nm = float(np.linalg.norm(m_stack))
    ns = float(np.linalg.norm(s_stack))
    if nm < 1e-12 or ns < 1e-12:
        return float("nan")
    return float(m_stack @ s_stack / (nm * ns))


def _plot_heatmap(
    concept_layers: list[tuple[str, dict[int, float]]],
    all_layers: list[int],
    out_path: Path,
    short_name: str,
) -> None:
    """Concept × layer heatmap of per-layer cosine. Layers are columns
    (left=early, right=late), concepts are rows in the order:
    unipolar, vs-NB, axes."""
    import matplotlib.pyplot as plt

    _use_cjk_font()
    n_concepts = len(concept_layers)
    n_layers = len(all_layers)

    mat = np.full((n_concepts, n_layers), np.nan, dtype=np.float32)
    for i, (_, layer_cos) in enumerate(concept_layers):
        for j, L in enumerate(all_layers):
            v = layer_cos.get(L)
            if v is not None and not np.isnan(v):
                mat[i, j] = v

    fig, ax = plt.subplots(figsize=(0.18 * n_layers + 4.0,
                                    0.30 * n_concepts + 2.0))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(all_layers, fontsize=7, rotation=0)
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels([c for c, _ in concept_layers], fontsize=8)
    ax.set_xlabel("layer index")

    # Group separators between unipolar / vs-NB / axes blocks.
    sep1 = len(_UNIPOLAR) - 0.5
    sep2 = sep1 + len(_VS_NB)
    ax.axhline(sep1, color="black", lw=0.7)
    ax.axhline(sep2, color="black", lw=0.7)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, label="cos(mirror, self_event)")
    cb.ax.tick_params(labelsize=8)

    ax.set_title(
        f"mirror vs self-event centroid cosine — {short_name}\n"
        "per-layer; +1 = unified, 0 = orthogonal, -1 = inverted; "
        "rows: q_<cell> unipolar / <cell>.nb bipolar / axes"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mirror-ns", default="llmoji",
        help="Mirror namespace under ~/.saklas/vectors/ (default: llmoji)",
    )
    parser.add_argument(
        "--self-event-ns", default="llmoji_self_event",
        help="Self-event namespace under ~/.saklas/vectors/ (default: llmoji_self_event)",
    )
    args = parser.parse_args()

    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"comparing namespaces: {args.mirror_ns!r} vs {args.self_event_ns!r}")

    rows: list[dict] = []
    concept_layers: list[tuple[str, dict[int, float]]] = []
    all_shared_layers: set[int] = set()

    for concept in _ALL_CONCEPTS:
        try:
            mirror = _load_profile(args.mirror_ns, concept, M.model_id)
            self_event = _load_profile(args.self_event_ns, concept, M.model_id)
        except FileNotFoundError as exc:
            print(f"  {concept}: SKIP — {exc}")
            continue

        layer_cos = _per_layer_cosine(mirror, self_event)
        stack_cos = _stack_cosine(mirror, self_event)
        concept_layers.append((concept, layer_cos))
        all_shared_layers.update(layer_cos)

        vals = np.array([v for v in layer_cos.values() if not np.isnan(v)])
        if vals.size == 0:
            mean_cos = min_cos = max_cos = float("nan")
        else:
            mean_cos = float(vals.mean())
            min_cos = float(vals.min())
            max_cos = float(vals.max())

        # Norms for context.
        m_norm = float(np.linalg.norm(np.concatenate(list(mirror.values()))))
        s_norm = float(np.linalg.norm(np.concatenate(list(self_event.values()))))

        rows.append({
            "concept": concept,
            "n_mirror_layers": len(mirror),
            "n_self_event_layers": len(self_event),
            "n_shared_layers": len(layer_cos),
            "stack_cosine": stack_cos,
            "mean_layer_cos": mean_cos,
            "min_layer_cos": min_cos,
            "max_layer_cos": max_cos,
            "mirror_norm": m_norm,
            "self_event_norm": s_norm,
        })

    df = pd.DataFrame(rows)

    print("\nper-concept summary:")
    print(f"  {'concept':<10} {'shared L':>9}  {'stack-cos':>9}  "
          f"{'mean L-cos':>10}  {'min L':>7}  {'max L':>7}")
    for r in rows:
        print(f"  {r['concept']:<10} {r['n_shared_layers']:>9}  "
              f"{r['stack_cosine']:>+9.4f}  "
              f"{r['mean_layer_cos']:>+10.4f}  "
              f"{r['min_layer_cos']:>+7.4f}  {r['max_layer_cos']:>+7.4f}")

    M.figures_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = M.figures_dir / "mirror_vs_self_event_cos.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"\nwrote {tsv_path}")

    if concept_layers:
        all_layers = sorted(all_shared_layers)
        out_path = M.figures_dir / "fig_mirror_vs_self_event_cos.png"
        _plot_heatmap(concept_layers, all_layers, out_path, M.short_name)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

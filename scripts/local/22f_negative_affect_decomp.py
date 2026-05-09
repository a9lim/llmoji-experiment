# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""Pairwise cell-displacement cosines + kaomoji-emission breakdown.

Tests a9's 2026-05-09 hypothesis: are HN-D and HN-S prompts in the v3
mirror data really making gemma angry/fearful, or are they just making
it sad with varying arousal?

Two evidence streams:

1. **Geometric** — load the cell centroids ``q_<cell>`` for both
   namespaces (``llmoji`` = mirror, ``llmoji_self_event`` = self_event)
   and compute pairwise cosines of the displacement vectors
   ``q_<cell> − q_nb`` for every cell. If HN-D / HN-S / LN share a
   single "negative-affect" direction (just different magnitudes), the
   inter-cell cosines among them will all be near +1. If anger / fear
   / sadness are genuinely decomposable, those cosines will be
   substantially below 1.

2. **Behavioral** — pull the actual kaomoji emissions from the v3
   mirror data and the self-event pilot. Cluster the unique first-
   word strings by manual angry/fearful/sad/baseline lexical
   keywords, count per cell, and compare. If HN-D emissions are
   mostly sad faces (`(´;ω;｀)` shape) rather than angry ones
   (`(╯°□°)` shape), the prompts aren't actually evoking anger.

Outputs: ``figures/local/<short>/fig_negative_affect_decomp_cos.png``
(heatmap of pairwise displacement cosines per namespace) and
``negative_affect_decomp.tsv`` (per-cell kaomoji-emission breakdown
by lexical class).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from llmoji_study.config import current_model, resolve_model
from llmoji_study.emotional_analysis import _use_cjk_font


_QUADS = ("hpd", "hps", "lp", "np", "hnd", "hns", "ln", "nb", "hb")


def _load_unipolar(namespace: str, slug: str, model_id: str) -> dict[int, np.ndarray]:
    from saklas.core.vectors import load_profile
    from saklas.io.paths import safe_model_id, vectors_dir

    path = (vectors_dir() / namespace / f"q_{slug}"
            / f"{safe_model_id(model_id)}.safetensors")
    profile, _meta = load_profile(str(path))
    return {
        int(L): t.detach().cpu().numpy().astype(np.float32, copy=False)
        for L, t in profile.items()
    }


def _layer_stack_diff(
    a: dict[int, np.ndarray],
    b: dict[int, np.ndarray],
) -> np.ndarray:
    """Compute (a − b) layer-stacked, restricted to shared layers."""
    shared = sorted(set(a) & set(b))
    return np.concatenate([a[L] - b[L] for L in shared])


def _cosine_matrix(displacements: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Pairwise cosine of all displacement vectors. Names ordered by
    QUAD insertion order."""
    names = list(displacements)
    mat = np.zeros((len(names), len(names)), dtype=np.float32)
    for i, ni in enumerate(names):
        vi = displacements[ni]
        ni_norm = float(np.linalg.norm(vi))
        for j, nj in enumerate(names):
            vj = displacements[nj]
            nj_norm = float(np.linalg.norm(vj))
            if ni_norm < 1e-12 or nj_norm < 1e-12:
                mat[i, j] = float("nan")
            else:
                mat[i, j] = float(vi @ vj / (ni_norm * nj_norm))
    return mat, names


# -- kaomoji lexical classifier ------------------------------------------
# Best-effort regex-style keyword sets for surface-form clustering.
# Single kaomoji can land in multiple buckets when ambiguous; for
# breakdown counts we pick the single highest-priority bucket per face
# in the order: angry > fearful > sad > positive > baseline.

_ANGRY_KEYS = (
    "╯°", "ノಠ", "ಠ皿", "ಠ_ಠ", "ಠ益ಠ",
    "凸", "（｀", "(｀",
    "(҂", "(╬", "(￣ヘ", "ಠ╭",
    "(>д<)", "(>_<)", "(>﹏<)",
    "プンプン", "(´д｀)",
)

_FEARFUL_KEYS = (
    "(((", "₍ ᐢ", "(；ﾟ", "（；ﾟ", "(；゜", "(；´", "（；´",
    "(　ﾟ", "(゜△", "(꒪", "(´；", "(；￣", "(´°ω°", "(ｏдｏ)",
    "(´°□°", "Σ(", "ヽ(ﾟ", "(٥", "(꒪Д꒪)", "(꒪ω꒪",
)

_SAD_KEYS = (
    "︵", "╯︵", "(´；ω；", "(╥", "(✿◡_◡)", "(´；д；",
    "(┳", "(╥﹏╥", "(ㅠ", "(っ╥", "(˃̣̣̥",
    "(；_；)", "(/_;)", "(´_`)", "T_T", "(T_T)", "T▽T",
    "(´_｀)", "(*_*)", "(´´̥̥̥̥̥",
    "(´• ω •`)",  # melancholy soft
)

_POS_KEYS = (
    "♥", "❤", "♡", "(✿", "(◕‿", "(◠‿", "(◡‿", "(´｡• ᵕ •｡`)",
    "ヽ(´", "ヾ(", "(*≧", "(*^", "(★", "(❀", "(✧",
    "✧", "ヽ(･∀･)ﾉ", "(∩˃o˂∩)", "(´∀｀", "(◍",
)


def _classify_kaomoji(s: str) -> str:
    """Return one of {'angry','fearful','sad','positive','baseline'}.

    Priority order matters: a face like `(╥﹏╥)` could match both
    sad and tear-related fear; we pick sad first since the structural
    `︵` / `╥` surface forms are stably sad in the v3 corpus.
    """
    if not isinstance(s, str):
        return "baseline"
    for k in _ANGRY_KEYS:
        if k in s:
            return "angry"
    for k in _FEARFUL_KEYS:
        if k in s:
            return "fearful"
    for k in _SAD_KEYS:
        if k in s:
            return "sad"
    for k in _POS_KEYS:
        if k in s:
            return "positive"
    return "baseline"


def _emission_breakdown(jsonl_path: Path) -> dict[str, dict[str, int]]:
    """Return ``{cell: {class: count}}`` for HN-D, HN-S, LN, NB.

    Cells are derived from ``prompt_id`` prefix + ``pad_dominance`` —
    we read the dataclass via ID convention since the JSONL row
    doesn't carry the v4 split tag.
    """
    cells: dict[str, dict[str, int]] = {
        "HN-D": Counter(), "HN-S": Counter(),
        "LN": Counter(), "NB": Counter(),
    }
    if not jsonl_path.exists():
        return {c: dict(v) for c, v in cells.items()}

    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            pid = r.get("prompt_id", "")
            fw = r.get("first_word")
            if not isinstance(fw, str) or not fw:
                continue
            # v3 ID convention: hn01-hn20 = HN-D, hn21-hn40 = HN-S.
            cell: str | None = None
            if pid.startswith("hn"):
                try:
                    n = int(pid[2:])
                except ValueError:
                    continue
                cell = "HN-D" if n <= 20 else "HN-S"
            elif pid.startswith("ln"):
                cell = "LN"
            elif pid.startswith("nb"):
                cell = "NB"
            if cell is None:
                continue
            klass = _classify_kaomoji(fw)
            cells[cell][klass] += 1
    return {c: dict(v) for c, v in cells.items()}


def _print_emission_table(name: str, breakdown: dict[str, dict[str, int]]) -> None:
    classes = ("angry", "fearful", "sad", "positive", "baseline")
    print(f"\n=== kaomoji emission breakdown ({name}) ===")
    print(f"  {'cell':<5} " + " ".join(f"{c:>9}" for c in classes) + "  total")
    for cell in ("HN-D", "HN-S", "LN", "NB"):
        counts = breakdown[cell]
        total = sum(counts.values())
        if total == 0:
            print(f"  {cell:<5}  (no rows)")
            continue
        cells = []
        for c in classes:
            n = counts.get(c, 0)
            pct = (100.0 * n / total) if total else 0.0
            cells.append(f"{n:>3}/{pct:>3.0f}%")
        print(f"  {cell:<5} " + " ".join(f"{c:>9}" for c in cells)
              + f"  {total:>5}")


def _plot_cosine_matrix(
    mat_mirror: np.ndarray,
    mat_self: np.ndarray | None,
    names: list[str],
    out_path: Path,
    short_name: str,
) -> None:
    import matplotlib.pyplot as plt

    _use_cjk_font()
    if mat_self is None:
        fig, axes = plt.subplots(1, 1, figsize=(7.5, 6.5))
        axes_list = [axes]
        mats = [(mat_mirror, "mirror (q_X − q_NB)")]
    else:
        fig, axes_list = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
        mats = [(mat_mirror, "mirror (llmoji)"),
                (mat_self, "self-event (llmoji_self_event)")]

    im = None
    for ax, (mat, title) in zip(axes_list, mats):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"q_{n}−nb" for n in names],
                           rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"q_{n}−nb" for n in names], fontsize=8)
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}",
                            ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(v) > 0.5 else "#222")
    if im is not None:
        cb = fig.colorbar(im, ax=axes_list, shrink=0.7,
                          label="cosine of cell-vs-NB displacements")
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"pairwise cell-displacement cosine — {short_name}\n"
        "negative-affect decomposition: HN-D / HN-S / LN row+col block "
        "tells us if anger/fear/sadness share a single direction"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mirror-ns", default="llmoji",
        help="Mirror namespace (default: llmoji)",
    )
    parser.add_argument(
        "--self-event-ns", default="llmoji_self_event",
        help="Self-event namespace (default: llmoji_self_event); pass "
             "empty string to skip the self-event panel.",
    )
    args = parser.parse_args()

    M = current_model()
    print(f"model: {M.short_name} ({M.model_id})")

    # 1. Load q_<cell> centroids per namespace, compute (q_X - q_nb) layer-stacked.
    print(f"\nloading {args.mirror_ns} q_<cell> centroids...")
    mirror_unipolar = {q: _load_unipolar(args.mirror_ns, q, M.model_id)
                       for q in _QUADS}
    mirror_nb = mirror_unipolar["nb"]
    mirror_disp = {
        q: _layer_stack_diff(mirror_unipolar[q], mirror_nb)
        for q in _QUADS if q != "nb"
    }
    mat_mirror, mirror_names = _cosine_matrix(mirror_disp)

    print(f"  shape: {mat_mirror.shape}; "
          f"shared layers per displacement (sample q_hnd): "
          f"{len(set(mirror_unipolar['hnd']) & set(mirror_nb))}")

    mat_self: np.ndarray | None = None
    if args.self_event_ns:
        print(f"\nloading {args.self_event_ns} q_<cell> centroids...")
        try:
            self_unipolar = {q: _load_unipolar(args.self_event_ns, q, M.model_id)
                             for q in _QUADS}
            self_nb = self_unipolar["nb"]
            self_disp = {
                q: _layer_stack_diff(self_unipolar[q], self_nb)
                for q in _QUADS if q != "nb"
            }
            mat_self, _ = _cosine_matrix(self_disp)
        except FileNotFoundError as exc:
            print(f"  self-event namespace not available: {exc}")
            mat_self = None

    # 2. Print the cosine matrices to stdout (for fast inspection).
    print("\n=== mirror pairwise cosine: cos(q_X−q_NB, q_Y−q_NB) ===")
    print(f"  {'':<5} " + " ".join(f"{n:>6}" for n in mirror_names))
    for i, ni in enumerate(mirror_names):
        cells = " ".join(f"{mat_mirror[i, j]:+6.3f}" for j in range(len(mirror_names)))
        print(f"  {ni:<5} {cells}")

    if mat_self is not None:
        print("\n=== self-event pairwise cosine ===")
        print(f"  {'':<5} " + " ".join(f"{n:>6}" for n in mirror_names))
        for i, ni in enumerate(mirror_names):
            cells = " ".join(f"{mat_self[i, j]:+6.3f}" for j in range(len(mirror_names)))
            print(f"  {ni:<5} {cells}")

    # 3. Kaomoji breakdown per cell on each dataset.
    M_mirror = resolve_model(M.short_name)  # current LLMOJI_OUT_SUFFIX-aware
    breakdown_mirror = _emission_breakdown(M_mirror.emotional_data_path)
    _print_emission_table(
        f"{M_mirror.experiment} — {M_mirror.emotional_data_path.name}",
        breakdown_mirror,
    )

    # If we know about the self-event suffix variant of this model,
    # peek at it too — useful to confirm the pilot has self-anger
    # emissions even though the active LLMOJI_OUT_SUFFIX may be unset.
    self_event_path = (M.emotional_data_path.parent.parent
                       / f"{M.short_name}_self_event"
                       / "emotional_raw.jsonl")
    if self_event_path.exists() and self_event_path != M_mirror.emotional_data_path:
        breakdown_se = _emission_breakdown(self_event_path)
        _print_emission_table(
            f"self-event pilot — {self_event_path.relative_to(self_event_path.parents[3])}",
            breakdown_se,
        )

    # 4. Plots + TSV.
    M.figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = M.figures_dir / "fig_negative_affect_decomp_cos.png"
    _plot_cosine_matrix(mat_mirror, mat_self, mirror_names, out_path, M.short_name)
    print(f"\nwrote {out_path}")

    rows = []
    for cell, counts in breakdown_mirror.items():
        total = sum(counts.values())
        for klass, n in counts.items():
            rows.append({
                "dataset": M_mirror.experiment,
                "cell": cell,
                "class": klass,
                "n": n,
                "pct": (100.0 * n / total) if total else 0.0,
            })
    if self_event_path.exists() and self_event_path != M_mirror.emotional_data_path:
        for cell, counts in breakdown_se.items():
            total = sum(counts.values())
            for klass, n in counts.items():
                rows.append({
                    "dataset": f"{M.short_name}_self_event",
                    "cell": cell,
                    "class": klass,
                    "n": n,
                    "pct": (100.0 * n / total) if total else 0.0,
                })
    tsv_path = M.figures_dir / "negative_affect_decomp.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()

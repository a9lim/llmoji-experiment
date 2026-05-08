"""Per-face Claude-modal-quadrant ground truth from the groundtruth runs.

Used by scripts 53 (subset search) and 56 (ensemble predict) when run
with ``--claude-gt``: replaces the pooled ``empirical_majority_quadrant``
(v3 + Claude + wild emit counts) with a Claude-only modal label.

Why: when the goal is to predict Claude's faces in production, GT
should be Claude's own modal quadrant — not a pooled measure that
mostly reflects v3 prompt distribution.

Layout (post 2026-05-08 merged-file refactor):

    data/harness/claude/
        emotional_raw.jsonl        # naturalistic arm (v3 + v4-ext cells merged)
        emotional_summary.tsv      # rolling summary, recomputed by script 10
    data/harness/claude_intro_v7/
        emotional_raw.jsonl        # introspection arm (v7 preamble)
        emotional_summary.tsv

Each row carries a ``run_index`` field stamped at write time. Sequential
runs share one file; saturation analysis groups by run_index. Mirrors
the local ``data/local/<model>/emotional_raw.jsonl`` convention so
harness and local pipelines have full schema parity.

Pre-2026-05-08 layout used per-run ``run-N.jsonl`` files under
``claude-runs/`` / ``claude-runs-introspection/`` / ``claude-runs-v4-extension/``.
The migration is one-shot (see ``scripts/harness/migrate_to_merged_emotional_raw.py``).

Note on canonicalization: the merged file stores the raw extracted
``first_word`` (no canonicalization), so loading this map requires
running ``canonicalize_kaomoji`` to match the keys used in the
face_likelihood summary TSVs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from llmoji.taxonomy import canonicalize_kaomoji

from llmoji_study.config import DATA_DIR

# ---------------------------------------------------------------------------
# Paths.
# ---------------------------------------------------------------------------

CLAUDE_GT_DIR = DATA_DIR / "harness" / "claude"
CLAUDE_GT_INTRO_DIR = DATA_DIR / "harness" / "claude_intro_v7"

EMOTIONAL_RAW_FILENAME = "emotional_raw.jsonl"
EMOTIONAL_SUMMARY_FILENAME = "emotional_summary.tsv"


def claude_emotional_raw_path(condition_dir: Path | None = None) -> Path:
    """Path to the merged emotional_raw.jsonl for a given condition.

    ``condition_dir=None`` resolves to the naturalistic arm
    (``CLAUDE_GT_DIR``). Pass ``CLAUDE_GT_INTRO_DIR`` for the
    introspection arm, or any other dir for forwards-compat with new
    arms.
    """
    d = condition_dir if condition_dir is not None else CLAUDE_GT_DIR
    return d / EMOTIONAL_RAW_FILENAME


def claude_emotional_summary_path(condition_dir: Path | None = None) -> Path:
    """Path to the rolling emotional_summary.tsv for a given condition."""
    d = condition_dir if condition_dir is not None else CLAUDE_GT_DIR
    return d / EMOTIONAL_SUMMARY_FILENAME


# ---------------------------------------------------------------------------
# Row loading.
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read one jsonl, skipping blank lines and error rows."""
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            out.append(r)
    return out


def load_emotional_raw(
    condition_dir: Path | None = None,
    *,
    run_index: int | None = None,
    up_to_index: int | None = None,
) -> list[dict]:
    """Load rows from the merged emotional_raw.jsonl.

    ``condition_dir``: defaults to ``CLAUDE_GT_DIR`` (naturalistic).
    Pass ``CLAUDE_GT_INTRO_DIR`` for introspection.

    ``run_index``: if given, only rows with that run_index are returned.
    ``up_to_index``: if given, only rows with run_index ≤ this. Useful
    for "what did we know before run N+1?" backtesting. Mutually
    exclusive with ``run_index``.

    Error rows (``"error" in row``) are skipped. The merged file always
    stamps a ``run_index`` field; legacy rows missing it are treated as
    run_index=0 for compatibility with the migration's intermediate
    states (should never happen in practice once migration completes).
    """
    if run_index is not None and up_to_index is not None:
        raise ValueError("pass at most one of run_index, up_to_index")
    rows = _read_jsonl(claude_emotional_raw_path(condition_dir))
    if run_index is not None:
        return [r for r in rows if int(r.get("run_index", 0)) == run_index]
    if up_to_index is not None:
        return [r for r in rows if int(r.get("run_index", 0)) <= up_to_index]
    return rows


def find_run_indices(condition_dir: Path | None = None) -> list[int]:
    """Sorted unique run_index values present in the merged file.

    Empty list if the file is missing or has no rows."""
    rows = _read_jsonl(claude_emotional_raw_path(condition_dir))
    seen = sorted({int(r.get("run_index", 0)) for r in rows})
    return seen


def latest_run_index(condition_dir: Path | None = None) -> int:
    """Highest run_index present, or ``-1`` if none."""
    indices = find_run_indices(condition_dir)
    return indices[-1] if indices else -1


def load_all_run_rows(
    condition_dir: Path | None = None,
    *,
    up_to_index: int | None = None,
) -> list[dict]:
    """Backwards-compat alias for ``load_emotional_raw(condition_dir,
    up_to_index=up_to_index)``. Older code paths hardcoded this name."""
    return load_emotional_raw(condition_dir, up_to_index=up_to_index)


# ---------------------------------------------------------------------------
# Per-face per-quadrant counts.
# ---------------------------------------------------------------------------


def _load_face_per_quadrant_counts(
    pilot_path: Path | None = None,
    *,
    condition_dir: Path | None = None,
    up_to_index: int | None = None,
    include_introspection: bool = True,
) -> dict[str, Counter[str]]:
    """Internal: build {canonical_face: Counter(quadrant -> emit_count)}
    from the merged emotional_raw.jsonl.

    ``pilot_path``: deprecated single-file mode kept for tests. Reads
    that file directly without dir-level merging.

    ``condition_dir``: defaults to naturalistic (``CLAUDE_GT_DIR``).
    Naturalistic includes both v3 cells (HP/LP/HN-D/HN-S/LN/NB) and
    v4-extension cells (HP-D/NP/HB) — they live in the same file post
    2026-05-08 merged-file refactor.

    ``include_introspection``: when True (default), also pool the
    introspection-arm rows from ``CLAUDE_GT_INTRO_DIR``. The
    introspection arm uses a different preamble but the same affective
    prompts, so the per-face per-quadrant emission counts are validly
    poolable for distribution-shape estimation. Set False for a
    "naturalistic-only, deployment-shaped" GT.
    """
    if pilot_path is not None:
        rows = _read_jsonl(pilot_path)
    else:
        rows = load_emotional_raw(condition_dir, up_to_index=up_to_index)
        if include_introspection and condition_dir != CLAUDE_GT_INTRO_DIR:
            intro_rows = load_emotional_raw(
                CLAUDE_GT_INTRO_DIR, up_to_index=up_to_index,
            )
            rows = rows + intro_rows

    # v4 quadrant remap (2026-05-07): rows store the quadrant label that
    # was current when each row was generated — for the original v3
    # naturalistic + introspection runs that's the v3 6-cell taxonomy
    # ({HP, LP, HN-D, HN-S, LN, NB}). The v4 schema bisects HP into
    # HP-D / HP-S via pad_dominance: hp01-20 (the only HP prompts Claude
    # saw under v3) are pad_dominance=-1 → HP-S in v4. Resolve via the
    # registry on read so downstream JSD math compares 9-cell encoder
    # distributions to a 9-cell GT (with zero mass on cells Claude
    # wasn't prompted on, so they shouldn't appear).
    from .emotional_analysis import _pad_split_map
    pad_split = _pad_split_map()

    counts: dict[str, Counter[str]] = {}
    for r in rows:
        f = str(r.get("first_word", ""))
        q = str(r.get("quadrant", ""))
        pid = str(r.get("prompt_id", ""))
        if not f or not q:
            continue
        # Apply v4 dominance split when the registry has a tag for this
        # prompt_id (HP, HN). Other quadrants pass through unchanged.
        q_v4 = pad_split.get(pid, q)
        f_canon = canonicalize_kaomoji(f)
        counts.setdefault(f_canon, Counter())[q_v4] += 1
    return counts


def load_claude_gt(
    pilot_path: Path | None = None,
    *,
    floor: int = 1,
    condition_dir: Path | None = None,
    up_to_index: int | None = None,
    include_introspection: bool = True,
) -> dict[str, tuple[str, int]]:
    """Return ``{canonical_face: (modal_quadrant, modal_n_emits)}``.

    Hard-modal mode. Kept for backwards compatibility and for production
    use where a deployed plugin emits a single quadrant call. For
    distribution-vs-distribution evaluation (the primary post-hoc
    metric since 2026-05-04) use ``load_claude_gt_distribution``.

    Sources rows by precedence:
      1. ``pilot_path`` if given (deprecated single-file mode — used
         only by callers that haven't migrated to the merged layout).
      2. Otherwise, the merged emotional_raw.jsonl in
         ``condition_dir`` (defaults to ``CLAUDE_GT_DIR``).

    ``floor``: faces with ``modal_n_emits < floor`` are excluded.
    Default ``floor=1`` includes every face Claude emitted at least
    once in the modal quadrant; ``floor=2`` requires ≥2 emits.

    ``up_to_index``: only meaningful in merged-file mode — pretend
    rows with ``run_index > N`` don't exist. For "what would the GT
    have looked like before adding run N+1?" backtesting.
    """
    counts = _load_face_per_quadrant_counts(
        pilot_path,
        condition_dir=condition_dir,
        up_to_index=up_to_index,
        include_introspection=include_introspection,
    )
    out: dict[str, tuple[str, int]] = {}
    for face, qmap in counts.items():
        modal_q, modal_n = qmap.most_common(1)[0]
        if modal_n >= floor:
            out[face] = (modal_q, modal_n)
    return out


def load_claude_gt_distribution(
    pilot_path: Path | None = None,
    *,
    floor: int = 3,
    condition_dir: Path | None = None,
    up_to_index: int | None = None,
    include_introspection: bool = True,
) -> dict[str, dict[str, int]]:
    """Return ``{canonical_face: {quadrant: raw_emit_count}}``.

    Distribution mode — the primary post-hoc evaluation surface since
    the 2026-05-04 soft-everywhere methodology shift. Returns raw
    counts; the consumer normalizes (with smoothing as appropriate).
    Faces with total emit count < ``floor`` are excluded — sparse
    counts (1-2 emits) give very noisy distribution estimates.
    Default ``floor=3`` matches the modal-stability threshold used
    elsewhere in the project (script 25's ``min_emits=3``).

    Use with ``llmoji_study.jsd.normalize`` + ``js`` for the
    distribution-vs-distribution comparison.

    ``include_introspection``: when True (default), pools the
    introspection-arm rows (``CLAUDE_GT_INTRO_DIR``) into the GT
    alongside naturalistic. Both arms emit kaomoji on the same
    affective prompts; pooling is honest for distribution-shape
    estimation. Set False if you want a deployment-shaped GT (no
    primed emissions).
    """
    counts = _load_face_per_quadrant_counts(
        pilot_path,
        condition_dir=condition_dir,
        up_to_index=up_to_index,
        include_introspection=include_introspection,
    )
    out: dict[str, dict[str, int]] = {}
    for face, qmap in counts.items():
        total = sum(qmap.values())
        if total < floor:
            continue
        out[face] = dict(qmap)
    return out

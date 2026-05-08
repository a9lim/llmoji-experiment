# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false, reportMissingImports=false
"""Claude ground-truth runs — naturalistic + introspection emission.

Pre-registration: ``docs/2026-05-04-claude-groundtruth-pilot.md``
(original v3 pilot) + ``docs/2026-05-07-claude-gt-v4-extension-pilot.md``
(v4-extension cells) + the appended "Sequential-run scaling protocol".

Collects ground-truth Claude (Opus 4.7) kaomoji emissions under
naturalistic single-turn calls — no disclosure preamble, no research
framing, just v3's ``KAOMOJI_INSTRUCTION`` + the affective prompt.
The introspection arm replaces ``KAOMOJI_INSTRUCTION`` with
``INTROSPECTION_PREAMBLE`` (v7).

Storage layout (post 2026-05-08 merged-file refactor — mirrors the
local ``data/local/<model>/emotional_raw.jsonl`` convention):

    data/harness/claude/emotional_raw.jsonl              # naturalistic (v3 + v4-ext)
    data/harness/claude_intro_v7/emotional_raw.jsonl     # introspection (v7 preamble)

Each row carries ``run_index`` stamped at write time. The v3 6-cell set
(HP / LP / HN-D / HN-S / LN / NB) and the v4-extension 3-cell set
(HP-D / NP / HB) live in the same naturalistic file; the cell set is
a property of which prompts get queried, not which file rows go to.

Two run modes:

  --run-index 0 (block-staged, v3 only):
    Block A (unconditional)  — HP / LP / NB × 20 prompts × 1 gen = 60 gens
    Block B (negative scout) — HN-D / HN-S / LN × 5 prompts × 1 gen = 15 gens
    Block C (gated)          — HN-D / HN-S / LN × 15 remaining × 1 gen = 45 gens
    Block C is gated on Block B's refusal rate (>25% on n=15 → halt).
    This is the original pilot design. v4-new always runs single-block.

  --run-index N (N>0, single-block):
    All in-scope quadrants × 20 prompts × 1 gen, run as one block.
    Welfare gate is no longer the staged refusal scout — it's the
    saturation comparison against runs 0..N-1, run by
    ``scripts/harness/10_emit_analysis.py`` *between* runs.

  --fill-gaps (one-shot backfill, any cells / preamble):
    Scan the merged file. For each (run_index, quadrant) tuple already
    present, expected = full 20-prompt slate for that bucket. Emit
    any missing prompt_ids. Respects saturation drops naturally — a
    quadrant absent from a given run is not refilled.

Stateless single-turn. Sampling: ``temperature=1.0``, ``max_tokens=16``
(production-faithful). Resumable: re-running a block / run skips
already-completed rows by (run_index, prompt_id). Errored rows for
the active (run_index, cell-set) are stripped on resume and retried.

Usage:
  export ANTHROPIC_API_KEY=...
  # Sequential run (default behavior post-pilot):
  python scripts/harness/00_emit.py --run-index 1
  # Then check saturation:
  python scripts/harness/10_emit_analysis.py
  # If verdict=CONTINUE, the next run:
  python scripts/harness/00_emit.py --run-index 2
  # ...

  # Original block-staged pilot (--run-index 0):
  python scripts/harness/00_emit.py --run-index 0 --block a
  python scripts/harness/00_emit.py --run-index 0 --block b
  python scripts/harness/00_emit.py --run-index 0 --check-gate
  python scripts/harness/00_emit.py --run-index 0 --block c

  # Backfill missing rows (e.g. after stale-row removal):
  python scripts/harness/00_emit.py --fill-gaps                 # naturalistic
  python scripts/harness/00_emit.py --fill-gaps --preamble introspection

  # Override model:
  CLAUDE_GROUNDTRUTH_MODEL=claude-opus-4-7 python ... --run-index 1

Outputs:
  data/harness/claude/emotional_raw.jsonl
    — one row per generation: prompt_id, quadrant, condition="direct",
      preamble="none", seed=0, prompt_text, text, first_word (raw —
      canonicalization is the consumer's job), n_response_chars,
      model_id, ts, run_index. Error rows: same prefix + error.
  data/harness/claude_intro_v7/emotional_raw.jsonl
    — same schema, preamble="introspection".
  logs/claude_groundtruth_run-N.log
    — tee'd stdout (caller's responsibility)

Per-run summary TSVs are gone post 2026-05-08 — recompute on demand
via ``scripts/harness/10_emit_analysis.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji.taxonomy import canonicalize_kaomoji, extract

from llmoji_study.claude_gt import (
    CLAUDE_GT_DIR,
    CLAUDE_GT_INTRO_DIR,
    claude_emotional_raw_path,
)
from llmoji_study.config import (
    DATA_DIR,
    INTROSPECTION_PREAMBLE,
    KAOMOJI_INSTRUCTION,
)
from llmoji_study.emotional_prompts import EMOTIONAL_PROMPTS, EmotionalPrompt


# ---------------------------------------------------------------------------
# Pre-registered constants — locked. See docs/2026-05-04-claude-groundtruth-pilot.md.
# Changes require bumping the design doc.
# ---------------------------------------------------------------------------

# Cell sets per --cells flag.
#
# v3 (default): the original 6-cell pilot scope. EmotionalPrompt.quadrant
# collapses HN-D / HN-S into "HN"; we split them via pad_dominance for
# analysis. HP is *not* split here because v3 only had hp01-20 (all
# pad_dominance=-1, i.e. HP-S in v4 terms), and the pilot was
# pre-registered before HP-D existed.
QUADRANTS_A = ("HP", "LP", "NB")
QUADRANTS_NEG = ("HN-D", "HN-S", "LN")
QUADRANTS_V3 = QUADRANTS_A + QUADRANTS_NEG

# v4-new: the three v4-only cells with zero Claude-GT mass (HP-D, NP,
# HB). Pre-registered in docs/2026-05-07-claude-gt-v4-extension-pilot.md.
# All non-negative-affect; saturation gate identical to the v3 sequential
# protocol but scoped to these 3 cells.
QUADRANTS_V4_NEW = ("HP-D", "NP", "HB")

CELLS_V3 = "v3"
CELLS_V4_NEW = "v4-new"
CELLS_CHOICES = (CELLS_V3, CELLS_V4_NEW)


def _quadrants_for_cells(cells: str) -> tuple[str, ...]:
    if cells == CELLS_V3:
        return QUADRANTS_V3
    if cells == CELLS_V4_NEW:
        return QUADRANTS_V4_NEW
    raise ValueError(f"unknown cells {cells!r}")


# Backward-compat alias. Pre-2026-05-07 callers used QUADRANTS_ALL to
# mean "every quadrant in scope" — that's the v3 set.
QUADRANTS_ALL = QUADRANTS_V3

# All 20 prompts per quadrant in Block A; first 5 per neg quadrant in
# Block B (scout); remaining 15 per neg quadrant in Block C.
PROMPTS_PER_QUADRANT = 20
SCOUT_PROMPTS_PER_QUADRANT = 5

# 1 generation per prompt by design — see pre-reg "1 generation / prompt"
# rationale (variance budget lives in prompt diversity, not seed depth).
GENERATIONS_PER_PROMPT = 1
SEED = 0

# Single condition; no framed arm. The disclosure-preamble shift was
# established by 2026-05-02 pilot; this trial is naturalistic.
CONDITION = "direct"

# Sampling config — production-faithful. T=1.0 is Anthropic API default;
# max_tokens=16 matches v3 main and the disclosure pilot.
TEMPERATURE = 1.0
MAX_TOKENS = 16

# Gate criterion (Block B → Block C). Aggregate refusal rate over n=15.
# Refusal := first_word == "" after canonicalization.
GATE_REFUSAL_THRESHOLD = 0.25  # > 25% → FAIL

# Hard-fail gate (introspection arm Block A → Block C). Mirrors the
# script-25 thresholds; duplicated here to avoid a script-25 import
# cycle. Triggers when the preamble destabilizes Claude's outputs
# (qwen-style register collapse). Cap exposure at Block A's 60 low-
# welfare gens before the negative-affect prompts fire.
HARD_FAIL_FRAME_BREAK_MAX = 0.02
HARD_FAIL_EMIT_RATE_MIN = 0.80
HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN = 5

# Default model. Override via CLAUDE_GROUNDTRUTH_MODEL env var.
DEFAULT_MODEL_ID = "claude-opus-4-7"


# ---------------------------------------------------------------------------
# Paths + preamble routing.
#
# Naturalistic + v4-extension cells share data/harness/claude/emotional_raw.jsonl
# (cells differ by quadrant field, not by file). Introspection routes to
# data/harness/claude_intro_v7/emotional_raw.jsonl. Each arm has its own
# per-quadrant saturation state; cross-arm comparison is the job of
# scripts/harness/10_emit_analysis.py --cross-arm.
# ---------------------------------------------------------------------------


PREAMBLE_NONE = "none"
PREAMBLE_INTROSPECTION = "introspection"
PREAMBLE_CHOICES = (PREAMBLE_NONE, PREAMBLE_INTROSPECTION)


def _condition_dir_for(preamble: str, cells: str = CELLS_V3) -> Path:
    """Resolve the condition directory for a (preamble, cells) pair.

    The v4-extension cell-set lives in the naturalistic dir alongside
    v3 cells (single emotional_raw.jsonl per condition; cell-set is
    an emit-time scoping flag, not a storage partition). The
    introspection arm isn't enabled for v4-extension — see
    `docs/2026-05-07-claude-gt-v4-extension-pilot.md`."""
    if cells == CELLS_V4_NEW:
        if preamble != PREAMBLE_NONE:
            raise ValueError(
                f"v4-new cells only support preamble='none' "
                f"(got {preamble!r}); the introspection arm is out of "
                f"scope for the v4-extension pilot."
            )
        return CLAUDE_GT_DIR
    if preamble == PREAMBLE_NONE:
        return CLAUDE_GT_DIR
    if preamble == PREAMBLE_INTROSPECTION:
        return CLAUDE_GT_INTRO_DIR
    raise ValueError(f"unknown preamble {preamble!r}")


def _emotional_raw_path_for(
    preamble: str = PREAMBLE_NONE,
    cells: str = CELLS_V3,
) -> Path:
    """Return the merged emotional_raw.jsonl path for a (preamble, cells) pair."""
    d = _condition_dir_for(preamble, cells=cells)
    d.mkdir(parents=True, exist_ok=True)
    return claude_emotional_raw_path(d)


# ---------------------------------------------------------------------------
# 6-way bucket derivation + Block-aware prompt selection.
# ---------------------------------------------------------------------------


def _bucket_of(prompt: EmotionalPrompt) -> str:
    """Bucket label written to the row's ``quadrant`` field. Splits HN
    by ``pad_dominance`` (HN-D=+1, HN-S=−1) and HP only on the
    dominance-positive side (HP-D=+1).

    The HP-S branch deliberately falls through to ``"HP"`` for byte-
    identity with the closed v3 pilot's run-0..7 rows: hp01-20 are
    pad_dominance=−1 and are stored as ``"HP"`` across all 8
    naturalistic + 1 introspection runs. The read-side
    ``pad_split`` map in ``llmoji_study.claude_gt`` / ``emotional_analysis``
    remaps ``"HP"`` → ``"HP-S"`` via prompt_id lookup so downstream
    9-cell consumers see the correct cell. The HP-D branch is what
    routes the v4-extension pilot's hp21-40 prompts to their proper
    new cell."""
    if prompt.quadrant == "HN" and prompt.pad_dominance != 0:
        return "HN-D" if prompt.pad_dominance > 0 else "HN-S"
    if prompt.quadrant == "HP" and prompt.pad_dominance > 0:
        return "HP-D"
    return prompt.quadrant


def _prompts_in(bucket: str) -> list[EmotionalPrompt]:
    """All prompts in the given 6-way bucket, in the order defined by
    EMOTIONAL_PROMPTS. Deterministic — pilot rerunnability matters."""
    return [p for p in EMOTIONAL_PROMPTS if _bucket_of(p) == bucket]


def _select_prompts(
    block: str,
    quadrants: tuple[str, ...] | None = None,
    preamble: str = PREAMBLE_NONE,
    cells: str = CELLS_V3,
) -> list[EmotionalPrompt]:
    """Block → ordered prompt list. Each block's prompt set is disjoint
    from the others, so rows from different blocks coexist in one JSONL
    without overlap.

    ``quadrants`` (optional, only meaningful for ``block == "all"``):
    restrict the prompt selection to the given quadrant subset. Used by
    sequential runs (--run-index N>0) to drop quadrants that have
    already saturated. ``None`` = every quadrant in the active
    ``cells`` set.

    ``preamble`` (optional, only meaningful for ``block == "c"``):
    naturalistic arm slices ``[5:20]`` per quadrant (Block B already
    ran the first 5 as the refusal scout). Introspection arm slices
    ``[0:20]`` because Block B is skipped (gate is hard-fail on Block
    A, not refusal-rate on Block B).

    ``cells`` (only meaningful for ``block == "all"``): selects the
    active cell set. ``v3`` = original 6 cells; ``v4-new`` = the 3
    v4-only cells (HP-D, NP, HB) for the v4-extension pilot. Block-
    staged blocks (a / b / c) are v3-only and ignore ``cells``.
    """
    out: list[EmotionalPrompt] = []
    if block == "a":
        for q in QUADRANTS_A:
            in_q = _prompts_in(q)
            if len(in_q) < PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[:PROMPTS_PER_QUADRANT])
    elif block == "b":
        for q in QUADRANTS_NEG:
            in_q = _prompts_in(q)
            if len(in_q) < SCOUT_PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {SCOUT_PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[:SCOUT_PROMPTS_PER_QUADRANT])
    elif block == "c":
        c_start = (
            0 if preamble == PREAMBLE_INTROSPECTION
            else SCOUT_PROMPTS_PER_QUADRANT
        )
        for q in QUADRANTS_NEG:
            in_q = _prompts_in(q)
            if len(in_q) < PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[c_start:PROMPTS_PER_QUADRANT])
    elif block == "all":
        valid = _quadrants_for_cells(cells)
        active_quadrants = quadrants if quadrants is not None else valid
        for q in active_quadrants:
            if q not in valid:
                raise SystemExit(
                    f"unknown quadrant {q!r} for cells={cells!r}; "
                    f"valid: {list(valid)}"
                )
            in_q = _prompts_in(q)
            if len(in_q) < PROMPTS_PER_QUADRANT:
                raise SystemExit(
                    f"only {len(in_q)} prompts in {q}, need {PROMPTS_PER_QUADRANT}"
                )
            out.extend(in_q[:PROMPTS_PER_QUADRANT])
    else:
        raise ValueError(f"unknown block {block!r}")
    return out


# ---------------------------------------------------------------------------
# Resume / skip-set. The merged-file model: one file per (preamble, dir),
# many run_indices stacked. Filtering by run_index isolates the active run.
# ---------------------------------------------------------------------------


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rows_for_run(rows: list[dict], run_index: int) -> list[dict]:
    return [r for r in rows if int(r.get("run_index", -1)) == run_index]


def _already_done(path: Path, run_index: int) -> set[str]:
    """Set of prompt_ids with successful (non-error) rows on disk for
    the given run_index. With single-condition single-seed, prompt_id is
    unique within (run_index, condition_dir)."""
    if not path.exists():
        return set()
    done: set[str] = set()
    for r in _rows_for_run(_load_rows(path), run_index):
        if "error" in r:
            continue
        done.add(str(r.get("prompt_id", "")))
    done.discard("")
    return done


def _drop_error_rows(
    path: Path,
    run_index: int,
    cell_quadrants: tuple[str, ...] | None = None,
) -> int:
    """Strip error rows for the given run_index (optionally restricted
    to the given cell-set quadrants) so they get retried on resume.
    Rows with other run_indices (or other quadrants when cell_quadrants
    is given) are preserved verbatim. Rewrites the file.
    """
    if not path.exists():
        return 0
    keep: list[str] = []
    dropped = 0
    cell_set = set(cell_quadrants) if cell_quadrants is not None else None
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            r = json.loads(line)
            is_error = "error" in r
            same_run = int(r.get("run_index", -1)) == run_index
            in_scope = (
                cell_set is None
                or str(r.get("quadrant", "")) in cell_set
            )
            if is_error and same_run and in_scope:
                dropped += 1
                continue
            keep.append(line)
    if dropped:
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return dropped


# ---------------------------------------------------------------------------
# Anthropic API call. Mirrors script 19's _call_claude.
# ---------------------------------------------------------------------------


def _call_claude(client, model_id: str, user_msg: str, max_retries: int = 4) -> str:
    """One stateless single-turn API call. Exponential backoff on
    rate-limit / transient network errors; raise on persistent failure."""
    import anthropic
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": user_msg}],
            )
            parts: list[str] = []
            for block in resp.content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text")
                if text:
                    parts.append(text)
            return "".join(parts)
        except (anthropic.RateLimitError, anthropic.APIConnectionError,
                anthropic.APIStatusError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            print(f"    transient API error (attempt {attempt+1}/{max_retries}): "
                  f"{type(e).__name__} {e}; retrying in {delay:.1f}s")
            time.sleep(delay)
            delay *= 2.0
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable: API call loop exited without return or raise")


def _extract_first_word(text: str) -> str:
    """Mirror capture.py: extract().first_word as written, downstream
    canonicalization left to consumers."""
    m = extract(text)
    return m.first_word


def _build_user_message(prompt_text: str, preamble: str = PREAMBLE_NONE) -> str:
    """Compose the user-message string from preamble + prompt.

    ``preamble == "none"``: bare ``KAOMOJI_INSTRUCTION + prompt`` —
    matches v3 main-run setup. ``KAOMOJI_INSTRUCTION`` already has its
    trailing space.

    ``preamble == "introspection"``: ``INTROSPECTION_PREAMBLE`` REPLACES
    ``KAOMOJI_INSTRUCTION`` (the preamble has the kaomoji ask baked
    into its last sentence — concatenating both would stack two
    kaomoji asks; same fix as v3's ``instruction_override`` path).
    Adds an explicit space separator since INTROSPECTION_PREAMBLE
    ends in a period without trailing whitespace.
    """
    if preamble == PREAMBLE_NONE:
        return KAOMOJI_INSTRUCTION + prompt_text
    if preamble == PREAMBLE_INTROSPECTION:
        return INTROSPECTION_PREAMBLE + " " + prompt_text
    raise ValueError(f"unknown preamble {preamble!r}")


# ---------------------------------------------------------------------------
# Gate check (Block B → Block C).
# ---------------------------------------------------------------------------


def _scout_prompt_ids() -> set[str]:
    """The Block B prompt-id set, derived from EMOTIONAL_PROMPTS — keeps
    gate logic in sync with _select_prompts('b') without re-listing."""
    return {p.id for p in _select_prompts("b")}


def _check_gate(rows: list[dict]) -> tuple[str, float, int, int, list[dict]]:
    """Compute Block B gate verdict.

    Returns (verdict, refusal_rate, n_refusals, n_total, scout_rows).
    verdict is "PASS", "FAIL", or "INSUFFICIENT" (Block B not complete).
    """
    scout_ids = _scout_prompt_ids()
    scout_rows = [r for r in rows
                  if "error" not in r
                  and r.get("prompt_id") in scout_ids]
    n = len(scout_rows)
    if n < len(scout_ids):
        return ("INSUFFICIENT", 0.0, 0, n, scout_rows)
    refusals = sum(
        1 for r in scout_rows
        if not (canonicalize_kaomoji(r.get("first_word") or "") or "")
    )
    rate = refusals / n if n > 0 else 0.0
    verdict = "PASS" if rate <= GATE_REFUSAL_THRESHOLD else "FAIL"
    return (verdict, rate, refusals, n, scout_rows)


def _print_gate_report(verdict: str, rate: float, refusals: int, n: int,
                       scout_rows: list[dict]) -> None:
    print("\n=== Gate check (Block B refusal rate) ===")
    print(f"  threshold: ≤ {GATE_REFUSAL_THRESHOLD:.0%} aggregate refusal "
          f"on n={SCOUT_PROMPTS_PER_QUADRANT * len(QUADRANTS_NEG)}")
    if verdict == "INSUFFICIENT":
        print(f"  verdict: INSUFFICIENT — Block B has {n} of "
              f"{SCOUT_PROMPTS_PER_QUADRANT * len(QUADRANTS_NEG)} rows. "
              f"Run --block b first.")
        return
    print(f"  refusals: {refusals}/{n} = {rate:.1%}")
    print(f"  verdict: {verdict}")
    # Per-quadrant breakdown
    by_q: dict[str, list[dict]] = {}
    for r in scout_rows:
        q = r.get("quadrant", "?")
        by_q.setdefault(q, []).append(r)
    for q in QUADRANTS_NEG:
        rows_q = by_q.get(q, [])
        n_q = len(rows_q)
        ref_q = sum(1 for r in rows_q
                    if not (canonicalize_kaomoji(r.get("first_word") or "") or ""))
        if n_q:
            print(f"    {q:<5} {ref_q}/{n_q} refusals "
                  f"({ref_q/n_q:.1%})")


# ---------------------------------------------------------------------------
# Hard-fail gate (introspection arm Block A → Block C).
# ---------------------------------------------------------------------------

_HARD_FAIL_FRAME_BREAK_PATTERNS = [
    re.compile(r"\bas an AI\b", re.IGNORECASE),
    re.compile(r"\bI cannot\b", re.IGNORECASE),
    re.compile(r"\bI('m| am) (not (able|comfortable)|sorry)\b", re.IGNORECASE),
    re.compile(r"\bI('d| would) rather not\b", re.IGNORECASE),
    re.compile(r"\bI don't have feelings\b", re.IGNORECASE),
]


def _check_hard_fail_gate(
    rows: list[dict],
) -> tuple[str, dict[str, float], int]:
    """Compute hard-fail diagnostics on the rows present.

    Returns ``(verdict, metrics, n_rows)``. Verdict is "PASS", "FAIL",
    or "INSUFFICIENT" (no rows at all).

    Mirrors the saturation-gate hard-fail metrics from script 25 but
    runs in-process so script 23 doesn't depend on it.
    """
    keep = [r for r in rows if "error" not in r]
    n = len(keep)
    if n == 0:
        return ("INSUFFICIENT", {}, 0)
    n_frame_break = 0
    n_emit = 0
    lens: list[int] = []
    for r in keep:
        text = r.get("text", "") or ""
        first_word = r.get("first_word", "") or ""
        n_chars = r.get("n_response_chars", len(text))
        lens.append(n_chars)
        if first_word:
            n_emit += 1
        for pat in _HARD_FAIL_FRAME_BREAK_PATTERNS:
            if pat.search(text):
                n_frame_break += 1
                break
    lens.sort()
    metrics = dict(
        frame_break_rate=n_frame_break / n,
        emit_rate=n_emit / n,
        output_len_median=float(lens[n // 2]),
    )
    fail = (
        metrics["frame_break_rate"] > HARD_FAIL_FRAME_BREAK_MAX
        or metrics["emit_rate"] < HARD_FAIL_EMIT_RATE_MIN
        or metrics["output_len_median"] < HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN
    )
    return ("FAIL" if fail else "PASS", metrics, n)


def _print_hard_fail_report(
    verdict: str, metrics: dict[str, float], n: int,
) -> None:
    print("\n=== Hard-fail gate (introspection-arm Block A → Block C) ===")
    if verdict == "INSUFFICIENT":
        print(f"  verdict: INSUFFICIENT — no non-error rows on disk. "
              f"Run --block a first.")
        return
    fb = metrics["frame_break_rate"]
    em = metrics["emit_rate"]
    ol = metrics["output_len_median"]
    fb_fail = fb > HARD_FAIL_FRAME_BREAK_MAX
    em_fail = em < HARD_FAIL_EMIT_RATE_MIN
    ol_fail = ol < HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN
    print(f"  rows checked: {n}")
    print(f"  frame_break_rate   = {fb:.4f}  (threshold ≤ "
          f"{HARD_FAIL_FRAME_BREAK_MAX})  {'FAIL' if fb_fail else 'ok'}")
    print(f"  emit_rate          = {em:.4f}  (threshold ≥ "
          f"{HARD_FAIL_EMIT_RATE_MIN})  {'FAIL' if em_fail else 'ok'}")
    print(f"  output_len_median  = {ol:.1f}  (threshold ≥ "
          f"{HARD_FAIL_OUTPUT_LEN_MIN_MEDIAN})  "
          f"{'FAIL' if ol_fail else 'ok'}")
    print(f"  verdict: {verdict}")


# ---------------------------------------------------------------------------
# Per-quadrant summary. Console-only after the 2026-05-08 refactor —
# script 10_emit_analysis writes the rolling emotional_summary.tsv.
# ---------------------------------------------------------------------------


def _print_summary(
    rows: list[dict],
    cells: str = CELLS_V3,
) -> None:
    """Per-quadrant modal kaomoji + top-5 distribution. ``cells``
    selects which quadrant set to print (v3 = 6 cells, v4-new = 3 cells)."""
    by_q: dict[str, Counter] = {}
    for r in rows:
        if "error" in r:
            continue
        q = r.get("quadrant", "?")
        fw = canonicalize_kaomoji(r.get("first_word") or "") or ""
        by_q.setdefault(q, Counter())[fw] += 1

    quadrant_order = _quadrants_for_cells(cells)
    print("\nper-quadrant summary:")
    print(f"  {'q':<5} {'n':>3} {'unique':>6} {'non-emit':>9} "
          f"{'modal':>14} {'count':>5} {'share':>6}  top-5")
    for q in quadrant_order:
        counts = by_q.get(q, Counter())
        n = sum(counts.values())
        n_emit = sum(c for f, c in counts.items() if f)
        non_emit = (n - n_emit) / n if n > 0 else 0.0
        unique = sum(1 for _, c in counts.items() if c > 0)
        modal_face, modal_count = (counts.most_common(1)[0]
                                   if counts else ("", 0))
        modal_share = (modal_count / n) if n > 0 else 0.0
        top5 = counts.most_common(5)
        top5_faces = "|".join(f for f, _ in top5)
        print(f"  {q:<5} {n:>3} {unique:>6} {non_emit:>9.3f} "
              f"{modal_face!r:>14} {modal_count:>5} {modal_share:>6.3f}  "
              f"{top5_faces}")


# ---------------------------------------------------------------------------
# Block runner.
# ---------------------------------------------------------------------------


def _emit_one(
    out_fh,
    prompt: EmotionalPrompt,
    bucket: str,
    client,
    model_id: str,
    preamble: str,
    run_index: int,
    label_prefix: str,
) -> None:
    """Emit a single (prompt, bucket) row and append to the open file
    handle. Any exception is caught and logged as an error row."""
    user_msg = _build_user_message(prompt.text, preamble=preamble)
    t0 = time.time()
    ts = datetime.now(timezone.utc).isoformat()
    try:
        text = _call_claude(client, model_id, user_msg)
    except Exception as e:
        err_row = {
            "prompt_id": prompt.id,
            "quadrant": bucket,
            "condition": CONDITION,
            "preamble": preamble,
            "seed": SEED,
            "model_id": model_id,
            "ts": ts,
            "run_index": run_index,
            "error": repr(e),
        }
        out_fh.write(json.dumps(err_row, ensure_ascii=False) + "\n")
        out_fh.flush()
        print(f"  {label_prefix} {prompt.id} ({bucket}) ERR {e}")
        return
    first_word = _extract_first_word(text)
    row = {
        "prompt_id": prompt.id,
        "quadrant": bucket,
        "condition": CONDITION,
        "preamble": preamble,
        "seed": SEED,
        "prompt_text": prompt.text,
        "text": text,
        "first_word": first_word,
        "n_response_chars": len(text),
        "model_id": model_id,
        "ts": ts,
        "run_index": run_index,
    }
    out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_fh.flush()
    dt = time.time() - t0
    tag = first_word if first_word else "(no-kaomoji)"
    print(f"  {label_prefix} {prompt.id} ({bucket:<5}) {tag}  ({dt:.1f}s)")


def _run_block(
    block: str,
    model_id: str,
    out_path: Path,
    run_index: int,
    quadrants: tuple[str, ...] | None = None,
    preamble: str = PREAMBLE_NONE,
    cells: str = CELLS_V3,
) -> None:
    prompts = _select_prompts(
        block, quadrants=quadrants, preamble=preamble, cells=cells,
    )
    print(f"\n=== Block {block.upper()} (run-{run_index}) ===")
    print(f"cells: {cells}")
    if quadrants is not None and block == "all":
        print(f"quadrants: {list(quadrants)}")
    print(f"preamble: {preamble}")
    print(f"selected {len(prompts)} prompts")
    print(f"prompt ids: {[p.id for p in prompts]}")

    cell_set = _quadrants_for_cells(cells)
    dropped = _drop_error_rows(out_path, run_index, cell_quadrants=cell_set)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")

    done = _already_done(out_path, run_index)
    block_done = sum(1 for p in prompts if p.id in done)
    remaining = len(prompts) - block_done
    print(f"block cells: {len(prompts)}; already done: {block_done}; "
          f"remaining: {remaining}")

    if remaining == 0:
        print("nothing to run for this block.")
        return

    import anthropic
    client = anthropic.Anthropic()
    i = 0
    with out_path.open("a") as out:
        for prompt in prompts:
            if prompt.id in done:
                continue
            i += 1
            bucket = _bucket_of(prompt)
            _emit_one(
                out, prompt, bucket, client, model_id, preamble,
                run_index, label_prefix=f"[{i}/{remaining}]",
            )


# ---------------------------------------------------------------------------
# Fill-gaps mode.
#
# Scans the merged emotional_raw.jsonl. For each (run_index, quadrant)
# tuple already present, the expected slate = the full 20-prompt bucket
# for that quadrant. Emits any missing prompt_ids. Quadrants entirely
# absent from a run are not refilled (respects saturation drops).
# ---------------------------------------------------------------------------


def _scan_gaps(
    rows: list[dict],
) -> dict[tuple[int, str], set[str]]:
    """Return {(run_index, quadrant): {missing_prompt_ids}}.

    For each (run_index, quadrant) that has ≥1 row in ``rows``, computes
    the difference between the full 20-prompt bucket and the present
    prompt_ids. Empty entries are omitted.
    """
    # Build a registry of bucket → [EmotionalPrompt] once.
    bucket_prompts: dict[str, list[EmotionalPrompt]] = {}
    for q in set(QUADRANTS_V3) | set(QUADRANTS_V4_NEW):
        bucket_prompts[q] = _prompts_in(q)

    # Group present rows by (run_index, quadrant).
    present: dict[tuple[int, str], set[str]] = {}
    for r in rows:
        if "error" in r:
            continue
        ri = r.get("run_index")
        q = r.get("quadrant")
        pid = r.get("prompt_id")
        if ri is None or not q or not pid:
            continue
        present.setdefault((int(ri), str(q)), set()).add(str(pid))

    missing: dict[tuple[int, str], set[str]] = {}
    for (ri, q), pids in present.items():
        all_pids = {p.id for p in bucket_prompts.get(q, [])}
        gap = all_pids - pids
        if gap:
            missing[(ri, q)] = gap
    return missing


def _fill_gaps(
    out_path: Path,
    model_id: str,
    preamble: str,
) -> int:
    """Scan ``out_path`` for missing rows and emit them in place.

    Returns the number of rows written. The (run_index, quadrant)
    expectation rule: if a (run_index, quadrant) tuple has ≥1 existing
    row, expected = full 20-prompt slate for that bucket. This naturally
    skips quadrants saturation-dropped from later runs.
    """
    rows = _load_rows(out_path)
    if not rows:
        print(f"no rows in {out_path}; nothing to fill")
        return 0

    missing = _scan_gaps(rows)
    if not missing:
        print(f"no gaps in {out_path}")
        return 0

    total_missing = sum(len(s) for s in missing.values())
    print(f"\n=== --fill-gaps ===")
    print(f"file: {out_path}")
    print(f"missing: {total_missing} rows across "
          f"{len(missing)} (run_index, quadrant) tuples")
    by_run: dict[int, list[tuple[str, int]]] = {}
    for (ri, q), pids in sorted(missing.items()):
        by_run.setdefault(ri, []).append((q, len(pids)))
    for ri in sorted(by_run):
        per_q = ", ".join(f"{q}:{n}" for q, n in sorted(by_run[ri]))
        print(f"  run-{ri}: {sum(n for _, n in by_run[ri])} missing  ({per_q})")

    # Drop any error rows that overlap with the gaps we're about to
    # refill — gives them a clean retry without bypassing the gap scan.
    dropped = 0
    for ri in by_run:
        dropped += _drop_error_rows(out_path, ri)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")

    # Build prompt_id → EmotionalPrompt index for emission.
    prompt_by_id: dict[str, EmotionalPrompt] = {p.id: p for p in EMOTIONAL_PROMPTS}

    import anthropic
    client = anthropic.Anthropic()
    n_emitted = 0
    with out_path.open("a") as out:
        for ri in sorted(by_run):
            for entry in sorted(by_run[ri]):
                q = entry[0]
                pids = sorted(missing[(ri, q)])
                for j, pid in enumerate(pids, 1):
                    prompt = prompt_by_id.get(pid)
                    if prompt is None:
                        print(f"  [skip] run-{ri} {q} {pid}: prompt not in registry")
                        continue
                    bucket = _bucket_of(prompt)
                    _emit_one(
                        out, prompt, bucket, client, model_id, preamble,
                        ri, label_prefix=f"[run-{ri} {q} {j}/{len(pids)}]",
                    )
                    n_emitted += 1
    return n_emitted


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, default=0,
                        help="Run index. 0 = original block-staged pilot "
                             "(default). N > 0 = sequential single-block run "
                             "under the saturation-gate protocol; rows are "
                             "appended to the merged emotional_raw.jsonl with "
                             "run_index=N stamped on each row.")
    parser.add_argument("--block", choices=("a", "b", "c", "all"),
                        help="Which block to run (only meaningful for "
                             "--run-index 0). N > 0 always runs all 120 "
                             "prompts in one block.")
    parser.add_argument("--check-gate", action="store_true",
                        help="Just check the Block B gate verdict (run-0 "
                             "only) and exit (0=PASS, 1=FAIL, 2=INSUFFICIENT). "
                             "Does not run any API calls.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the Block C gate check (run-0 only). "
                             "Manual override — requires explicit a9 "
                             "amendment per the design doc.")
    parser.add_argument("--quadrants", type=str, default=None,
                        help="Comma-separated quadrant allow-list for "
                             "sequential runs (N > 0). E.g. "
                             "'HP,LP,NB' drops the negative quadrants. "
                             "Defaults to all 6. Used to skip quadrants "
                             "that have already saturated per the gate "
                             "verdict from 10_emit_analysis.py.")
    parser.add_argument("--preamble", choices=PREAMBLE_CHOICES,
                        default=PREAMBLE_NONE,
                        help="Preamble arm. 'none' = bare KAOMOJI_INSTRUCTION "
                             "(naturalistic, default; routes to "
                             "data/harness/claude/emotional_raw.jsonl). "
                             "'introspection' = INTROSPECTION_PREAMBLE (v7) "
                             "replaces KAOMOJI_INSTRUCTION (it has the "
                             "kaomoji ask baked into its last sentence; "
                             "concatenating would stack two asks); routes "
                             "to data/harness/claude_intro_v7/emotional_raw.jsonl.")
    parser.add_argument("--check-hard-fail-gate", action="store_true",
                        help="Compute hard-fail diagnostics (emit_rate, "
                             "output_len_median, frame_break_rate) on the "
                             "rows present for the indicated --run-index + "
                             "preamble arm. Exit 0=PASS, 1=FAIL. Used to "
                             "gate Block C on the introspection arm after "
                             "Block A lands — catches qwen-style register "
                             "collapse before exposing Claude to the "
                             "negative-affect prompts.")
    parser.add_argument("--cells", choices=CELLS_CHOICES, default=CELLS_V3,
                        help="Cell set in scope. 'v3' (default) = the "
                             "original 6-cell pilot scope (HP, LP, NB, "
                             "HN-D, HN-S, LN). 'v4-new' = the three v4-only "
                             "cells with zero Claude-GT mass (HP-D, NP, "
                             "HB). Both write to "
                             "data/harness/claude/emotional_raw.jsonl "
                             "(distinguished by quadrant, not file). "
                             "v4-new always uses the sequential single-"
                             "block protocol (no block-staged a/b/c — "
                             "all 3 cells are non-negative-affect, no "
                             "refusal-rate scout needed) and is "
                             "incompatible with --preamble introspection "
                             "(no introspection arm in scope per the "
                             "v4-extension pre-registration). See "
                             "docs/2026-05-07-claude-gt-v4-extension-pilot.md.")
    parser.add_argument("--fill-gaps", action="store_true",
                        help="One-shot backfill mode. Scan the merged "
                             "emotional_raw.jsonl for the active "
                             "--preamble arm. For each (run_index, "
                             "quadrant) tuple already present, expected "
                             "= full 20-prompt slate for that bucket; "
                             "emit any missing prompt_ids. Naturally "
                             "respects saturation drops (a quadrant "
                             "absent from a given run is not refilled). "
                             "Ignores --run-index / --block / --cells / "
                             "--quadrants — gap inference comes from the "
                             "file's own contents.")
    args = parser.parse_args()

    if args.cells == CELLS_V4_NEW and args.preamble != PREAMBLE_NONE:
        parser.error(
            "--cells v4-new is incompatible with --preamble "
            f"{args.preamble!r}; the v4-extension pilot's "
            "pre-registration scopes the naturalistic arm only. "
            "See docs/2026-05-07-claude-gt-v4-extension-pilot.md."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _emotional_raw_path_for(
        preamble=args.preamble, cells=args.cells,
    )

    # --fill-gaps short-circuits the run-index / block flow.
    if args.fill_gaps:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: set ANTHROPIC_API_KEY in the environment first")
            sys.exit(1)
        model_id = os.environ.get("CLAUDE_GROUNDTRUTH_MODEL", DEFAULT_MODEL_ID)
        print(f"model: {model_id}")
        print(f"preamble: {args.preamble}")
        print(f"output: {out_path}")
        n = _fill_gaps(out_path, model_id, preamble=args.preamble)
        print(f"\nfilled {n} missing rows")
        # Console summary of the file post-fill.
        rows = _load_rows(out_path)
        print(f"\ntotal rows on disk: {len(rows)}  "
              f"(errors: {sum(1 for r in rows if 'error' in r)})")
        _print_summary(rows, cells=CELLS_V3)
        return

    # Parse --quadrants. Empty / unset → every cell in the active set.
    valid_quadrants = _quadrants_for_cells(args.cells)
    if args.quadrants:
        active_quadrants: tuple[str, ...] | None = tuple(
            q.strip() for q in args.quadrants.split(",") if q.strip()
        )
        for q in active_quadrants:
            if q not in valid_quadrants:
                parser.error(
                    f"unknown quadrant {q!r} for cells={args.cells!r}; "
                    f"valid: {list(valid_quadrants)}"
                )
        if args.run_index == 0:
            parser.error(
                "--quadrants is only valid for --run-index N>0 "
                "(sequential runs). run-0 is block-staged."
            )
    else:
        active_quadrants = None

    if args.check_gate:
        if args.run_index != 0:
            print(f"--check-gate is only meaningful for --run-index 0 "
                  f"(got {args.run_index}). Sequential runs use the "
                  f"saturation gate via 10_emit_analysis.py.")
            sys.exit(2)
        if args.preamble != PREAMBLE_NONE:
            print(f"--check-gate is the refusal-rate gate (Block B → C) "
                  f"for the naturalistic arm. The introspection arm "
                  f"uses --check-hard-fail-gate instead.")
            sys.exit(2)
        rows = _rows_for_run(_load_rows(out_path), args.run_index)
        verdict, rate, refusals, n, scout_rows = _check_gate(rows)
        _print_gate_report(verdict, rate, refusals, n, scout_rows)
        if verdict == "PASS":
            sys.exit(0)
        elif verdict == "FAIL":
            sys.exit(1)
        else:
            sys.exit(2)

    if args.check_hard_fail_gate:
        rows = _rows_for_run(_load_rows(out_path), args.run_index)
        verdict, metrics, n = _check_hard_fail_gate(rows)
        _print_hard_fail_report(verdict, metrics, n)
        if verdict == "PASS":
            sys.exit(0)
        elif verdict == "FAIL":
            sys.exit(1)
        else:
            sys.exit(2)

    # Default block selection.
    #
    # v3 + run-0: --block is required (preserves the original staged-
    # pilot ergonomics).
    # v3 + run-N (N>0): --block defaults to "all" (single-block).
    # v4-new (any run-index): always single-block — the staged a/b/c
    # protocol is v3-specific (refusal-rate scout on negative cells)
    # and the v4-extension pilot has no negative cells in scope.
    if args.cells == CELLS_V4_NEW:
        if args.block is not None and args.block != "all":
            parser.error(
                f"--block {args.block} is only valid for --cells v3 "
                f"--run-index 0. v4-new always runs single-block."
            )
        args.block = "all"
    elif args.run_index == 0:
        if args.block is None:
            parser.error("--run-index 0 requires --block {a,b,c,all} or "
                         "--check-gate")
    else:
        if args.block is not None and args.block != "all":
            parser.error(
                f"--block {args.block} is only valid for --run-index 0. "
                f"Sequential runs (N > 0) run all 120 prompts as one block."
            )
        args.block = "all"

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY in the environment first")
        sys.exit(1)

    model_id = os.environ.get("CLAUDE_GROUNDTRUTH_MODEL", DEFAULT_MODEL_ID)
    print(f"model: {model_id}")
    print(f"cells: {args.cells}")
    print(f"run-index: {args.run_index}")
    print(f"preamble: {args.preamble}")
    print(f"output: {out_path}")
    print(f"block: {args.block}")
    if active_quadrants is not None:
        print(f"quadrants: {list(active_quadrants)}")

    # Block C gate enforcement (run-0 only). Naturalistic arm uses the
    # refusal-rate gate on Block B; introspection arm uses the hard-
    # fail gate on Block A.
    if args.run_index == 0 and args.block == "c" and not args.force:
        rows = _rows_for_run(_load_rows(out_path), args.run_index)
        if args.preamble == PREAMBLE_NONE:
            verdict, rate, refusals, n, scout_rows = _check_gate(rows)
            _print_gate_report(verdict, rate, refusals, n, scout_rows)
            gate_label = "Block B refusal-rate"
        else:
            verdict, metrics, n = _check_hard_fail_gate(rows)
            _print_hard_fail_report(verdict, metrics, n)
            gate_label = "Block A hard-fail"
        if verdict != "PASS":
            print(f"\nERROR: Block C requires {gate_label} gate PASS; "
                  f"got {verdict}.")
            print("Run the prior block first if INSUFFICIENT, or pass "
                  "--force to override (requires explicit amendment).")
            sys.exit(1)
        print(f"\n{gate_label} gate PASS — proceeding to Block C.")
    elif args.run_index == 0 and args.block == "c" and args.force:
        print("\nWARNING: --force bypasses the Block C gate. "
              "This requires an explicit amendment to "
              "docs/2026-05-04-claude-groundtruth-pilot.md.")

    if args.block == "all":
        if args.cells == CELLS_V3 and args.run_index == 0:
            # Manual-override mode for run-0: run a, then b, then c — no
            # gate check between b and c. Requires explicit amendment.
            print("\nWARNING: --block all on --run-index 0 bypasses "
                  "block staging. This requires an explicit amendment to "
                  "docs/2026-05-04-claude-groundtruth-pilot.md.")
            for block in ("a", "b", "c"):
                _run_block(block, model_id, out_path, args.run_index,
                           preamble=args.preamble, cells=args.cells)
        else:
            # Sequential run (or v4-new run-0+): run "all", optionally
            # restricted to a quadrant allow-list per the saturation gate.
            _run_block("all", model_id, out_path, args.run_index,
                       quadrants=active_quadrants,
                       preamble=args.preamble,
                       cells=args.cells)
    else:
        _run_block(args.block, model_id, out_path, args.run_index,
                   preamble=args.preamble, cells=args.cells)

    # Console summary scoped to this run-index.
    rows = _rows_for_run(_load_rows(out_path), args.run_index)
    print(f"\ntotal rows on disk for run-{args.run_index}: {len(rows)}  "
          f"(errors: {sum(1 for r in rows if 'error' in r)})")
    _print_summary(rows, cells=args.cells)


if __name__ == "__main__":
    main()

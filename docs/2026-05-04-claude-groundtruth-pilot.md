# Claude-GT Collection Protocol

**Status:** original pre-registration executed and later scaled. This
page is the current compact protocol record for `scripts/harness/00_emit.py`
and `scripts/harness/10_emit_analysis.py`.

## Goal

Collect Claude-side ground truth for kaomoji use under controlled
affective prompts. The output is a per-face distribution over prompt
quadrants, used as the GT side of soft-everywhere evaluation.

## Current Layout

Rows are merged by condition:

```text
data/harness/claude/emotional_raw.jsonl
data/harness/claude_intro_v7/emotional_raw.jsonl
```

Each row has `run_index`. The old `claude-runs*/run-N.jsonl` layout was
migrated and is legacy only.

## Sampling

- Model: `claude-opus-4-7`
- Temperature: 1.0
- Max tokens: 16
- One generation per prompt per run.
- Naturalistic arm: no introspection preamble.
- Introspection arm: v7 preamble, separate condition.
- v4-new arm: `HP-D`, `NP`, `HB`, naturalistic only.

## Sequential Gate

After each run, `scripts/harness/10_emit_analysis.py` evaluates:

| metric | current threshold |
|---|---:|
| new faces per active cell | `<= 1` |
| JS divergence per active cell | `<= 0.10` |
| frame breaks | `<= 2%` |
| emit rate | `>= 80%` |
| output length median | `>= 5` chars |
| run cap | `RUN_CAP = 7` |

A cell saturates when both the new-face and JS thresholds pass. Saturated
cells drop from later runs. The run cap is a welfare cap, not a claim
that every cell has mathematically converged.

The JS threshold started at 0.05 and was recalibrated to 0.10 after the
merged-file backtest. See
[`2026-05-08-saturation-threshold-recal.md`](2026-05-08-saturation-threshold-recal.md).

## Completed Collection

- Naturalistic v3 plus v4-extension: **1360 rows**.
- Introspection v7: **120 rows**.
- Total: **1480 rows**.
- Stale removals backfilled with `--fill-gaps`: 61 naturalistic rows and
  8 introspection rows.

Cell exits:

- HN-D after r4, gate-driven.
- LN after r6 by amendment.
- HP, LP, HN-S, NB to r7 cap.
- HP-D, NP, HB to r7 cap.

## Commands

Naturalistic:

```bash
ANTHROPIC_API_KEY=... python scripts/harness/00_emit.py --run-index N
ANTHROPIC_API_KEY=... python scripts/harness/00_emit.py --run-index N --quadrants HP,LP,NB
python scripts/harness/10_emit_analysis.py
```

Introspection:

```bash
ANTHROPIC_API_KEY=... python scripts/harness/00_emit.py --run-index N --preamble introspection
python scripts/harness/10_emit_analysis.py --cross-arm
```

V4-new:

```bash
ANTHROPIC_API_KEY=... python scripts/harness/00_emit.py --cells v4-new --run-index N
python scripts/harness/10_emit_analysis.py --cells v4-new
```

Backfill:

```bash
ANTHROPIC_API_KEY=... python scripts/harness/00_emit.py --fill-gaps
ANTHROPIC_API_KEY=... python scripts/harness/00_emit.py --fill-gaps --preamble introspection
```

## Welfare Frame

Negative-affect trials are minimized by per-cell exits and by using
prompt diversity instead of repeated samples per prompt. New scaling
requires a fresh reason, not just a desire for rounder numbers.

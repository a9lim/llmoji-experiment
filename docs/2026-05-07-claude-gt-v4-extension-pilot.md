# Claude-GT V4-Extension Pilot: HP-D / NP / HB

**Status:** complete at RUN_CAP=7. The old pre-registration lived here;
this page now records the outcome.

## Why

The original Claude-GT collection covered the v3 cells:
`HP`, `LP`, `HN-D`, `HN-S`, `LN`, `NB`. The v4 registry split HP into
`HP-D` and `HP-S`, and added `NP` and `HB`. The three cells with zero
Claude-GT mass were:

- `HP-D`: playful or agentic mischief.
- `NP`: relief and gratitude.
- `HB`: uncertainty, skepticism, evaluative arrest.

This pilot filled those cells under the same sequential saturation
protocol as the main Claude-GT run.

## Design

- Model: `claude-opus-4-7`
- Temperature: 1.0
- Max tokens: 16
- Condition: naturalistic, no introspection preamble
- Scope: 3 cells x 20 prompts x 1 generation per run
- Run cap: 7, meaning run indices 0 through 7
- Worst case: 480 generations

Rows now colocate with the naturalistic Claude-GT merged file:

```text
data/harness/claude/emotional_raw.jsonl
```

The cell is carried by the row's `quadrant`, not by a separate
directory.

## Outcome

- 480 non-negative-affect generations.
- 0 errors.
- 0 frame breaks.
- 100% emit rate.
- Modal-quadrant agreement: 100% on faces with at least 3 emits
  (`27/27`).
- Pooled Claude-GT now has nonzero mass in all 9 v4 cells.

Unique-face mass after pooling:

| cell | unique faces |
|---|---:|
| HP-D | 26 |
| HP-S | 38 |
| LP | 38 |
| NP | 40 |
| HN-D | 10 |
| HN-S | 34 |
| LN | 18 |
| NB | 34 |
| HB | 19 |

## Saturation Note

The original `PER_Q_JS_MAX = 0.05` threshold was too strict for these
cells. New-face counts came down, but distribution-shape JS stayed in a
0.10 to 0.20 band. The project recalibrated the live threshold to 0.10
after the merged-file refactor and backtest.

Detail:
[`2026-05-08-saturation-threshold-recal.md`](2026-05-08-saturation-threshold-recal.md).

## Commands

```bash
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --cells v4-new --run-index N
.venv/bin/python scripts/harness/10_emit_analysis.py --cells v4-new
```

`--cells v4-new` is naturalistic only and uses the three-cell set
`HP-D,NP,HB`.

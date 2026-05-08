# Saturation Threshold Recalibration

**Status:** applied 2026-05-08. `PER_Q_JS_MAX` is now 0.10.

## Why

The merged-emotional-raw refactor exposed two issues:

1. Gap detection could precisely backfill stale removals because rows now
   share one file per condition and carry `run_index`.
2. The historical JS threshold of 0.05 was below the observed noise
   floor for some active cells, especially LN and the v4-new cells.

## Backtest Verdict

The gate is:

```text
new_faces <= 1 AND JS <= PER_Q_JS_MAX
```

With `PER_Q_JS_MAX = 0.10`:

- **HN-D** remains a clean gate-driven exit after r4.
- **LN** should be described as an amendment after r6, not as a clean
  gate-driven r5/r6 exit under the old 0.05 threshold.
- **HP, LP, HN-S, NB** still ran to cap.
- **HP-D, NP, HB** have higher entropy and stayed nominally active to
  cap; that is not a signal-quality failure.

## Current Wording

Use this wording in active docs:

> HN-D exited after r4, gate-driven. LN exited after r6 by amendment.
> HP, LP, HN-S, and NB ran to r7. HP-D, NP, and HB also ran to r7 in the
> v4-extension pilot.

Avoid:

- "HN-D exited after r2."
- "LN gate-exited cleanly under 0.05."
- "0.05 is the live JS threshold."

## Rows Backfilled

After stale prompt removals, `--fill-gaps` restored:

- 61 naturalistic rows.
- 8 introspection rows.

Backfill respects the rows present for each `(run_index, quadrant)` pair;
it does not resurrect quadrants that had already dropped from a run.

## Files

- Threshold lives in `scripts/harness/10_emit_analysis.py`.
- Collection protocol: [`2026-05-04-claude-groundtruth-pilot.md`](2026-05-04-claude-groundtruth-pilot.md).
- v4-extension outcome: [`2026-05-07-claude-gt-v4-extension-pilot.md`](2026-05-07-claude-gt-v4-extension-pilot.md).

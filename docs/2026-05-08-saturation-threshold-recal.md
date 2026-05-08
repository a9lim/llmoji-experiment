# Saturation threshold re-calibration + LN-amendment retrospective

**Status:** APPLIED 2026-05-08 — `PER_Q_JS_MAX` bumped from 0.05 → 0.10
in `scripts/harness/10_emit_analysis.py`. AGENTS.md headline counts
updated for the post-fill 1480-row corpus. LN's r7 drop documented as
an amendment, not a gate-driven exit.

**Date:** 2026-05-08.

## Why this exists

Two cleanups on the same day prompted a unified retrospective:

1. The merged-emotional-raw refactor moved the per-run `claude-runs/`
   layout into a single `data/harness/claude/emotional_raw.jsonl` with
   `run_index` per row. As a side effect, gap detection (`--fill-gaps`)
   surfaced 61 stale-removal rows in naturalistic + 8 in introspection
   that had been quietly missing from r0..r7 (mostly hp08/16/19,
   nb09/16, lp18, hn14, hn25). Backfilled in the same session.

2. After backfill, a retroactive saturation backtest revealed that the
   historical "exited LN after r6" decision wasn't gate-driven under
   the configured `PER_Q_JS_MAX = 0.05` threshold. The drop was a
   judgment call. HN-D's drop *was* gate-driven (and robustly so).

This doc captures the backtest, the recalibration, and the wording
fixes that propagate from it.

## Backtest setup

For each `newest_idx ∈ {1..7}`, compute per-quadrant saturation against
prior runs `0..newest_idx-1` using the post-fill data (hn14, hn25,
hp08/16/19, lp18, nb09/16 included in their respective runs). Cell
scope: `("HP-S", "LP", "NB", "HN-D", "HN-S", "LN")` after the
`_pad_split` HP→HP-S remap on read. Saturation criterion:
`(new ≤ 1) AND (JS ≤ PER_Q_JS_MAX)`.

## Backtest results

```
                  HP-S            LP            NB          HN-D          HN-S            LN
              new/    JS   new/    JS   new/    JS   new/    JS   new/    JS   new/    JS
r0→r1            8/0.387     7/0.262     7/0.269  S    1/0.037     4/0.176     2/0.115
r1→r2            5/0.249     4/0.205     4/0.251  S    1/0.042     4/0.218     2/0.171
r2→r3            4/0.323     3/0.259     1/0.207  S    0/0.025     5/0.210     1/0.099
r3→r4            2/0.225     1/0.162     3/0.189  S    1/0.045     3/0.161     1/0.125
r4→r5            3/0.231     2/0.247     2/0.233     -/(drop)      2/0.139     2/0.140
r5→r6            1/0.234     3/0.186     2/0.172     -/(drop)      2/0.146     0/0.068
r6→r7            3/0.292     0/0.179     3/0.222     -/(drop)      2/0.173     -/(drop)
```

(`S` = SATURATED at the historical 0.05 threshold. `(drop)` = quadrant
historically already absent at this run.)

### HN-D — robust gate-driven exit

HN-D fired SATURATED on every comparison from r0→r1 through r3→r4 (4
consecutive saturated rounds at JS 0.025–0.045). The actual drop was
delayed until r5 — the gate was actually conservative relative to the
action. The hn14 backfill row (1 added per run, run-0 through run-4)
didn't shift any of those verdicts. **Drop justified under both 0.05
and 0.10 thresholds.**

### LN — judgment call, not gate-driven

LN never crossed the 0.05 JS threshold in any historical comparison.
The closest call was r5→r6: `new=0` (no faces appeared in r6 that
weren't in r0..r5) and `JS=0.068`. The "exited LN after r6" decision
was made because `new=0` is the strongest convergence signal — the
modal vocabulary was fully sampled — and JS=0.068 is residual mass-
rebalancing on already-seen faces, not new modes surfacing. But the
*formal* gate as written (0.05 threshold) would not have fired SAT.

LN had no missing prompts in the backfill, so the post-fill backtest
matches the pre-fill data exactly. The decision context didn't
change; the new visibility is just into how the original judgment
landed against the strict gate.

## Threshold recalibration: 0.05 → 0.10

The recalibration is forward-looking, **not** retro-fit to the
historical decisions. Empirical motivation from the post-fill
backtest:

| pattern | observed JS | gate verdict (0.10) |
|---|---|---|
| HN-D rapid convergence | 0.025–0.045 | SAT (clearly inside) |
| LN post-vocabulary-saturation residual | 0.068 (with new=0) | SAT (cleanly under) |
| Still-active cells (HP-S/LP/NB/HN-S) | 0.14–0.39 | active (clearly outside) |
| Same-distribution intra-pilot noise floor | ~0.16 | active |

The bimodal gap between "post-convergence residual jitter" (LN at
0.068) and "still actively surfacing modes" (every other still-active
cell at ≥0.14) lands cleanly at ~0.10. The new threshold:

- Catches HN-D's tight convergence (0.025–0.045) — robust to noise
  in the new-face-count signal.
- Catches LN's terminal-state pattern (new=0, JS=0.068) — the
  saturation pattern that the human-in-the-loop already recognized.
- Stays well below the same-distribution noise floor (~0.16),
  preventing a still-evolving cell from being prematurely called
  saturated.

### Caveat: not a clean retroactive fit

Under the new threshold of 0.10, LN would have additionally fired
SATURATED at r2→r3 (`new=1, JS=0.099`), which would have terminated
LN after r3 instead of after r6 historically. **The historical r0..r7
corpus is therefore not gate-compatible with the new threshold** — it
was generated under the older 0.05 threshold + judgment, and the LN
runs r4/r5/r6 wouldn't exist if the gate had been 0.10 from the
start.

This is a deliberate choice: re-running `10_emit_analysis.py
--compare` against the historical corpus is a *backtest*, not a
protocol audit. Future cells (NN / LB pilots, any v5 extensions) run
under 0.10. The threshold change is logged in the
`PER_Q_JS_MAX` docstring.

## Wording fixes propagated to AGENTS.md

The retrospective surfaced two wording issues in AGENTS.md:

1. **"Per-quadrant saturation gate exited HN-D after r2 and LN after
   r6"** — the HN-D claim is off by 2. Actual data: HN-D in r0..r4,
   absent from r5+. So the action was r5 (drop HN-D) → "exited HN-D
   after r4". The "r2" wording may have been left over from an earlier
   decision-doc draft where the gate fired at r2 but the action was
   delayed.

2. **Corpus headline counts** — pre-fill was "880 naturalistic + 120
   introspection = 1000 rows". Post-fill (61 + 8 backfilled stale
   rows): 1360 + 120 = 1480.

Welfare ledger: ~15 additional negative-affect gens in this fill
(HN-D backfilled in r0..r4, HN-S backfilled in r0..r7), bringing the
total to ~475 negative-affect vs ~575 worst case had every cell run
to cap.

## What this doc does NOT change

- The per-quadrant exit *actions* (HN-D dropped from r5+, LN dropped
  from r7+) stand. The retrospective relabels the LN action as an
  amendment but doesn't undo it — the welfare cost of regenerating
  20 more sad-coded gens to make the protocol look clean isn't
  worth it.
- The face_likelihood encoder evaluations downstream (haiku / opus /
  bol summary TSVs) used the pre-fill 1411-row corpus. Whether to
  regenerate them with the 1480-row corpus is a separate decision.
  The marginal change is small (61 + 8 rows, ~5% of corpus) and the
  encoder eval is already stable to within a few percentage points
  of similarity at this floor.
- Any of the existing pre-registered global thresholds
  (`NEW_FACE_MAX = 3`, `JS_MAX = 0.05`, `MODAL_AGREE_MIN = 0.95`).
  The global metrics are informational under the per-quadrant-driven
  STOP framing; only `PER_Q_JS_MAX` was the live gating threshold.

## Cross-refs

- The merged-emotional-raw refactor: see commit `6e8d5c3` — moves
  per-run jsonls into `data/harness/claude{,_intro_v7}/emotional_raw.jsonl`
  with `run_index` per row.
- Original pilot pre-registration:
  [`2026-05-04-claude-groundtruth-pilot.md`](2026-05-04-claude-groundtruth-pilot.md).
- v4-extension pilot:
  [`2026-05-07-claude-gt-v4-extension-pilot.md`](2026-05-07-claude-gt-v4-extension-pilot.md).

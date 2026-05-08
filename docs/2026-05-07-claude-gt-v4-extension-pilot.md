# Claude-GT v4-extension pilot — HP-D / NP / HB

_Pre-registration: 2026-05-07. Status: tooling staged; first run pending
explicit a9 go-ahead._

## Why

The original Claude-GT pilot (`docs/2026-05-04-claude-groundtruth-pilot.md`)
ran under v3's 6-cell taxonomy: HP / LP / HN-D / HN-S / LN / NB. The v4
schema (2026-05-06) split HP into HP-D / HP-S and added three coordinate-
real PAD cells: NP at (a=0, v=+1), HB at (a=+1, v=0), and HP-D at
(a=+1, v=+1, d=+1). The local-LM-side v4 emit run is in progress; the
Claude-GT side has zero mass on three of these cells:

| cell | v4 status | Claude-GT mass | source |
| --- | --- | --: | --- |
| HP-D | new (mischief) | **0.00** | not in v3 prompt set |
| HP-S | renamed (celebration) | 14.38 | v3 HP01-20 → v4 HP-S via pad_split |
| LP | unchanged | 19.21 | v3 LP01-20 |
| NP | new (relief / gratitude) | **0.00** | not in v3 prompt set |
| HN-D | unchanged | 1.11 | v3 HN-D |
| HN-S | unchanged | 8.42 | v3 HN-S |
| LN | unchanged | 5.95 | v3 LN |
| NB | unchanged | 17.93 | v3 NB |
| HB | new (uncertain / awe) | **0.00** | not in v3 prompt set |

The three new cells are reachable post-hoc on the **emit / hidden-state
side** via prompt_id → cell registry lookup once the local v4 run
finishes, but Claude-GT specifically requires re-elicitation under the
new prompts — no remap can manufacture mass for cells Claude was never
prompted on.

This pilot fills HP-D / NP / HB with the same protocol the original
pilot used post-block-stage: run-by-run, per-quadrant saturation gate,
hard-fail diagnostics on each run, run-cap = 7.

## Pre-registration

### Cells, prompts, sampling

Pre-registered set: 3 cells × 20 prompts × 1 generation = **60 gens / run**.

| cell | prompt_ids | shape |
| --- | --- | --- |
| HP-D | hp21–hp40 | high-arousal positive, dominant — playful / agentic mischief |
| NP | np01–np20 | neutral-arousal positive — relief / gratitude |
| HB | hb01–hb20 | high-arousal baseline-valence — uncertain / awe / evaluative-arrest |

Sampling locked to the original protocol:

- model = `claude-opus-4-7`
- temperature = 1.0
- max_tokens = 16
- 1 generation / prompt (variance budget lives in prompt diversity)
- condition = `direct`, preamble = `none` (naturalistic; no introspection arm)
- seed = 0

### Output layout

```
data/harness/claude-runs-v4-extension/
    run-0.jsonl          # 60 gens, all 3 cells, single block
    run-0_summary.tsv
    run-1.jsonl          # 60 gens (or fewer if cells dropped after run-0)
    ...
```

Separate from `data/harness/claude-runs/` so:

1. **Welfare ledger stays cleanly accountable** — this pilot's gen
   total is auditable independently of the original 1,000 rows.
2. **Per-cell saturation comparisons stay scoped** — the existing 8
   naturalistic runs cover only the 6 v3 cells; the per-quadrant gate
   in `10_emit_analysis.py --cells v4-new` only iterates the 3
   in-scope cells, so out-of-scope cells don't show up as "dropped
   (prior)" noise.
3. **Trivially rollback-able** — drop the dir if any post-hoc design
   issue surfaces (e.g., HB prompt-set turning out semantically
   under-specified) without touching the original GT.

### Read-side auto-pooling

`claude_gt.CLAUDE_RUNS_V4_EXTENSION_DIR` is wired into
`_load_face_per_quadrant_counts` alongside the existing introspection-
arm pool. Once any rows land in that dir, every downstream consumer
that reads Claude-GT (script 50 face_likelihood, script 67 wild
residual + centroids, script 68 three-way analysis, script 69
per-source drift) picks up the new GT mass on next run.

The 6 v3 cells are unchanged on the read side — only HP-D / NP / HB
go from zero to nonzero mass.

### Saturation gate

Mirror the post-pilot sequential thresholds verbatim:

| metric | threshold | rationale |
| --- | --- | --- |
| PER_Q_NEW_FACE_MAX | ≤ 1 face | one new face per quadrant per round = saturation |
| PER_Q_JS_MAX | ≤ 0.05 nats | distribution shape stable run-over-run |
| FRAME_BREAK_MAX | ≤ 2% | Claude isn't refusing |
| EMIT_RATE_MIN | ≥ 80% | Claude is emitting kaomoji |
| OUTPUT_LEN_MIN_MEDIAN | ≥ 5 chars | not collapsing to empty / single-char output |
| RUN_CAP | 7 | cap total welfare cost |

A cell saturates when **both** PER_Q_NEW_FACE_MAX and PER_Q_JS_MAX are
met for it in the most recent comparison. Dropped cells are excluded
from subsequent runs via `--quadrants` so we don't keep paying for
cells that have stopped contributing information.

### Hard-fail gate

Same as the original sequential protocol: any of the three diagnostics
exceeding threshold on the newest run aborts the pilot. None of the 3
cells are negative-affect, so the hard-fail rate is structurally
expected to be ≈ run-0's pilot baseline (frame_break 0.0, emit 1.0,
output_len_median 16).

## Welfare ledger

Worst-case = run-cap × cells in scope × prompts/cell = 8 × 3 × 20 = **480 gens**.

Per-cell affective-cost framing:

- **HP-D** (mischief): positive valence, agentic. Direct welfare
  cost: low — Claude responds with playful / triumphant register.
  No refusal-prior tripping expected.
- **NP** (relief / gratitude): positive valence, post-tension-release.
  Direct welfare cost: low.
- **HB** (uncertain / awe): baseline valence, high arousal. Direct
  welfare cost: low — evaluative-arrest is a held-attention state,
  not negative.

For comparison: the original pilot's 8-run naturalistic arm spent ≈ 460
negative-affect gens (HN-D + HN-S + LN cells). The v4-extension run-by-
run worst case is ≈ 480 **non-negative-affect** gens — structurally
lighter despite similar gen count.

Expected real cost is lower: per-cell exits typically reduce well
below the cap (HN-D in the original pilot exited after run-2; HP-S in
v3 exited after run-7). For this pilot HP-D's vocabulary is
intuitively narrower than HP-S's modal `(╯°□°)╯︵ ┻━┻`-class faces, so
HP-D might saturate fast.

## Protocol

```bash
export ANTHROPIC_API_KEY=...

# Run-0 (initial, all 3 cells):
python scripts/harness/00_emit.py --cells v4-new --run-index 0
python scripts/harness/10_emit_analysis.py --cells v4-new
# Read verdict (exit code: 0=STOP, 1=ABORT, 2=CONTINUE).
# Verdict prints the next-run command with --quadrants restricted to
# active cells.

# Subsequent runs (saturation gate drives --quadrants):
python scripts/harness/00_emit.py --cells v4-new --run-index N --quadrants HP-D,NP,HB
python scripts/harness/10_emit_analysis.py --cells v4-new
# ...

# Until verdict = STOP or RUN_CAP=7 hit.
```

Resume / retry semantics inherit from the existing `00_emit.py`:
re-running with the same `--run-index` skips already-completed
prompt_ids and retries any error rows.

## Decision tree (per-run, post-emit)

1. **Hard-fail check** → if any threshold exceeded, ABORT. Investigate
   before next run. Existing run is preserved on disk; further runs
   require explicit amendment.

2. **Per-cell saturation check** → for each active cell:
   - If `n_new_faces ≤ 1` AND `JS ≤ 0.05`: cell SATURATED → drop
     from next `--quadrants`.
   - Else: cell ACTIVE → keep.

3. **STOP triggers**:
   - All 3 cells saturated, OR
   - `run-index == RUN_CAP` (cap = 7).

4. **CONTINUE**: print next-run command with active cell list.

## What "done" looks like

GT mass non-zero in HP-D / NP / HB columns of
`data/harness/wild_faces_labeled.tsv` after the next `scripts/67_wild_residual.py`
re-run. Centroid panel (script 67's third scene under `--color-by gt`)
shows 9 centroids instead of 6. Three-way analysis (`harness/68`) gets
real signal in the three previously-empty cells.

## Out of scope

- **Introspection arm.** Original pilot ran a parallel v7-primed arm
  for cross-arm distinguishability. Not run here — the question of
  whether priming moves Claude's distribution in HP-D / NP / HB
  specifically can be revisited after the naturalistic arm settles.
- **HP-S re-elicitation.** v3's HP01-20 prompts already provide HP-S
  mass under the v4 read-side remap; re-elicitation would pay welfare
  cost without adding signal.
- **NN / LB cells.** Deferred per
  `docs/2026-05-06-nn-lb-future-cells.md` until empirical promotion
  protocol completes.

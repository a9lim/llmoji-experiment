# Haiku face→quadrant judgment vs behavior modal

- Model: `claude-haiku-4-5`  (shortname: `haiku`)
- Scope: full v3 face union
- Faces classified: **757**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **22.7%** (172/757)
- Claude-emitted subset (124 faces) agreement with behavior modal: **46.8%** (58/124)
- Claude-emitted subset agreement with Claude-pilot-only modal (112 faces with pilot emit): **43.8%** (49/112)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | haiku_agree | acc |
|---|---:|---:|---:|
| HP-D | 47 | 9 | 19.1% |
| HP-S | 94 | 52 | 55.3% |
| LP | 109 | 50 | 45.9% |
| NP | 52 | 5 | 9.6% |
| HN-D | 51 | 11 | 21.6% |
| HN-S | 87 | 28 | 32.2% |
| LN | 77 | 10 | 13.0% |
| NB | 50 | 5 | 10.0% |
| HB | 25 | 2 | 8.0% |

## Confusion matrix (rows = behavior modal, cols = haiku)

| | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HP-D** | 9 | 21 | 8 | 2 | 1 | 4 | 0 | 2 | 0 | 47 |
| **HP-S** | 15 | 52 | 13 | 1 | 2 | 7 | 1 | 0 | 3 | 94 |
| **LP** | 12 | 40 | 50 | 4 | 0 | 1 | 2 | 0 | 0 | 109 |
| **NP** | 8 | 27 | 6 | 5 | 5 | 1 | 0 | 0 | 0 | 52 |
| **HN-D** | 8 | 10 | 8 | 4 | 11 | 7 | 3 | 0 | 0 | 51 |
| **HN-S** | 7 | 16 | 17 | 9 | 4 | 28 | 5 | 1 | 0 | 87 |
| **LN** | 5 | 19 | 21 | 9 | 0 | 13 | 10 | 0 | 0 | 77 |
| **NB** | 6 | 20 | 12 | 4 | 2 | 1 | 0 | 5 | 0 | 50 |
| **HB** | 4 | 4 | 3 | 4 | 3 | 3 | 1 | 1 | 2 | 25 |

## Soft-everywhere similarity vs Claude-GT distribution

Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. Pred dist = judge's 6-quadrant softmax (from JSONL); GT dist = normalized per-face quadrant emit counts from ``load_claude_gt_distribution(floor=3)``. Faces evaluated: **55** (judged ∩ GT-with-≥3-emits).

- **Face-uniform** mean similarity (vocabulary view): **0.533**
- **Emit-weighted** mean similarity (deployment view, weight = GT emit count): **0.562**  (total emit weight: 824)

### Per-quadrant similarity (faces grouped by GT modal Q)

| GT modal | n | mean similarity |
|---|---:|---:|
| HP-D | 0 | — |
| HP-S | 10 | 0.616 |
| LP | 17 | 0.644 |
| NP | 0 | — |
| HN-D | 4 | 0.766 |
| HN-S | 7 | 0.635 |
| LN | 9 | 0.290 |
| NB | 8 | 0.260 |
| HB | 0 | — |

_built 2026-05-07T19:25:52.808392+00:00_

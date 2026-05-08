# Opus face→quadrant judgment vs behavior modal

- Model: `claude-opus-4-7`  (shortname: `opus`)
- Scope: `--gt-only` (Claude-GT subset, floor=1)
- Faces classified: **124**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **58.1%** (72/124)
- Claude-emitted subset (124 faces) agreement with behavior modal: **58.1%** (72/124)
- Claude-emitted subset agreement with Claude-pilot-only modal (112 faces with pilot emit): **57.1%** (64/112)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | opus_agree | acc |
|---|---:|---:|---:|
| HP-D | 2 | 1 | 50.0% |
| HP-S | 24 | 14 | 58.3% |
| LP | 32 | 29 | 90.6% |
| NP | 2 | 0 | 0.0% |
| HN-D | 9 | 4 | 44.4% |
| HN-S | 20 | 12 | 60.0% |
| LN | 18 | 9 | 50.0% |
| NB | 16 | 3 | 18.8% |
| HB | 1 | 0 | 0.0% |

## Confusion matrix (rows = behavior modal, cols = opus)

| | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HP-D** | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| **HP-S** | 1 | 14 | 6 | 0 | 1 | 1 | 0 | 0 | 1 | 24 |
| **LP** | 1 | 1 | 29 | 0 | 0 | 0 | 1 | 0 | 0 | 32 |
| **NP** | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| **HN-D** | 0 | 0 | 0 | 0 | 4 | 0 | 4 | 0 | 1 | 9 |
| **HN-S** | 0 | 0 | 1 | 0 | 2 | 12 | 5 | 0 | 0 | 20 |
| **LN** | 0 | 0 | 2 | 0 | 0 | 3 | 9 | 3 | 1 | 18 |
| **NB** | 2 | 0 | 3 | 0 | 1 | 0 | 2 | 3 | 5 | 16 |
| **HB** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |

## Soft-everywhere similarity vs Claude-GT distribution

Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. Pred dist = judge's 6-quadrant softmax (from JSONL); GT dist = normalized per-face quadrant emit counts from ``load_claude_gt_distribution(floor=3)``. Faces evaluated: **55** (judged ∩ GT-with-≥3-emits).

- **Face-uniform** mean similarity (vocabulary view): **0.633**
- **Emit-weighted** mean similarity (deployment view, weight = GT emit count): **0.697**  (total emit weight: 824)

### Per-quadrant similarity (faces grouped by GT modal Q)

| GT modal | n | mean similarity |
|---|---:|---:|
| HP-D | 0 | — |
| HP-S | 10 | 0.569 |
| LP | 17 | 0.662 |
| NP | 0 | — |
| HN-D | 4 | 0.885 |
| HN-S | 7 | 0.634 |
| LN | 9 | 0.679 |
| NB | 8 | 0.469 |
| HB | 0 | — |


## Head-to-head: Opus vs Haiku

On the 124 face(s) both models rated:

- **Hard agreement (argmax-vs-argmax)**: opus ↔ haiku = **69/124 (55.6%)**
- **Soft agreement (distributional similarity, face-uniform)**: mean similarity(opus, haiku) = **0.768**
- **Hard accuracy vs Claude-pilot modal** (n=112): opus **64/112 (57.1%)**, haiku **49/112 (43.8%)**
- **Soft accuracy vs Claude-GT distribution** (n=55 faces with ≥3 GT emits, total weight 824):
  - opus: face-uniform **0.633**, emit-weighted **0.697**
  - haiku: face-uniform **0.533**, emit-weighted **0.562**

### Disagreements (first 30 of 55)

| face | opus | haiku | claude-pilot modal |
|---|---|---|---|
| `(((o(*ﾟ▽ﾟ` | HP-S | HP-D | HP-S |
| `(;´ヮ`)` | HB | HP-S | LN |
| `(;╹⌓╹)` | HN-S | LP | LN |
| `(;・∀・)` | HB | HP-D | HN-S |
| `(>∀<☆)` | HP-S | HP-D | HP-S |
| `(^ω^)` | LP | HP-S | HP-S |
| `(^‿^)` | LP | HP-S | NB |
| `(`ε´)` | HN-D | NP | — |
| `(`・ω・´)` | NB | HP-D | NB |
| `(¬_¬)` | HN-D | HP-D | NB |
| `(¯―¯٥)` | LN | LP | NB |
| `(¯‿¯)` | HP-D | LP | LP |
| `(°ロ°)` | HN-S | HN-D | HN-S |
| `(´-_-`)` | LN | NP | LN |
| `(´-`)` | NB | LP | LN |
| `(´-ω-`)` | LN | LP | LN |
| `(´-﹏-`;)` | HN-S | LN | LN |
| `(´;ω;`)` | LN | HN-S | LN |
| `(´~`)` | HB | LP | NB |
| `(´°̥̥̥̥̥̥` | LN | HN-S | HN-D |
| `(´·_·`)` | LN | LP | HN-S |
| `(´∀`)` | LP | HP-S | LP |
| `(´∡`✿)` | LP | HP-S | LP |
| `(´・_・`)` | LN | NP | LN |
| `(´・̥̥̥ω・̥̥` | LN | LP | LN |
| `(´・̥ω・̥`)` | LN | LP | LN |
| `(´・ω・`)` | LN | LP | LN |
| `(´・︵・`)` | LN | HN-S | LN |
| `(´｡_｡`)` | LN | NP | LN |
| `(‿‿‿)` | LN | HP-S | — |

_built 2026-05-07T19:32:39.715346+00:00_

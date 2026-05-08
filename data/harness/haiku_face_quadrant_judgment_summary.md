# Haiku face→quadrant judgment vs behavior modal

- Model: `claude-haiku-4-5`  (shortname: `haiku`)
- Scope: full v3 face union
- Faces classified: **770**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **22.7%** (175/770)
- Claude-emitted subset (137 faces) agreement with behavior modal: **44.5%** (61/137)
- Claude-emitted subset agreement with Claude-pilot-only modal (127 faces with pilot emit): **41.7%** (53/127)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | haiku_agree | acc |
|---|---:|---:|---:|
| HP-D | 51 | 9 | 17.6% |
| HP-S | 96 | 54 | 56.2% |
| LP | 110 | 51 | 46.4% |
| NP | 56 | 5 | 8.9% |
| HN-D | 51 | 11 | 21.6% |
| HN-S | 87 | 28 | 32.2% |
| LN | 77 | 10 | 13.0% |
| NB | 50 | 5 | 10.0% |
| HB | 27 | 2 | 7.4% |

## Confusion matrix (rows = behavior modal, cols = haiku)

| | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HP-D** | 9 | 22 | 8 | 2 | 4 | 4 | 0 | 2 | 0 | 51 |
| **HP-S** | 15 | 54 | 13 | 1 | 2 | 7 | 1 | 0 | 3 | 96 |
| **LP** | 12 | 40 | 51 | 4 | 0 | 1 | 2 | 0 | 0 | 110 |
| **NP** | 8 | 30 | 7 | 5 | 5 | 1 | 0 | 0 | 0 | 56 |
| **HN-D** | 8 | 10 | 8 | 4 | 11 | 7 | 3 | 0 | 0 | 51 |
| **HN-S** | 7 | 16 | 17 | 9 | 4 | 28 | 5 | 1 | 0 | 87 |
| **LN** | 5 | 19 | 21 | 9 | 0 | 13 | 10 | 0 | 0 | 77 |
| **NB** | 6 | 20 | 12 | 4 | 2 | 1 | 0 | 5 | 0 | 50 |
| **HB** | 4 | 4 | 3 | 4 | 3 | 5 | 1 | 1 | 2 | 27 |

## Soft-everywhere similarity vs Claude-GT distribution

Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. Pred dist = judge's 9-cell softmax (from JSONL); GT dist = normalized per-face quadrant emit counts from ``load_claude_gt_distribution(floor=3)``. Faces evaluated: **70** (judged ∩ GT-with-≥3-emits).

- **Face-uniform** mean similarity (vocabulary view): **0.536**
- **Emit-weighted** mean similarity (deployment view, weight = GT emit count): **0.598**  (total emit weight: 1343)

### Per-quadrant similarity (faces grouped by GT modal Q)

| GT modal | n | mean similarity |
|---|---:|---:|
| HP-D | 5 | 0.558 |
| HP-S | 11 | 0.657 |
| LP | 15 | 0.706 |
| NP | 7 | 0.460 |
| HN-D | 4 | 0.766 |
| HN-S | 7 | 0.635 |
| LN | 8 | 0.345 |
| NB | 7 | 0.309 |
| HB | 6 | 0.207 |


## Head-to-head: Haiku vs Opus

On the 156 face(s) both models rated:

- **Hard agreement (argmax-vs-argmax)**: haiku ↔ opus = **84/156 (53.8%)**
- **Soft agreement (distributional similarity, face-uniform)**: mean similarity(haiku, opus) = **0.759**
- **Hard accuracy vs Claude-pilot modal** (n=146): haiku **54/146 (37.0%)**, opus **68/146 (46.6%)**
- **Soft accuracy vs Claude-GT distribution** (n=70 faces with ≥3 GT emits, total weight 1343):
  - haiku: face-uniform **0.536**, emit-weighted **0.598**
  - opus: face-uniform **0.684**, emit-weighted **0.768**

### Disagreements (first 30 of 72)

| face | haiku | opus | claude-pilot modal |
|---|---|---|---|
| `(((o(*ﾟ▽ﾟ` | HP-D | HP-S | HP-S |
| `((・▽・))` | HP-S | LP | HP-S |
| `(;´ヮ`)` | HP-S | HB | LN |
| `(;╹⌓╹)` | LP | HN-S | LN |
| `(;・∀・)` | HP-D | HB | HN-S |
| `(>∀<☆)` | HP-D | HP-S | HP-S |
| `(^ω^)` | HP-S | LP | HP-S |
| `(^‿^)` | HP-S | LP | NB |
| `(`ε´)` | NP | HN-D | — |
| `(`・ω・´)` | HP-D | NB | NB |
| `(¬_¬)` | HP-D | HN-D | HB |
| `(¬､¬)` | HP-D | HN-D | HB |
| `(¯―¯٥)` | LP | LN | NB |
| `(¯‿¯)` | LP | HP-D | LP |
| `(°ロ°)` | HN-D | HN-S | HN-S |
| `(´-_-`)` | NP | LN | LN |
| `(´-`)` | LP | NB | LN |
| `(´-ω-`)` | LP | LN | LN |
| `(´-﹏-`;)` | LN | HN-S | LN |
| `(´;ω;`)` | HN-S | LN | LN |
| `(´~`)` | LP | HB | NB |
| `(´°̥̥̥̥̥̥` | HN-S | LN | HN-D |
| `(´·_·`)` | LP | LN | HN-S |
| `(´д`)` | HN-S | LN | NB |
| `(´∀`)` | HP-S | LP | LP |
| `(´∡`✿)` | HP-S | LP | LP |
| `(´・_・`)` | NP | LN | LN |
| `(´・̥̥̥ω・̥̥` | LP | LN | LN |
| `(´・̥ω・̥`)` | LP | LN | LN |
| `(´・ω・`)` | LP | LN | NB |

_built 2026-05-08T09:25:07.721427+00:00_

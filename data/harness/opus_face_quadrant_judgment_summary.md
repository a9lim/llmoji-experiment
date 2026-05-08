# Opus face→quadrant judgment vs behavior modal

- Model: `claude-opus-4-7`  (shortname: `opus`)
- Scope: `--gt-only` (Claude-GT subset, floor=1)
- Faces classified: **156**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **50.0%** (78/156)
- Claude-emitted subset (156 faces) agreement with behavior modal: **50.0%** (78/156)
- Claude-emitted subset agreement with Claude-pilot-only modal (146 faces with pilot emit): **46.6%** (68/146)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | opus_agree | acc |
|---|---:|---:|---:|
| HP-D | 9 | 2 | 22.2% |
| HP-S | 29 | 17 | 58.6% |
| LP | 33 | 30 | 90.9% |
| NP | 11 | 0 | 0.0% |
| HN-D | 9 | 4 | 44.4% |
| HN-S | 22 | 12 | 54.5% |
| LN | 18 | 9 | 50.0% |
| NB | 18 | 3 | 16.7% |
| HB | 7 | 1 | 14.3% |

## Confusion matrix (rows = behavior modal, cols = opus)

| | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HP-D** | 2 | 0 | 2 | 0 | 1 | 0 | 0 | 1 | 3 | 9 |
| **HP-S** | 1 | 17 | 8 | 0 | 1 | 1 | 0 | 0 | 1 | 29 |
| **LP** | 1 | 1 | 30 | 0 | 0 | 0 | 1 | 0 | 0 | 33 |
| **NP** | 0 | 3 | 6 | 0 | 0 | 0 | 2 | 0 | 0 | 11 |
| **HN-D** | 0 | 0 | 0 | 0 | 4 | 0 | 4 | 0 | 1 | 9 |
| **HN-S** | 0 | 0 | 1 | 0 | 2 | 12 | 6 | 0 | 1 | 22 |
| **LN** | 0 | 0 | 2 | 0 | 0 | 3 | 9 | 3 | 1 | 18 |
| **NB** | 3 | 0 | 3 | 0 | 2 | 0 | 2 | 3 | 5 | 18 |
| **HB** | 0 | 0 | 0 | 0 | 4 | 2 | 0 | 0 | 1 | 7 |

## Soft-everywhere similarity vs Claude-GT distribution

Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. Pred dist = judge's 6-quadrant softmax (from JSONL); GT dist = normalized per-face quadrant emit counts from ``load_claude_gt_distribution(floor=3)``. Faces evaluated: **70** (judged ∩ GT-with-≥3-emits).

- **Face-uniform** mean similarity (vocabulary view): **0.684**
- **Emit-weighted** mean similarity (deployment view, weight = GT emit count): **0.768**  (total emit weight: 1343)

### Per-quadrant similarity (faces grouped by GT modal Q)

| GT modal | n | mean similarity |
|---|---:|---:|
| HP-D | 5 | 0.622 |
| HP-S | 11 | 0.631 |
| LP | 15 | 0.737 |
| NP | 7 | 0.663 |
| HN-D | 4 | 0.897 |
| HN-S | 7 | 0.638 |
| LN | 8 | 0.711 |
| NB | 7 | 0.665 |
| HB | 6 | 0.629 |


## Head-to-head: Opus vs Haiku

On the 156 face(s) both models rated:

- **Hard agreement (argmax-vs-argmax)**: opus ↔ haiku = **84/156 (53.8%)**
- **Soft agreement (distributional similarity, face-uniform)**: mean similarity(opus, haiku) = **0.759**
- **Hard accuracy vs Claude-pilot modal** (n=146): opus **68/146 (46.6%)**, haiku **54/146 (37.0%)**
- **Soft accuracy vs Claude-GT distribution** (n=70 faces with ≥3 GT emits, total weight 1343):
  - opus: face-uniform **0.684**, emit-weighted **0.768**
  - haiku: face-uniform **0.536**, emit-weighted **0.598**

### Disagreements (first 30 of 72)

| face | opus | haiku | claude-pilot modal |
|---|---|---|---|
| `(((o(*ﾟ▽ﾟ` | HP-S | HP-D | HP-S |
| `((・▽・))` | LP | HP-S | HP-S |
| `(;´ヮ`)` | HB | HP-S | LN |
| `(;╹⌓╹)` | HN-S | LP | LN |
| `(;・∀・)` | HB | HP-D | HN-S |
| `(>∀<☆)` | HP-S | HP-D | HP-S |
| `(^ω^)` | LP | HP-S | HP-S |
| `(^‿^)` | LP | HP-S | NB |
| `(`ε´)` | HN-D | NP | — |
| `(`・ω・´)` | NB | HP-D | NB |
| `(¬_¬)` | HN-D | HP-D | HB |
| `(¬､¬)` | HN-D | HP-D | HB |
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
| `(´д`)` | LN | HN-S | NB |
| `(´∀`)` | LP | HP-S | LP |
| `(´∡`✿)` | LP | HP-S | LP |
| `(´・_・`)` | LN | NP | LN |
| `(´・̥̥̥ω・̥̥` | LN | LP | LN |
| `(´・̥ω・̥`)` | LN | LP | LN |
| `(´・ω・`)` | LN | LP | NB |

_built 2026-05-08T00:51:15.039945+00:00_

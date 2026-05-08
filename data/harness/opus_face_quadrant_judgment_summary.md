# Opus face→quadrant judgment vs behavior modal

- Model: `claude-opus-4-7`  (shortname: `opus`)
- Scope: full v3 face union
- Faces classified: **770**
- Overall agreement with behavior modal (argmax of v3 + Claude pilot + wild emit counts): **27.7%** (213/770)
- Claude-emitted subset (156 faces) agreement with behavior modal: **50.0%** (78/156)
- Claude-emitted subset agreement with Claude-pilot-only modal (146 faces with pilot emit): **46.6%** (68/146)

## Per-quadrant accuracy (behavior-modal as ground truth)

| behavior_modal | n | opus_agree | acc |
|---|---:|---:|---:|
| HP-D | 51 | 9 | 17.6% |
| HP-S | 95 | 35 | 36.8% |
| LP | 110 | 76 | 69.1% |
| NP | 59 | 1 | 1.7% |
| HN-D | 51 | 18 | 35.3% |
| HN-S | 86 | 23 | 26.7% |
| LN | 76 | 31 | 40.8% |
| NB | 51 | 9 | 17.6% |
| HB | 30 | 11 | 36.7% |

## Confusion matrix (rows = behavior modal, cols = opus)

| | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HP-D** | 9 | 3 | 19 | 0 | 4 | 5 | 3 | 4 | 4 | 51 |
| **HP-S** | 8 | 35 | 28 | 0 | 5 | 4 | 7 | 1 | 7 | 95 |
| **LP** | 6 | 7 | 76 | 0 | 3 | 5 | 6 | 6 | 1 | 110 |
| **NP** | 6 | 12 | 22 | 1 | 5 | 2 | 6 | 2 | 3 | 59 |
| **HN-D** | 1 | 1 | 11 | 0 | 18 | 7 | 8 | 0 | 5 | 51 |
| **HN-S** | 3 | 3 | 11 | 0 | 9 | 23 | 19 | 6 | 12 | 86 |
| **LN** | 3 | 7 | 12 | 0 | 2 | 8 | 31 | 7 | 6 | 76 |
| **NB** | 10 | 3 | 16 | 0 | 3 | 3 | 2 | 9 | 5 | 51 |
| **HB** | 2 | 0 | 6 | 0 | 6 | 2 | 1 | 2 | 11 | 30 |

## Soft-everywhere similarity vs Claude-GT distribution

Per-face score: ``similarity = 1 - JSD(pred, gt) / ln 2`` ∈ [0, 1]. Pred dist = judge's 9-cell softmax (from JSONL); GT dist = normalized per-face quadrant emit counts from ``load_claude_gt_distribution(floor=3)``. Faces evaluated: **70** (judged ∩ GT-with-≥3-emits).

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

On the 770 face(s) both models rated:

- **Hard agreement (argmax-vs-argmax)**: opus ↔ haiku = **306/770 (39.7%)**
- **Soft agreement (distributional similarity, face-uniform)**: mean similarity(opus, haiku) = **0.734**
- **Hard accuracy vs Claude-pilot modal** (n=146): opus **68/146 (46.6%)**, haiku **54/146 (37.0%)**
- **Soft accuracy vs Claude-GT distribution** (n=70 faces with ≥3 GT emits, total weight 1343):
  - opus: face-uniform **0.684**, emit-weighted **0.768**
  - haiku: face-uniform **0.536**, emit-weighted **0.598**

### Disagreements (first 30 of 464)

| face | opus | haiku | claude-pilot modal |
|---|---|---|---|
| `(((o(*ﾟ▽ﾟ` | HP-S | HP-D | HP-S |
| `((✿◕‿◕))` | LP | HP-S | — |
| `((・▽・))` | LP | HP-S | HP-S |
| `(-_-;)` | HB | NP | — |
| `(..)` | NB | LP | — |
| `(._.)` | NB | LN | — |
| `(;'ω';)` | HN-S | LP | — |
| `(;;)` | LN | LP | — |
| `(;;_;;)` | LN | HN-S | — |
| `(;_;)` | LN | HN-S | — |
| `(;`ﾉωﾉ´)` | HN-S | HP-S | — |
| `(;¬_¬)` | HN-D | HP-D | — |
| `(;´༎ຶд༎ຶ`)` | LN | HN-S | — |
| `(;´∀`)` | HB | HP-S | — |
| `(;´▽`A)` | HB | HP-S | — |
| `(;´ヮ`)` | HB | HP-S | LN |
| `(;´꒳`)` | HN-S | LP | — |
| `(;ω;)` | LN | HN-S | — |
| `(;д;)` | LN | HN-S | — |
| `(;′⌒`)` | LN | LP | — |
| `(;⁎_⁎)` | HN-S | NP | — |
| `(;⌒_⌒)` | HB | LP | — |
| `(;╹⌓╹)` | HN-S | LP | LN |
| `(;へ:)` | LN | HN-D | — |
| `(;・`O・´)` | HN-D | HP-D | — |
| `(;・∀・)` | HB | HP-D | HN-S |
| `(;一_一)` | HN-D | LN | — |
| `(;￣д￣)` | HN-D | HP-D | — |
| `(;￣□￣)` | HN-S | HN-D | — |
| `(=^-^=)` | LP | HP-S | — |

_built 2026-05-08T09:54:21.181044+00:00_

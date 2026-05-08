# Three-way per-face analysis: GT (use) × Opus/Haiku (read) × BoL (act)

Three structurally different windows on the same per-face quadrant association:

- **GT (use)** — Opus 4.7 emitting the face under known Russell-prompted conditions (`data/harness/claude{,_intro_v7}/emotional_raw.jsonl`). *What the face is actually emitted under.*
- **Opus / Haiku (read)** — model shown the face cold, asked what affective state it represents (`face_likelihood_{opus,haiku}_summary.tsv`). *Symbolic interpretation, no context.*
- **BoL (act)** — Haiku synthesizer pooling adjective-bag picks across many *real in-context emits* of the face → 9-d cell distribution from the 26 circumplex anchors in the locked v2 LEXICON (`claude_faces_lexicon_bag.parquet`). *In-deployment behavior summarized.*

Inner-join shared by all four channels: **50 faces × 1138 GT emissions** (Claude-GT floor ≥ 3).

## Pairwise channel similarity

Mean of `1 − JSD/ln2` over the shared face set. Two flavors: face-uniform (each face counts equally) and emit-weighted (each face counts as Claude actually emits it). Same-cell diagonal is 1.0 by definition.

**Face-uniform**:

| · | GT (use) | Opus (read) | Haiku (read) | BoL (act) |
|---|---|---|---|---|
| GT (use) | 1.000 | 0.684 | 0.525 | 0.464 |
| Opus (read) | 0.684 | 1.000 | 0.797 | 0.550 |
| Haiku (read) | 0.525 | 0.797 | 1.000 | 0.466 |
| BoL (act) | 0.464 | 0.550 | 0.466 | 1.000 |

**Emit-weighted**:

| · | GT (use) | Opus (read) | Haiku (read) | BoL (act) |
|---|---|---|---|---|
| GT (use) | 1.000 | 0.761 | 0.585 | 0.454 |
| Opus (read) | 0.761 | 1.000 | 0.774 | 0.502 |
| Haiku (read) | 0.585 | 0.774 | 1.000 | 0.429 |
| BoL (act) | 0.454 | 0.502 | 0.429 | 1.000 |

**Reading the matrix** — highest pairwise (face-uniform) is `opus↔haiku` (0.797), `gt↔opus` (0.684); lowest is `haiku↔bol` (0.466), `gt↔bol` (0.464).

## Per-GT-quadrant pairwise similarity

Restrict to faces with each modal-GT label, then re-mean the per-pair similarities. Reveals which channels handle which quadrants well — e.g. NB tends to be a BoL win (the lexicon has explicit `neutral`/`detached` anchors), HP often a BoL weakness (deployment use diverges from denoted meaning).

| GT modal | n faces | n emit | gt↔opus | gt↔haiku | gt↔bol | opus↔haiku | opus↔bol | haiku↔bol |
|---|---|---|---|---|---|---|---|---|
| HP-D | 4 | 151 | 0.62 | 0.62 | 0.32 | 0.84 | 0.64 | 0.49 |
| HP-S | 7 | 82 | 0.62 | 0.68 | 0.84 | 0.90 | 0.62 | 0.64 |
| LP | 11 | 216 | 0.80 | 0.73 | 0.48 | 0.92 | 0.61 | 0.56 |
| NP | 7 | 123 | 0.66 | 0.46 | 0.56 | 0.82 | 0.64 | 0.55 |
| HN-D | 1 | 90 | 0.92 | 0.75 | 0.19 | 0.83 | 0.20 | 0.52 |
| HN-S | 4 | 82 | 0.52 | 0.62 | 0.15 | 0.68 | 0.42 | 0.28 |
| LN | 5 | 139 | 0.74 | 0.30 | 0.11 | 0.57 | 0.25 | 0.29 |
| NB | 7 | 101 | 0.67 | 0.31 | 0.34 | 0.77 | 0.54 | 0.43 |
| HB | 4 | 154 | 0.64 | 0.23 | 0.79 | 0.62 | 0.62 | 0.19 |

## Modal-agreement patterns

Each pattern is a 3-bit code `(opus==gt)(haiku==gt)(bol==gt)` — which subset of the three non-GT channels agrees with the GT modal quadrant. 8 patterns total.

| pattern | meaning | n faces | % faces | n emit | % emit |
|---|---|---:|---:|---:|---:|
| `000` | all introspection/synthesis channels disagree with GT | 14 | 28.0% | 177 | 15.6% |
| `110` | opus+haiku read GT; BoL acts differently | 12 | 24.0% | 328 | 28.8% |
| `111` | all channels agree | 7 | 14.0% | 158 | 13.9% |
| `100` | only opus agrees with GT | 6 | 12.0% | 115 | 10.1% |
| `101` | opus reads + BoL acts agree with GT; haiku diverges | 4 | 8.0% | 126 | 11.1% |
| `001` | only BoL agrees with GT | 4 | 8.0% | 144 | 12.7% |
| `010` | only haiku agrees with GT | 3 | 6.0% | 90 | 7.9% |
| `011` | haiku reads + BoL acts agree with GT; opus diverges | 0 | 0.0% | 0 | 0.0% |

## Top-12 most-divergent faces (by max pairwise JSD)

These are the diagnostic faces — where the use / read / act channels pull in different directions most. The agreement pattern column tells you which subset of channels GT-aligns; the per-channel modals tell you the specific disagreement.

| face | emit | pattern | gt | opus | haiku | bol | max-pair JSD | tightest pair |
|---|---:|---|---|---|---|---|---:|---|
| `(´-_-`)` | 4 | `100` | LN | LN | NP | HP-S | 0.693 | opus↔haiku (sim 0.84) |
| `(´-`)` | 52 | `000` | LN | NB | LP | HP-S | 0.693 | opus↔haiku (sim 0.53) |
| `(´-ω-`)` | 9 | `100` | LN | LN | LP | NP | 0.693 | gt↔opus (sim 0.86) |
| `(´;ω;`)` | 42 | `100` | LN | LN | HN-S | HB | 0.693 | gt↔opus (sim 0.93) |
| `(´·_·`)` | 3 | `000` | HN-S | LN | LP | NP | 0.693 | opus↔bol (sim 0.56) |
| `(´・ω・`)` | 24 | `000` | NB | LN | LP | NP | 0.693 | gt↔opus (sim 0.84) |
| `(๑˃‿˂)` | 5 | `000` | NP | HP-S | HP-S | HP-S | 0.693 | opus↔haiku (sim 0.92) |
| `(⌒_⌒;)` | 3 | `000` | HP-D | HB | NP | HN-S | 0.693 | opus↔bol (sim 0.83) |
| `(╥﹏╥)` | 13 | `010` | HN-S | LN | HN-S | LN | 0.693 | gt↔haiku (sim 0.79) |
| `(・∀・)` | 9 | `000` | NB | HP-D | HP-S | NP | 0.693 | opus↔haiku (sim 0.84) |
| `(｡・́︿・̀｡)` | 60 | `010` | HN-S | LN | HN-S | NP | 0.693 | gt↔haiku (sim 0.91) |
| `(｡・ω・｡)` | 32 | `110` | LP | LP | LP | NP | 0.693 | opus↔haiku (sim 0.89) |

## Files

- `data/harness/three_way_per_face.tsv` — per-face data
- `figures/harness/three_way_pairwise_heatmap.png`
- `figures/harness/three_way_top_divergent.png`


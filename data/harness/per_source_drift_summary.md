# Per-source-model BoL drift

Splits the BoL channel from script 68's three-way analysis by source model. The pooled BoL aggregates across every source's per-face synthesis; here each source-model's BoL is kept separate. Headline question: when pooled BoL drifts from Claude-GT (e.g. on `(╯°□°)`), is the drift shared across providers (a kaomoji-vocabulary fact) or concentrated in claude-opus-4-7's deployment (a Claude-specific behavior)?

Coverage: **493 (face, source_model) cells** across 8 source models. 112 faces appear under ≥2 sources. Claude-GT (floor ≥ 3) covers 38 faces in this set.

For reference: pooled-BoL solo similarity vs Claude-GT (face-uniform across the same 38-face set) = **0.255**.

## Per-source-model summary

Each row: how that source's per-face BoL stacks up against Claude-GT (Opus-4.7 elicitation). `modal_agree_rate` is the fraction of source-cells whose argmax quadrant matches GT's argmax.

| source_model | n cells | n emits | n with GT | sim vs GT (face-uniform) | sim vs GT (emit-weighted) | modal agree |
|---|---:|---:|---:|---:|---:|---:|
| `claude-opus-4-7` | 287 | 3227 | 38 | 0.242 | 0.073 | 26% |
| `codex-hook` | 88 | 323 | 24 | 0.201 | 0.190 | 12% |
| `gpt-5.5` | 31 | 272 | 6 | 0.203 | 0.039 | 17% |
| `claude-opus-4-6` | 49 | 95 | 15 | 0.195 | 0.222 | 13% |
| `gpt-5.4` | 15 | 27 | 3 | 0.000 | 0.000 | 0% |
| `gpt-5-5-thinking` | 11 | 19 | 2 | 0.000 | 0.000 | 0% |
| `gpt-5-4-thinking` | 7 | 11 | 2 | 0.436 | 0.436 | 50% |
| `<synthetic>` | 5 | 5 | 2 | 0.218 | 0.218 | 0% |

## Cross-source-model pairwise BoL similarity

On faces synthesized under both sources (≥5 shared faces), mean similarity (`1 − JSD/ln2`) of their per-face BoL distributions. High = sources synthesize the face the same way; low = sources read the face's deployment context differently.

| sm_a | sm_b | n shared | mean sim | modal agree |
|---|---|---:|---:|---:|
| `claude-opus-4-7` | `codex-hook` | 88 | 0.547 | 47% |
| `claude-opus-4-6` | `claude-opus-4-7` | 37 | 0.542 | 41% |
| `claude-opus-4-6` | `codex-hook` | 26 | 0.640 | 58% |
| `claude-opus-4-7` | `gpt-5.5` | 23 | 0.364 | 9% |
| `codex-hook` | `gpt-5.5` | 14 | 0.236 | 14% |
| `claude-opus-4-6` | `gpt-5.5` | 13 | 0.337 | 15% |
| `gpt-5.4` | `gpt-5.5` | 12 | 0.361 | 17% |
| `claude-opus-4-7` | `gpt-5.4` | 11 | 0.487 | 27% |
| `claude-opus-4-7` | `gpt-5-5-thinking` | 8 | 0.345 | 38% |
| `codex-hook` | `gpt-5.4` | 7 | 0.506 | 14% |
| `claude-opus-4-6` | `gpt-5.4` | 7 | 0.551 | 14% |
| `claude-opus-4-7` | `gpt-5-4-thinking` | 6 | 0.250 | 17% |
| `gpt-5-5-thinking` | `gpt-5.5` | 6 | 0.083 | 0% |
| `<synthetic>` | `codex-hook` | 5 | 0.596 | 40% |
| `codex-hook` | `gpt-5-4-thinking` | 5 | 0.219 | 20% |
| `codex-hook` | `gpt-5-5-thinking` | 5 | 0.238 | 20% |
| `<synthetic>` | `claude-opus-4-7` | 5 | 0.600 | 40% |

## Case files

Per-face breakdowns for the divergent faces from script 68's top-divergent table. The pattern to read: does GT's modal match more sources, or fewer? Where does the gap between GT and pooled-BoL come from?

### `(╯°□°)` — case file

| channel | n | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | modal |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| **GT (use)** | 56 | 0.00 | 0.00 | 0.00 | 0.00 | 0.59 | 0.41 | 0.00 | 0.00 | 0.00 | **HN-D** |
| BoL pooled | — | 0.00 | 0.73 | 0.00 | 0.08 | 0.02 | 0.02 | 0.00 | 0.15 | 0.00 | HP-S |
| BoL · claude-opus-4-6 | 2 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | NP |
| BoL · claude-opus-4-7 | 19 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP-S |
| BoL · codex-hook | 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | NB |
| BoL · gpt-5.5 | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 | 0.00 | HN-D |

### `(´;ω;`)` — case file

| channel | n | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | modal |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| **GT (use)** | 38 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.13 | 0.87 | 0.00 | 0.00 | **LN** |
| BoL pooled | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | HB |
| BoL · claude-opus-4-7 | 17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | HB |

### `(╥﹏╥)` — case file

| channel | n | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | modal |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| **GT (use)** | 13 | 0.00 | 0.08 | 0.00 | 0.00 | 0.00 | 0.92 | 0.00 | 0.00 | 0.00 | **HN-S** |
| BoL pooled | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.50 | LN |
| BoL · claude-opus-4-7 | 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.50 | LN |

### `(>∀<☆)` — case file

| channel | n | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | modal |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| **GT (use)** | 10 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **HP-S** |
| BoL pooled | — | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP-S |
| BoL · claude-opus-4-7 | 4 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP-S |

### `(´-`)` — case file

| channel | n | HP-D | HP-S | LP | NP | HN-D | HN-S | LN | NB | HB | modal |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| **GT (use)** | 52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 0.94 | 0.00 | 0.00 | **LN** |
| BoL pooled | — | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP-S |
| BoL · claude-opus-4-7 | 19 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | HP-S |

## Reading the result

Two diagnostic comparisons matter for the use/act gap interpretation:

1. **claude-opus-4-7 vs claude-opus-4-6 BoL similarity** — if these two are very close to each other and both diverge from non-Claude sources, the pattern is Claude-deployment-specific (likely a Claude-deployment register fact, not a model-version fact).
2. **claude-opus-4-7 vs gpt-5.5 / codex-hook BoL similarity** — if these are notably *lower* than Claude-vs-Claude, deployment patterns differ across providers on the same face vocabulary.

Caveat to internalize: every per-source BoL was synthesized by **the same Haiku model** reading provider-conditioned transcript context. So this measures how Haiku reads the context surrounding the kaomoji when the surrounding text is in each provider's style. That's still a real deployment-pattern signal — the surrounding text *is* deployment evidence — but it isn't a clean comparison of what each provider's model 'thinks'.


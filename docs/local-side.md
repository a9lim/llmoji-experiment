# Local Side: Hidden State And Face Likelihood

The local side asks the question with open-weight causal LMs: can the
state that produces a kaomoji be read from hidden activations, and can a
face be mapped back to the likely affective quadrant?

Current lineup:

| short | model | role |
|---|---|---|
| `gemma` | `google/gemma-4-31b-it` | reference |
| `qwen` | `Qwen/Qwen3.6-27B` | hybrid LinearAttention reasoning model |
| `ministral` | `mistralai/Ministral-3-14B-Reasoning-2512` | smaller reasoning model |
| `gpt_oss_20b` | `openai/gpt-oss-20b` | OpenAI-lineage MoE |
| `granite` | `ibm-granite/granite-4.1-30b` | IBM enterprise-tuned model |
| `gemma_intro_v7_primed` | gemma + v7 preamble | historical primed condition |

## Prompt Registry

Current taxonomy is v4: `HP-D`, `HP-S`, `LP`, `NP`, `HN-D`, `HN-S`,
`LN`, `NB`, `HB`. The source of truth is
`llmoji_study/emotional_prompts.py`; ordering and colors live in
`llmoji_study/quadrants.py`.

The v3 six-cell base still matters because most local hidden-state
numbers were collected there: HP, LP, HN-D, HN-S, LN, NB, 20 prompts
each, 8 seeds per model. v4 adds HP-D, NP, and HB. The old
`gemma_intro_v7_primed` condition is useful for the priming result but
does not have a current face-likelihood summary in the active ensemble
search.

## Capture

Generation uses the naturalistic kaomoji ask and `MAX_NEW_TOKENS=16`.
The state of interest is `h_first`: the residual at the kaomoji-emission
token. This stays invariant when later output length changes.

Model-specific capture fixes live in `llmoji_study/capture.py`:

- `gpt_oss_20b`: harmony final-channel override plus Lenny byte
  suppression.
- `ministral`: reasoning-template and byte-decode fixes, plus modern
  emoji suppression with a decoration whitelist.
- `granite`: modern-emoji suppression plus bare-kaomoji extraction for
  forms like `^_^`, `T_T`, and `ಥ﹏ಥ`.
- `qwen`: hybrid LinearAttention cache patch from saklas.

## Hidden-State Representation

Current analyses use the layer-stack representation:

```text
(n_rows, n_probe_layers * hidden_dim)
```

Each row is the concat of every probe layer's `h_first`. This replaced
the old `preferred_layer` heuristic on 2026-05-04. Do not reintroduce
single-layer reads unless the analysis explicitly compares layers.

Primary helpers:

- `load_emotional_features_stack`
- `load_emotional_features_stack_at`
- `apply_pad_split`

## Geometry

The shared result: five model families recover the same affect geometry
up to model-specific PCA axes. Positive cells cluster together,
negative cells cluster together, arousal modulates within valence, and
the HN-D/HN-S dominance split sits on a separable direction.

Layer-stack PCA variance:

| model | PC1 | PC2 | PC3 |
|---|---:|---:|---:|
| gemma | 30.2% | 15.7% | 9.3% |
| qwen | 30.5% | 17.3% | 9.5% |
| ministral | 21.9% | 14.0% | 8.4% |
| granite | 27.6% | 14.1% | 7.5% |
| gpt_oss_20b | 15.8% | 12.5% | 9.5% |

Prompt-grouped hidden -> quadrant accuracy:

| model | accuracy |
|---|---:|
| gemma | 0.992 |
| qwen | 0.985 |
| ministral | 0.984 |
| granite | 0.980 |
| gpt_oss_20b | 0.876 |

## Face As Readout

The kaomoji is not the whole state, but it is not decoration either.

- Hidden -> quadrant is nearly perfect across the lineup.
- Face -> quadrant is strong for tighter vocabularies (`gemma`, `qwen`)
  and weaker for models with broader low-count face inventories.
- Face-centroid R2 is high enough to make face identity a real state
  readout, but not high enough to collapse state to the face string.

Face-stability scripts:

```bash
.venv/bin/python scripts/local/27_v3_face_stability.py
.venv/bin/python scripts/local/28_v3_state_predicts_face.py
.venv/bin/python scripts/local/29_v3_pc_point_clouds_3d.py
```

## Face Likelihood

Face_likelihood inverts the local question. Given a face, it computes
`log P(face | prompt)` under the LM head, aggregates by quadrant, and
softmaxes within face to get a quadrant distribution. It needs forward
passes but no hidden-state access.

Run after `scripts/40_face_union.py`:

```bash
.venv/bin/python scripts/local/50_face_likelihood.py --model gemma
```

Cross-encoder search:

```bash
.venv/bin/python scripts/52_subset_search.py --prefer-full --top-k 25
.venv/bin/python scripts/53_topk_pooling.py --prefer-full
.venv/bin/python scripts/54_ensemble_predict.py --models gemma,ministral,qwen
```

Current overlap-search headline: `{gemma, ministral, opus}` is the best
pooled-GT floor-3 ensemble on the all-encoder overlap (`n=102`), at
0.733 face-uniform and 0.881 emit-weighted similarity. The emitted
770-face lookup table for the same trio scores 0.669 face-uniform and
0.847 emit-weighted over 243 GT-scored faces.

## Introspection V7

`preambles/introspection_v7.txt` is gemma-specific. It replaces the
normal kaomoji ask through `instruction_override`; it is not appended to
the normal instruction. The old double-ask bug is historical.

Use v7 for `gemma_intro_v7_primed`. Do not assume it transfers to qwen;
the observed qwen result was a collapse.

## Reproducing

```bash
.venv/bin/python scripts/local/90_hidden_state_smoke.py
scripts/run_per_model.sh gemma
scripts/run_local_chain.sh
```

`scripts/run_local_chain.sh` is the current orchestration surface after
a v4 emit refresh.

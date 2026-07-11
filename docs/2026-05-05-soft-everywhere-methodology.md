# Soft-Everywhere Methodology

**Status:** canonical since 2026-05-05.

The old face_likelihood evaluation used hard argmax accuracy against
Claude's modal quadrant. That was wrong for this object: both the model
prediction and Claude-GT are distributions. The current metric compares
distribution to distribution.

## Metric

For each face:

```text
similarity(face) = 1 - JSD(pred_dist(face), gt_dist(face)) / ln(2)
```

Range: 0 to 1. Higher is better.

Report two means:

- **Face-uniform**: each GT face counts once. Best for vocabulary
  coverage and long-tail honesty.
- **Emit-weighted**: weighted by Claude's emit count for that face.
  Best for deployment relevance.

Per-face deliverable is always the full distribution over the current
9-cell taxonomy. Modal labels are convenience columns.

## Current Headline

| subset | best ensemble | face-uniform | emit-weighted |
|---|---|---:|---:|
| pooled-GT overlap floor 3 (`n=102`) | `{gemma, ministral, opus}` | 0.733 | 0.881 |
| strict Claude-GT overlap (`n=50`) | `{gemma, opus}` | 0.708 | 0.781 |

These are the corrected 9-cell overlap-search numbers from the current
full summary artifacts. Older notes that cite `gemma_v7primed` are
historical unless a fresh primed face-likelihood summary is regenerated.

## Code Surface

- `llmoji_experiment/jsd.py`: normalization, JSD, and similarity helpers.
- `llmoji_experiment/claude_gt.py`: Claude-GT distribution loader.
- `scripts/52_subset_search.py`: exhaustive subset search.
- `scripts/53_topk_pooling.py`: pooling variants and diagnostics.
- `scripts/54_ensemble_predict.py`: ensemble output.
- `scripts/local/50_face_likelihood.py`: local LM-head encoder.
- `scripts/harness/50_face_likelihood.py`: Anthropic introspection
  encoder.
- `scripts/harness/55_bol_encoder.py`: BoL encoder adapter.

## Rules Of Thumb

- Do not cite argmax accuracy as the headline.
- Do not majority-vote encoders. Average their soft distributions.
- Do not collapse Claude-GT to one-hot labels.
- Use emit-weighted for deployment claims and face-uniform for coverage
  claims.

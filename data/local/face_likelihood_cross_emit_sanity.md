# Cross-emit sanity check

**Ground-truth floor:** total_emit_count ≥ 3
**Encoders compared:** bol, gemma, gpt_oss_20b, granite, haiku, ministral, opus, qwen

## Partition counts (in GT subset)

| origin | n |
|---|---:|
| gemma_only | 32 |
| qwen_only | 50 |
| ministral_only | 25 |
| shared_2 | 22 |
| shared_3 | 12 |

## Accuracy by encoder × origin

Each cell: accuracy (n_correct/n) | κ.  **Bold cells** are cross-prediction (encoder predicting on faces only EMITTED by other v3 models).

| encoder | gemma_only | qwen_only | ministral_only | shared_2 | shared_3 |
|---|---|---|---|---|---|
| bol | 18% (2/11) | κ=0.03 | 42% (10/24) | κ=0.33 | 0% (0/5) | κ=-0.04 | 26% (5/19) | κ=0.18 | 20% (2/10) | κ=0.08 |
| gemma | 62% (20/32) | κ=0.57 | **56% (28/50) | κ=0.50** | **20% (5/25) | κ=0.09** | 36% (8/22) | κ=0.26 | 75% (9/12) | κ=0.71 |
| gpt_oss_20b | 31% (10/32) | κ=0.19 | 34% (17/50) | κ=0.25 | 20% (5/25) | κ=0.06 | 45% (10/22) | κ=0.36 | 50% (6/12) | κ=0.41 |
| granite | 22% (7/32) | κ=0.08 | 32% (16/50) | κ=0.22 | 24% (6/25) | κ=0.15 | 36% (8/22) | κ=0.25 | 25% (3/12) | κ=0.15 |
| haiku | 41% (13/32) | κ=0.30 | 26% (13/50) | κ=0.15 | 12% (3/25) | κ=0.02 | 32% (7/22) | κ=0.20 | 58% (7/12) | κ=0.50 |
| ministral | **22% (7/32) | κ=0.11** | **34% (17/50) | κ=0.24** | 28% (7/25) | κ=0.18 | 9% (2/22) | κ=-0.05 | 33% (4/12) | κ=0.24 |
| opus | 44% (4/9) | κ=0.30 | 59% (13/22) | κ=0.53 | 0% (0/2) | κ=0.00 | 57% (8/14) | κ=0.42 | 67% (6/9) | κ=0.60 |
| qwen | **16% (5/32) | κ=0.06** | 24% (12/50) | κ=0.16 | **20% (5/25) | κ=0.11** | 41% (9/22) | κ=0.30 | 8% (1/12) | κ=0.00 |

## Headline cross-predictions

| encoder | origin | accuracy | κ | reading |
|---|---|---:|---:|---|
| gemma | qwen_only | 56% (28/50) | 0.50 | ✓ converging |
| qwen | gemma_only | 16% (5/32) | 0.06 | ✗ encoder-specific |
| gemma | ministral_only | 20% (5/25) | 0.09 | ✗ encoder-specific |
| ministral | gemma_only | 22% (7/32) | 0.11 | ✗ encoder-specific |
| qwen | ministral_only | 20% (5/25) | 0.11 | ✗ encoder-specific |
| ministral | qwen_only | 34% (17/50) | 0.24 | ~ ambiguous |

**Threshold heuristic** (per user's request): >50% =  encoders converge on shared intrinsic affect; <30% = the empirical-majority signal is too tied to the emitting model's sampling preference (would mean we need broader v3 coverage and/or a Claude-direct baseline).

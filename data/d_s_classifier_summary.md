# D/S classifier — layer-stack hidden state

Pipeline: PCA(cap=20) → StandardScaler → L2-logistic (C=0.1). CV: StratifiedGroupKFold(k=5) with prompt_id as group. Null: 30 within-group label shuffles → 95th-percentile CV bal_acc / AUC.

**Verdict**: bal_acc > null_q95 → the dominance axis is linearly recoverable from layer-stack hidden state in this model. HN is the positive control (rule-3-redesign confirmed); HP is the null reference (post-hoc analysis showed no D/S split universally).

| model | axis | D/S prompts | bal_acc | null q95 | AUC | null q95 | verdict |
|---|---|---|---:|---:|---:|---:|---|
| gemma | HN | 20/20 | 1.000 | 0.652 | 1.000 | 0.687 | **SEPARABLE** |
| gemma | HP | 20/20 | 1.000 | 0.625 | 1.000 | 0.662 | **SEPARABLE** |
| qwen | HN | 20/20 | 1.000 | 0.625 | 1.000 | 0.649 | **SEPARABLE** |
| qwen | HP | 20/20 | 1.000 | 0.598 | 1.000 | 0.625 | **SEPARABLE** |
| ministral | HN | 20/20 | 0.973 | 0.646 | 1.000 | 0.697 | **SEPARABLE** |
| ministral | HP | 20/20 | 1.000 | 0.649 | 1.000 | 0.657 | **SEPARABLE** |
| gpt_oss_20b | HN | 20/20 | 1.000 | 0.650 | 1.000 | 0.655 | **SEPARABLE** |
| gpt_oss_20b | HP | 20/20 | 0.925 | 0.589 | 0.990 | 0.584 | **SEPARABLE** |
| granite | HN | 20/20 | 1.000 | 0.618 | 1.000 | 0.655 | **SEPARABLE** |
| granite | HP | 20/19 | 1.000 | 0.654 | 1.000 | 0.700 | **SEPARABLE** |

## Methodology notes

- Same-prompt seeds are forced into the same fold via `StratifiedGroupKFold(groups=prompt_id)`; without this, row-level CV leaks prompt-text features and inflates accuracy.

- Permutation null shuffles labels *within groups* — preserves the per-prompt seed structure, only randomizes the D-vs-S assignment. This is the right null for *latent geometric* encoding (vs prompt-text-level memorization).

- PCA cap = min(20, n_rows // 3). With ~120-160 rows per axis this lands at 20 for HN/LN, ~20 for HP. The classifier still has more degrees of freedom than the data; regularization (C=0.1) carries it.


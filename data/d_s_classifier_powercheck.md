# D/S classifier — powercheck (matched-n subsampling)

Reference axis: **LN** at 5D / 15S prompts. Non-ref axes subsampled to the same prompt counts; **k=20** random draws per (model, axis). Each draw: full StratifiedGroupKFold(k=5) CV + 30-permutation null on the *subsampled* prompt set.

**Question**: at the LN sample size, does HN still separate? If yes → LN's null is genuine (the design CAN detect a real signal at this n). If HN drops to chance → LN null is power-confounded.

| model | axis | mode | n D/S | bal_acc median (IQR) | null q95 median | separable %ile |
|---|---|---|---|---|---:|---:|
| gemma | HN | matched (k=20) | 5/15 | 0.817 ([0.725,0.867]) | 0.652 | 90% |
| gemma | LN | full (k=1) | 5/15 | 0.667 ([0.667,0.667]) | 0.685 | 0% |
| qwen | HN | matched (k=20) | 5/15 | 0.900 ([0.599,0.925]) | 0.637 | 70% |
| qwen | LN | full (k=1) | 5/15 | 0.560 ([0.560,0.560]) | 0.649 | 0% |
| ministral | HN | matched (k=20) | 5/15 | 0.458 ([0.331,0.589]) | 0.665 | 5% |
| ministral | LN | full (k=1) | 5/15 | 0.638 ([0.638,0.638]) | 0.657 | 0% |
| gpt_oss_20b | HN | matched (k=20) | 5/15 | 0.667 ([0.600,0.767]) | 0.650 | 50% |
| gpt_oss_20b | LN | full (k=1) | 5/15 | 0.500 ([0.500,0.500]) | 0.733 | 0% |
| granite | HN | matched (k=20) | 5/15 | 0.513 ([0.436,0.557]) | 0.668 | 5% |
| granite | LN | full (k=1) | 5/15 | 0.405 ([0.405,0.405]) | 0.688 | 0% |

## Reading

- **LN (mode=full)** is the question: a single point estimate at this prompt-count budget.
- **HN (mode=matched)** is the positive-control answer at matched n. If `separable %` ≈ 100%, the design has power; LN's null is real. If `separable %` drops sharply (toward 50% or below), LN's null is power-confounded and we need more LN-D prompts before concluding.
- **HP (mode=matched)** is the negative-control reference: HP was already null at full n, so it should remain null at matched n.


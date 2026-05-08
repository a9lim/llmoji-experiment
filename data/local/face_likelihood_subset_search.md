# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 8  (bol, gemma, gpt_oss_20b, granite, haiku, ministral, opus, qwen)
**Faces (overlap):** 80
**GT subset (≥3 pooled emits):** 67
**Subsets evaluated:** 255

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **opus** at **face-uniform similarity = 0.738** (emit-weighted 0.848)
- Best ensemble subset: **{gemma,granite,ministral,opus}** at **face-uniform similarity = 0.855** (emit-weighted 0.926); size 4; Δ vs best solo (face-uniform) = +0.117

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| opus | 0.738 | 0.848 | 0.1819 |
| gpt_oss_20b | 0.686 | 0.799 | 0.2174 |
| gemma | 0.654 | 0.717 | 0.2397 |
| ministral | 0.638 | 0.762 | 0.2508 |
| qwen | 0.612 | 0.728 | 0.2693 |
| haiku | 0.587 | 0.722 | 0.2864 |
| granite | 0.564 | 0.641 | 0.3023 |
| bol | 0.449 | 0.433 | 0.3818 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| gemma ↔ opus | 0.466 |
| haiku ↔ opus | 0.402 |
| gemma ↔ gpt_oss_20b | 0.348 |
| gpt_oss_20b ↔ opus | 0.311 |
| gemma ↔ haiku | 0.249 |
| bol ↔ gemma | 0.210 |
| granite ↔ opus | 0.182 |
| bol ↔ opus | 0.170 |
| opus ↔ qwen | 0.168 |
| granite ↔ ministral | 0.162 |
| gemma ↔ granite | 0.156 |
| gpt_oss_20b ↔ granite | 0.126 |
| gemma ↔ qwen | 0.119 |
| gpt_oss_20b ↔ haiku | 0.118 |
| gpt_oss_20b ↔ ministral | 0.110 |
| bol ↔ haiku | 0.097 |
| haiku ↔ qwen | 0.075 |
| gpt_oss_20b ↔ qwen | 0.068 |
| bol ↔ gpt_oss_20b | 0.068 |
| gemma ↔ ministral | 0.061 |
| bol ↔ qwen | 0.057 |
| ministral ↔ qwen | 0.054 |
| ministral ↔ opus | 0.053 |
| bol ↔ granite | 0.049 |
| granite ↔ haiku | 0.036 |
| bol ↔ ministral | 0.015 |
| haiku ↔ ministral | 0.007 |
| granite ↔ qwen | -0.016 |

## Top 25 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 4 | {gemma,granite,ministral,opus} | 0.855 | 0.926 |
| 2 | 4 | {gemma,gpt_oss_20b,granite,opus} | 0.855 | 0.920 |
| 3 | 3 | {gemma,granite,opus} | 0.853 | 0.910 |
| 4 | 3 | {gemma,ministral,opus} | 0.852 | 0.933 |
| 5 | 5 | {gemma,gpt_oss_20b,granite,ministral,opus} | 0.851 | 0.925 |
| 6 | 3 | {gemma,gpt_oss_20b,opus} | 0.850 | 0.924 |
| 7 | 5 | {bol,gemma,granite,ministral,opus} | 0.849 | 0.905 |
| 8 | 2 | {gemma,opus} | 0.849 | 0.911 |
| 9 | 5 | {bol,gemma,gpt_oss_20b,granite,opus} | 0.848 | 0.900 |
| 10 | 6 | {bol,gemma,gpt_oss_20b,granite,ministral,opus} | 0.848 | 0.909 |
| 11 | 3 | {gemma,gpt_oss_20b,granite} | 0.847 | 0.907 |
| 12 | 4 | {gemma,granite,opus,qwen} | 0.847 | 0.921 |
| 13 | 4 | {gemma,gpt_oss_20b,ministral,opus} | 0.847 | 0.928 |
| 14 | 5 | {gemma,gpt_oss_20b,granite,opus,qwen} | 0.844 | 0.919 |
| 15 | 4 | {gemma,gpt_oss_20b,granite,ministral} | 0.844 | 0.915 |
| 16 | 5 | {gemma,granite,ministral,opus,qwen} | 0.844 | 0.922 |
| 17 | 4 | {bol,gemma,granite,opus} | 0.844 | 0.884 |
| 18 | 5 | {bol,gemma,granite,opus,qwen} | 0.844 | 0.901 |
| 19 | 5 | {gemma,granite,haiku,ministral,opus} | 0.843 | 0.923 |
| 20 | 6 | {bol,gemma,granite,ministral,opus,qwen} | 0.843 | 0.907 |
| 21 | 6 | {bol,gemma,gpt_oss_20b,granite,opus,qwen} | 0.842 | 0.904 |
| 22 | 3 | {gemma,granite,ministral} | 0.842 | 0.910 |
| 23 | 5 | {bol,gemma,gpt_oss_20b,granite,ministral} | 0.842 | 0.897 |
| 24 | 5 | {gemma,gpt_oss_20b,granite,haiku,opus} | 0.842 | 0.918 |
| 25 | 6 | {gemma,gpt_oss_20b,granite,haiku,ministral,opus} | 0.842 | 0.922 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {opus} | 0.834 | 0.906 |
| 2 | {gemma,opus} | 0.849 | 0.911 |
| 3 | {gemma,granite,opus} | 0.853 | 0.910 |
| 4 | {gemma,granite,ministral,opus} | 0.855 | 0.926 |
| 5 | {gemma,gpt_oss_20b,granite,ministral,opus} | 0.851 | 0.925 |
| 6 | {bol,gemma,gpt_oss_20b,granite,ministral,opus} | 0.848 | 0.909 |
| 7 | {bol,gemma,gpt_oss_20b,granite,haiku,ministral,opus} | 0.841 | 0.910 |
| 8 | {bol,gemma,gpt_oss_20b,granite,haiku,ministral,opus,qwen} | 0.836 | 0.908 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| opus | 43.3% (28/67) | 0.332 |
| gemma | 35.8% (24/67) | 0.277 |
| haiku | 31.3% (21/67) | 0.196 |
| gpt_oss_20b | 29.9% (20/67) | 0.207 |
| granite | 28.4% (19/67) | 0.175 |
| bol | 25.4% (17/67) | 0.151 |
| qwen | 13.4% (9/67) | 0.050 |
| ministral | 9.0% (6/67) | -0.043 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 1 | {opus} | 43.3% (29/67) | 0.332 | 0.834 |
| 2 | {opus,qwen} | 43.3% (29/67) | 0.337 | 0.815 |
| 4 | {bol,granite,haiku,opus} | 41.8% (28/67) | 0.325 | 0.822 |
| 4 | {gemma,granite,haiku,opus} | 41.8% (28/67) | 0.315 | 0.839 |
| 4 | {bol,gemma,granite,opus} | 40.3% (27/67) | 0.311 | 0.844 |
| 5 | {gpt_oss_20b,granite,haiku,opus,qwen} | 40.3% (27/67) | 0.297 | 0.819 |
| 7 | {bol,gemma,gpt_oss_20b,granite,haiku,ministral,opus} | 40.3% (27/67) | 0.305 | 0.841 |
| 6 | {gemma,gpt_oss_20b,granite,haiku,opus,qwen} | 40.3% (27/67) | 0.300 | 0.835 |
| 6 | {bol,gemma,granite,haiku,ministral,opus} | 40.3% (27/67) | 0.305 | 0.842 |
| 7 | {bol,gemma,granite,haiku,ministral,opus,qwen} | 40.3% (27/67) | 0.305 | 0.837 |


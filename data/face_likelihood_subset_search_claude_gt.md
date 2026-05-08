# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 10  (bol, gemma, gpt_oss_20b, granite, haiku, ministral, opus, qwen, rinna_bilingual_4b_jpfull, rinna_jp_3_6b_jpfull)
**Faces (overlap):** 293
**GT subset (Claude empirical, total ≥ 3):** 50
**Subsets evaluated:** 1023

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **opus** at **face-uniform similarity = 0.684** (emit-weighted 0.761)
- Best ensemble subset: **{gemma,opus}** at **face-uniform similarity = 0.708** (emit-weighted 0.781); size 2; Δ vs best solo (face-uniform) = +0.024

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| opus | 0.684 | 0.761 | 0.2191 |
| gemma | 0.639 | 0.687 | 0.2501 |
| gpt_oss_20b | 0.550 | 0.609 | 0.3121 |
| haiku | 0.525 | 0.585 | 0.3291 |
| granite | 0.500 | 0.553 | 0.3464 |
| ministral | 0.494 | 0.581 | 0.3508 |
| bol | 0.464 | 0.454 | 0.3712 |
| rinna_bilingual_4b_jpfull | 0.461 | 0.512 | 0.3737 |
| qwen | 0.457 | 0.534 | 0.3761 |
| rinna_jp_3_6b_jpfull | 0.452 | 0.533 | 0.3801 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| haiku ↔ opus | 0.326 |
| gemma ↔ opus | 0.320 |
| gemma ↔ gpt_oss_20b | 0.189 |
| gpt_oss_20b ↔ opus | 0.181 |
| gemma ↔ haiku | 0.177 |
| granite ↔ opus | 0.150 |
| bol ↔ gemma | 0.125 |
| granite ↔ ministral | 0.122 |
| opus ↔ rinna_jp_3_6b_jpfull | 0.116 |
| bol ↔ opus | 0.102 |
| gemma ↔ granite | 0.099 |
| gpt_oss_20b ↔ haiku | 0.097 |
| ministral ↔ opus | 0.095 |
| gpt_oss_20b ↔ ministral | 0.095 |
| gemma ↔ ministral | 0.086 |
| gemma ↔ rinna_jp_3_6b_jpfull | 0.080 |
| gpt_oss_20b ↔ qwen | 0.078 |
| opus ↔ qwen | 0.076 |
| gpt_oss_20b ↔ granite | 0.075 |
| opus ↔ rinna_bilingual_4b_jpfull | 0.075 |
| haiku ↔ rinna_bilingual_4b_jpfull | 0.071 |
| bol ↔ haiku | 0.063 |
| bol ↔ rinna_bilingual_4b_jpfull | 0.060 |
| gemma ↔ qwen | 0.058 |
| bol ↔ gpt_oss_20b | 0.057 |
| granite ↔ haiku | 0.050 |
| rinna_bilingual_4b_jpfull ↔ rinna_jp_3_6b_jpfull | 0.045 |
| qwen ↔ rinna_bilingual_4b_jpfull | 0.043 |
| granite ↔ rinna_jp_3_6b_jpfull | 0.043 |
| gpt_oss_20b ↔ rinna_jp_3_6b_jpfull | 0.041 |
| qwen ↔ rinna_jp_3_6b_jpfull | 0.039 |
| bol ↔ qwen | 0.039 |
| bol ↔ granite | 0.038 |
| haiku ↔ rinna_jp_3_6b_jpfull | 0.037 |
| ministral ↔ qwen | 0.037 |
| granite ↔ qwen | 0.034 |
| gpt_oss_20b ↔ rinna_bilingual_4b_jpfull | 0.027 |
| gemma ↔ rinna_bilingual_4b_jpfull | 0.025 |
| haiku ↔ qwen | 0.021 |
| granite ↔ rinna_bilingual_4b_jpfull | 0.017 |
| bol ↔ ministral | 0.014 |
| haiku ↔ ministral | 0.014 |
| ministral ↔ rinna_bilingual_4b_jpfull | 0.012 |
| ministral ↔ rinna_jp_3_6b_jpfull | 0.007 |
| bol ↔ rinna_jp_3_6b_jpfull | 0.005 |

## Top 25 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 2 | {gemma,opus} | 0.708 | 0.781 |
| 2 | 3 | {bol,gemma,opus} | 0.688 | 0.744 |
| 3 | 3 | {gemma,granite,opus} | 0.687 | 0.760 |
| 4 | 4 | {bol,gemma,granite,opus} | 0.686 | 0.744 |
| 5 | 1 | {opus} | 0.684 | 0.761 |
| 6 | 3 | {gemma,haiku,opus} | 0.683 | 0.763 |
| 7 | 3 | {gemma,gpt_oss_20b,opus} | 0.680 | 0.757 |
| 8 | 5 | {bol,gemma,granite,haiku,opus} | 0.679 | 0.747 |
| 9 | 4 | {bol,gemma,haiku,opus} | 0.678 | 0.744 |
| 10 | 4 | {gemma,granite,haiku,opus} | 0.677 | 0.757 |
| 11 | 4 | {bol,gemma,gpt_oss_20b,opus} | 0.674 | 0.738 |
| 12 | 5 | {bol,gemma,gpt_oss_20b,granite,opus} | 0.673 | 0.736 |
| 13 | 4 | {gemma,gpt_oss_20b,granite,opus} | 0.669 | 0.744 |
| 14 | 3 | {gemma,ministral,opus} | 0.668 | 0.752 |
| 15 | 6 | {bol,gemma,gpt_oss_20b,granite,haiku,opus} | 0.667 | 0.737 |
| 16 | 4 | {bol,gemma,ministral,opus} | 0.667 | 0.735 |
| 17 | 5 | {bol,gemma,gpt_oss_20b,haiku,opus} | 0.666 | 0.735 |
| 18 | 4 | {gemma,gpt_oss_20b,haiku,opus} | 0.666 | 0.745 |
| 19 | 5 | {bol,gemma,granite,ministral,opus} | 0.665 | 0.731 |
| 20 | 4 | {bol,gemma,granite,haiku} | 0.664 | 0.724 |
| 21 | 5 | {gemma,gpt_oss_20b,granite,haiku,opus} | 0.663 | 0.741 |
| 22 | 5 | {bol,gemma,granite,opus,rinna_jp_3_6b_jpfull} | 0.663 | 0.729 |
| 23 | 3 | {gemma,opus,rinna_bilingual_4b_jpfull} | 0.662 | 0.735 |
| 24 | 6 | {bol,gemma,granite,haiku,ministral,opus} | 0.662 | 0.734 |
| 25 | 4 | {bol,gemma,opus,rinna_jp_3_6b_jpfull} | 0.662 | 0.727 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {opus} | 0.684 | 0.761 |
| 2 | {gemma,opus} | 0.708 | 0.781 |
| 3 | {bol,gemma,opus} | 0.688 | 0.744 |
| 4 | {bol,gemma,granite,opus} | 0.686 | 0.744 |
| 5 | {bol,gemma,granite,haiku,opus} | 0.679 | 0.747 |
| 6 | {bol,gemma,gpt_oss_20b,granite,haiku,opus} | 0.667 | 0.737 |
| 7 | {bol,gemma,gpt_oss_20b,granite,haiku,ministral,opus} | 0.653 | 0.725 |
| 8 | {bol,gemma,gpt_oss_20b,granite,haiku,ministral,opus,rinna_jp_3_6b_jpfull} | 0.638 | 0.712 |
| 9 | {bol,gemma,gpt_oss_20b,granite,haiku,ministral,opus,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 0.626 | 0.698 |
| 10 | {bol,gemma,gpt_oss_20b,granite,haiku,ministral,opus,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 0.613 | 0.686 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| opus | 58.0% (28/50) | 0.508 |
| gemma | 52.0% (26/50) | 0.452 |
| haiku | 44.0% (22/50) | 0.339 |
| granite | 34.0% (17/50) | 0.235 |
| gpt_oss_20b | 32.0% (16/50) | 0.223 |
| bol | 30.0% (15/50) | 0.193 |
| rinna_jp_3_6b_jpfull | 26.0% (13/50) | 0.181 |
| qwen | 24.0% (12/50) | 0.154 |
| ministral | 14.0% (7/50) | 0.025 |
| rinna_bilingual_4b_jpfull | 14.0% (7/50) | 0.081 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 7 | {bol,gemma,haiku,ministral,opus,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 66.0% (33/50) | 0.605 | 0.626 |
| 8 | {bol,gemma,gpt_oss_20b,granite,haiku,opus,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 66.0% (33/50) | 0.605 | 0.634 |
| 5 | {gemma,gpt_oss_20b,granite,haiku,opus} | 64.0% (32/50) | 0.581 | 0.663 |
| 5 | {gemma,haiku,opus,qwen,rinna_jp_3_6b_jpfull} | 64.0% (32/50) | 0.582 | 0.625 |
| 8 | {bol,gemma,gpt_oss_20b,haiku,ministral,opus,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 64.0% (32/50) | 0.583 | 0.621 |
| 8 | {bol,gemma,gpt_oss_20b,haiku,ministral,opus,qwen,rinna_bilingual_4b_jpfull} | 64.0% (32/50) | 0.583 | 0.618 |
| 9 | {bol,gemma,gpt_oss_20b,haiku,ministral,opus,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 64.0% (32/50) | 0.583 | 0.607 |
| 6 | {gemma,gpt_oss_20b,granite,haiku,opus,qwen} | 64.0% (32/50) | 0.581 | 0.639 |
| 9 | {bol,gemma,gpt_oss_20b,granite,haiku,opus,qwen,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 64.0% (32/50) | 0.582 | 0.620 |
| 8 | {bol,gemma,granite,haiku,ministral,opus,rinna_bilingual_4b_jpfull,rinna_jp_3_6b_jpfull} | 64.0% (32/50) | 0.582 | 0.631 |


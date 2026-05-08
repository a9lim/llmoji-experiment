# Face_likelihood — exhaustive subset search (soft / JSD)

**Encoders:** 3  (gemma, haiku, opus)
**Faces (overlap):** 124
**GT subset (≥3 pooled emits):** 78
**Subsets evaluated:** 7

## Methodology

**Headline metric: distribution similarity.** For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes); GT is Claude's (or pooled) empirical per-quadrant distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report ``similarity = 1 − JSD/ln 2`` ∈ [0, 1] (1.0 = distributions identical, 0.0 = maximally divergent; max JSD ≈ 0.6931). Argmax accuracy + Cohen's κ are available in the supplementary appendix below — they are the production-shaped reading but lose information at GT-tie boundaries, so they don't drive ranking.

**Two flavors of mean similarity, reported side-by-side:**

- **Face-uniform (`similarity`)** — arithmetic mean of per-face JSD across the GT subset. Each face counts equally regardless of how often Claude emits it. Reads as: "how well does the ensemble characterize Claude's *vocabulary*?" — sensitive to long-tail failures.
- **Emit-weighted (`similarity_weighted`)** — weighted by per-face Claude emit count. Faces Claude uses more contribute proportionally more to the score. Reads as: "how well does the ensemble characterize Claude's *emission distribution*?" — closer to deployment-relevant (plugin user encounters faces at frequency, not uniformly). Tends to read higher than face-uniform because modal faces are easier wins.

Subset ranking below is by **face-uniform similarity** (stricter / more honest about coverage). Weighted column shown alongside.

## Headline

- Best single encoder: **opus** at **face-uniform similarity = 0.726** (emit-weighted 0.839)
- Best ensemble subset: **{gemma,opus}** at **face-uniform similarity = 0.856** (emit-weighted 0.906); size 2; Δ vs best solo (face-uniform) = +0.130

## Per-encoder solo distribution-similarity

| encoder | similarity (face-uniform) | similarity (emit-weighted) | mean JSD (face-uniform) |
|---|---:|---:|---:|
| opus | 0.726 | 0.839 | 0.1901 |
| gemma | 0.692 | 0.703 | 0.2133 |
| haiku | 0.608 | 0.718 | 0.2717 |

## Pairwise Cohen's κ across encoders (whole overlap)

Higher κ = more correlated. Encoder pairs with low κ make complementary errors and are more useful to combine.

| pair | κ |
|---|---:|
| gemma ↔ opus | 0.499 |
| haiku ↔ opus | 0.460 |
| gemma ↔ haiku | 0.273 |

## Top 5 subsets by face-uniform similarity

| rank | size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---:|---|---:|---:|
| 1 | 2 | {gemma,opus} | 0.856 | 0.906 |
| 2 | 1 | {opus} | 0.831 | 0.907 |
| 3 | 3 | {gemma,haiku,opus} | 0.830 | 0.900 |
| 4 | 1 | {gemma} | 0.818 | 0.808 |
| 5 | 2 | {gemma,haiku} | 0.815 | 0.873 |

## Per-size best subset (by face-uniform similarity)

| size | encoders | similarity (face-uniform) | similarity (emit-weighted) |
|---:|---|---:|---:|
| 1 | {opus} | 0.831 | 0.907 |
| 2 | {gemma,opus} | 0.856 | 0.906 |
| 3 | {gemma,haiku,opus} | 0.830 | 0.900 |

## Supplementary: argmax accuracy + Cohen's κ (production-shaped reading)

These metrics treat GT as a one-hot modal label. They characterize a deployed plugin that emits a single quadrant call, not the distribution-shipping ensemble this script ranks. Reported here for legibility against older numbers in the project history.

### Per-encoder solo (argmax)

| encoder | accuracy | κ |
|---|---:|---:|
| opus | 59.0% (46/78) | 0.491 |
| gemma | 50.0% (39/78) | 0.420 |
| haiku | 47.4% (37/78) | 0.365 |

### Top-10 subsets by argmax accuracy

| size | encoders | accuracy | κ | similarity |
|---:|---|---:|---:|---:|
| 1 | {opus} | 59.0% (46/78) | 0.491 | 0.831 |
| 3 | {gemma,haiku,opus} | 51.3% (40/78) | 0.406 | 0.830 |
| 2 | {gemma,opus} | 50.0% (39/78) | 0.398 | 0.856 |
| 1 | {gemma} | 50.0% (39/78) | 0.420 | 0.818 |
| 2 | {haiku,opus} | 50.0% (39/78) | 0.393 | 0.792 |
| 2 | {gemma,haiku} | 47.4% (37/78) | 0.366 | 0.815 |
| 1 | {haiku} | 47.4% (37/78) | 0.365 | 0.729 |


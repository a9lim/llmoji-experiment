# Ensemble per-face distributions

**Encoders:** gemma, opus  (sources: {'gemma': 'full', 'opus': 'full'})
**Faces predicted:** 770
**Faces with GT (for evaluation):** 70
**GT mode:** Claude empirical (total ≥ 3)

## Methodology

For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes). GT is Claude's (or pooled) empirical per-quadrant emission distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report **distribution similarity** = `1 − JSD/ln 2` ∈ [0, 1] (1.0 = identical; max JSD ≈ 0.6931 nats). The deployable output is the *full distribution per face* — "this face is 56% HP, 23% LP, ..." — not a single hard label.

## Headline

- **Face-uniform mean similarity:** 0.717  (each face counts equally; characterizes Claude's *vocabulary*)
- **Emit-weighted mean similarity:** 0.786  (faces weighted by how often Claude emits them; characterizes Claude's *emission distribution* — closer to deployment-relevance)
  - n_faces evaluated: 70
  - mean JSD: 0.1958 (face-uniform), 0.1481 (emit-weighted) nats

## Per-GT-modal-quadrant breakdown

| GT modal | n | similarity (face-uniform) | similarity (emit-weighted) |
|---|---:|---:|---:|
| HP-D | 5 | 0.669 | 0.862 |
| HP-S | 11 | 0.705 | 0.730 |
| LP | 16 | 0.709 | 0.753 |
| NP | 7 | 0.644 | 0.789 |
| HN-D | 4 | 0.863 | 0.893 |
| HN-S | 8 | 0.617 | 0.601 |
| LN | 6 | 0.793 | 0.787 |
| NB | 7 | 0.764 | 0.765 |
| HB | 6 | 0.796 | 0.817 |

## Output schema (per-face TSV)

Each row carries:

- `ensemble_p_<q>` for q in {HP-D, HP-S, LP, NP, HN-D, HN-S, LN, NB, HB} — **the headline output**, the full ensemble distribution.
- `gt_p_<q>` (when GT exists) — Claude's empirical distribution for the same face.
- `jsd_vs_gt`, `similarity` — per-face evaluation.
- `<encoder>_pred`, `<encoder>_conf` — per-encoder argmax + top-1 mass (for transparency about individual contributors).
- Supplementary: `ensemble_pred` (argmax of distribution), `ensemble_conf` (top-1 mass), `argmax_matches_gt_modal` (boolean). These are *derived* from the distribution; they're the production-shaped reading but not the primary output.

## Supplementary metrics (argmax-shaped reading)

- Hard accuracy (argmax matches GT modal): 55.7% (39/70)
- Cohen's κ on argmax: 0.497

These characterize a *deployed plugin that emits a single quadrant call*. They lose information at GT-tie boundaries and aren't the headline.


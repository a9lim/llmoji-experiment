# Ensemble per-face distributions

**Encoders:** gemma, qwen, ministral, gpt_oss_20b, granite  (sources: {'gemma': 'full', 'qwen': 'full', 'ministral': 'full', 'gpt_oss_20b': 'full', 'granite': 'full'})
**Faces predicted:** 770
**Faces with GT (for evaluation):** 243
**GT mode:** pooled v3+Claude+wild (total ≥ 3)

## Methodology

For each face the ensemble emits a per-quadrant probability distribution (mean of subset softmaxes). GT is Claude's (or pooled) empirical per-quadrant emission distribution. We compare distribution-to-distribution via Jensen-Shannon divergence and report **distribution similarity** = `1 − JSD/ln 2` ∈ [0, 1] (1.0 = identical; max JSD ≈ 0.6931 nats). The deployable output is the *full distribution per face* — "this face is 56% HP, 23% LP, ..." — not a single hard label.

## Headline

- **Face-uniform mean similarity:** 0.636  (each face counts equally; characterizes Claude's *vocabulary*)
- **Emit-weighted mean similarity:** 0.806  (faces weighted by how often Claude emits them; characterizes Claude's *emission distribution* — closer to deployment-relevance)
  - n_faces evaluated: 243
  - mean JSD: 0.2520 (face-uniform), 0.1343 (emit-weighted) nats

## Per-GT-modal-quadrant breakdown

| GT modal | n | similarity (face-uniform) | similarity (emit-weighted) |
|---|---:|---:|---:|
| HP-D | 23 | 0.629 | 0.679 |
| HP-S | 32 | 0.541 | 0.838 |
| LP | 42 | 0.662 | 0.836 |
| NP | 28 | 0.637 | 0.797 |
| HN-D | 17 | 0.632 | 0.858 |
| HN-S | 33 | 0.564 | 0.665 |
| LN | 30 | 0.686 | 0.774 |
| NB | 23 | 0.751 | 0.830 |
| HB | 15 | 0.667 | 0.822 |

## Output schema (per-face TSV)

Each row carries:

- `ensemble_p_<q>` for q in {HP, LP, HN-D, HN-S, LN, NB} — **the headline output**, the full ensemble distribution.
- `gt_p_<q>` (when GT exists) — Claude's empirical distribution for the same face.
- `jsd_vs_gt`, `similarity` — per-face evaluation.
- `<encoder>_pred`, `<encoder>_conf` — per-encoder argmax + top-1 mass (for transparency about individual contributors).
- Supplementary: `ensemble_pred` (argmax of distribution), `ensemble_conf` (top-1 mass), `argmax_matches_gt_modal` (boolean). These are *derived* from the distribution; they're the production-shaped reading but not the primary output.

## Supplementary metrics (argmax-shaped reading)

- Hard accuracy (argmax matches GT modal): 42.8% (104/243)
- Cohen's κ on argmax: 0.352

These characterize a *deployed plugin that emits a single quadrant call*. They lose information at GT-tie boundaries and aren't the headline.


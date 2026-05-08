# Findings

Current results worth citing. Historical framings that were replaced are
summarized in [`previous-experiments.md`](previous-experiments.md).

## Current Methodology

- **Taxonomy**: v4 9-cell PAD registry:
  `HP-D / HP-S / LP / NP / HN-D / HN-S / LN / NB / HB`.
- **Hidden-state representation**: layer-stack concat of every probe
  layer's `h_first`.
- **Evaluation**: distribution-vs-distribution comparison with
  Jensen-Shannon similarity:
  `similarity = 1 - JSD(pred, gt) / ln(2)`.
- **Reported means**: face-uniform for vocabulary coverage and
  emit-weighted for deployment relevance.
- **Claude-GT loader**:
  `claude_gt.load_claude_gt_distribution()`, with prompt-id based
  remapping so old v3 rows read cleanly into the v4 9-cell taxonomy.

## Best Face-Likelihood Ensembles

Pooled-GT is the deployment-shaped denominator: v3 local emits,
Claude-GT, introspection, and wild contributor faces, with floor 3
emits anywhere.

| subset | best ensemble | face-uniform | emit-weighted |
|---|---|---:|---:|
| pooled-GT, floor 3 (`n=54`) | `{gemma, gemma_v7primed, ministral, opus}` | 0.832 | 0.904 |
| strict Claude-GT (`n=40`) | `{gemma_v7primed, opus}` | 0.792 | 0.820 |

The old `{gemma_v7primed, haiku}` and early `{gemma_v7primed, opus}`
headlines are superseded on the broader deployment view. The strict
pair survives as the cheaper drop-in when the caller only cares about
faces Claude itself emitted at least 3 times.

### Solo Encoders

Strict Claude-GT (`n=40`, 9 encoders, no JP rinna variants):

| encoder | face-uniform | emit-weighted |
|---|---:|---:|
| gemma_v7primed | 0.790 | 0.798 |
| gemma | 0.754 | 0.742 |
| opus | 0.736 | 0.781 |
| haiku | 0.675 | 0.702 |
| gpt_oss_20b | 0.588 | 0.643 |
| bol | 0.549 | 0.455 |
| ministral | 0.537 | 0.623 |
| granite | 0.520 | 0.575 |
| qwen | 0.494 | 0.546 |

Pooled-GT (`n=54`) flips the top solo encoder to Opus:

| encoder | face-uniform | emit-weighted |
|---|---:|---:|
| opus | 0.784 | 0.859 |
| gemma_v7primed | 0.769 | 0.792 |
| gemma | 0.763 | 0.787 |
| haiku | 0.715 | 0.815 |
| gpt_oss_20b | 0.700 | 0.800 |
| ministral | 0.669 | 0.780 |

Interpretation: pure Opus introspection scales better than expected,
especially on broader wild-face vocabulary. Gemma v7 priming remains
the strongest LM-head encoder on the strict subset.

## Claude-GT Collection

Sequential Claude-GT scaling is complete.

- Naturalistic: **1360 Opus 4.7 rows** across v3 plus v4-extension
  cells.
- Introspection: **120 Opus 4.7 rows** under v7 introspection.
- Total: **1480 rows** in merged `emotional_raw.jsonl` files:
  `data/harness/claude/` and `data/harness/claude_intro_v7/`.
- Stale-row removals were backfilled with `--fill-gaps`: 61
  naturalistic rows and 8 introspection rows.

Saturation gate after recalibration:

| cell group | outcome |
|---|---|
| HN-D | exited after r4, gate-driven |
| LN | exited after r6 by amendment |
| HP, LP, HN-S, NB | ran to r7 cap |
| HP-D, NP, HB | ran to r7 cap in the v4-extension pilot |

`PER_Q_JS_MAX` is now 0.10. The 0.05 threshold was too tight for the
observed per-run noise floor, especially on higher-entropy v4-new cells.
Detail: [`2026-05-08-saturation-threshold-recal.md`](2026-05-08-saturation-threshold-recal.md).

## Local Hidden-State Geometry

Across `gemma`, `qwen`, `ministral`, `gpt_oss_20b`, and `granite`, the
same affect geometry recurs under different tokenizers and model
families. PCA axes are model-specific, but quadrant centroids preserve a
stable Russell/PAD arrangement.

Layer-stack PCA variance, first three PCs:

| model | PC1 | PC2 | PC3 |
|---|---:|---:|---:|
| gemma | 30.2% | 15.7% | 9.3% |
| qwen | 30.5% | 17.3% | 9.5% |
| ministral | 21.9% | 14.0% | 8.4% |
| granite | 27.6% | 14.1% | 7.5% |
| gpt_oss_20b | 15.8% | 12.5% | 9.5% |

Prompt-grouped predictiveness:

| metric | gemma | qwen | ministral | granite | gpt_oss |
|---|---:|---:|---:|---:|---:|
| hidden -> quadrant | 0.992 | 0.985 | 0.984 | 0.980 | 0.876 |
| face -> quadrant | 0.806 | 0.785 | ~0.43 | ~0.55 | ~0.40 |
| face-centroid R2 | 0.55 | 0.52 | 0.13 | 0.38 | 0.13 |

The kaomoji is a substantial but partial readout. Hidden state predicts
the quadrant almost perfectly; face alone recovers much of the quadrant
signal for tighter vocabularies, but breaks down when a model spreads
across too many low-count faces.

## Introspection V7

`preambles/introspection_v7.txt` is canonical for gemma priming and is
baked into `config.INTROSPECTION_PREAMBLE`. It replaces the normal
kaomoji instruction through `instruction_override`; it is not
concatenated with the normal ask.

Use v7 with gemma only unless a new pilot says otherwise. It degrades
qwen badly: lower emit rate, vocabulary collapse, and opposite-valence
collisions.

Under soft-everywhere evaluation, v7 priming helps. The old hard-argmax
read made primed gemma look worse because it punished diffuse
distributional matches.

## Harness BoL And Use/Read/Act

The harness corpus now uses BoL: a 50-dimensional count-weighted bag over
the `llmoji` v2 LEXICON, pooled per canonical face. This replaced the
old MiniLM-on-prose eriskii-parity representation.

Three channels:

- **use**: Claude-GT emission distributions.
- **read**: Opus or Haiku cold introspection on the face symbol.
- **act**: BoL over in-context wild deployment synthesis.

Structural findings:

- Opus and Haiku read each other closely (`0.906` cross-similarity on
  the shared subset).
- GT vs introspection improves under emit-weighting, while GT vs BoL
  worsens under emit-weighting.
- The `110` agreement pattern (Opus and Haiku read GT, BoL differs)
  covers 27.4% of emit volume on the shared `n=40` subset.
- BoL appears positivity-biased on negative-affect contexts. Treat it as
  diagnostic, not as direct deployment-state ground truth.

Detail: [`2026-05-06-use-read-act-channels.md`](2026-05-06-use-read-act-channels.md).

## Wild Residuals

`scripts/67_wild_residual.py` clusters HF-corpus faces in BoL space.
The k=6 and k=9 reports show structure beyond the original six
Russell cells: separate positive registers, high-arousal negative
clusters, and a broader HN-S tail than strict elicitation surfaced.

Current tracked generated report:
[`2026-05-05-residual-state-axes.md`](2026-05-05-residual-state-axes.md).

## Deferred Cells

NN `(a=0, v=-1)` and LB `(a=-1, v=0)` are coordinate-real but not
promoted. The 2026-05-07 pilot did not show enough hidden-state or
face-distribution evidence to justify moving from 9 to 11 cells.

Keep their prompts and promotion criteria parked in
[`2026-05-06-nn-lb-future-cells.md`](2026-05-06-nn-lb-future-cells.md).

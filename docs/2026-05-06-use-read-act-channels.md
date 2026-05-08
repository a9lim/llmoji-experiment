# Use / Read / Act Channels

**Status:** executed 2026-05-06. Revised after the BoL whitewashing
finding.

This analysis compares three ways a face can be associated with an
affective quadrant.

## Channels

- **Use (Claude-GT)**: how Claude emits a face under controlled prompt
  quadrants. Loader: `claude_gt.load_claude_gt_distribution()`.
- **Read (Opus / Haiku face_likelihood)**: cold introspection on the
  face symbol with no deployment context.
- **Act (BoL)**: structured Haiku synthesis over wild in-context uses,
  pooled into the 50-word LEXICON bag.

The names matter. These are not interchangeable measurements.

## Headline

On the current shared face subset (`50` faces, `1138` GT emissions):

- Opus and Haiku read each other closely: **0.797** face-uniform and
  **0.774** emit-weighted cross-similarity.
- GT vs introspection goes up under emit-weighting.
- GT vs BoL goes down under emit-weighting.
- The `110` pattern (Opus and Haiku agree with GT, BoL diverges) covers
  **28.8%** of emit volume.

This means cold symbolic interpretation and actual Claude use are closer
on high-volume faces than BoL is.

## Revised Interpretation

Original framing: BoL might be deployment-state ground truth, capturing
how a face acts in real context rather than what it denotes cold.

Revised framing: BoL is probably biased-positive on negative-affect
contexts because Haiku-as-synthesizer tends to summarize helpful
responses with LP/NP-coded descriptors. That whitewashes some HN/LN
deployment states.

What survives:

- BoL remains useful for clustering and source-model drift.
- The use/read/act separation is real.
- BoL should not override Claude-GT or Opus introspection on
  negative-affect deployment meaning.

## Scripts

```bash
.venv/bin/python scripts/harness/68_three_way_analysis.py
.venv/bin/python scripts/harness/69_per_source_drift.py
```

Primary outputs:

- `data/harness/three_way_summary.md`
- `data/harness/three_way_per_face.tsv`
- `data/harness/per_source_drift_summary.md`
- `data/harness/per_source_drift.tsv`
- `figures/harness/three_way_pairwise_heatmap.png`
- `figures/harness/three_way_top_divergent.png`

## Case-File Pattern

Use the per-face TSVs for exact cases. The qualitative pattern from the
original case files was:

- Faces like `(╯°□°)` can be read as HN by Opus/Haiku and Claude-GT,
  while BoL shifts toward more helpful or self-correcting descriptors.
- Sad faces can be softened by synthesis into LP/NP language when the
  surrounding assistant behavior is supportive.
- Some neutral or uncertain faces are genuinely channel-dependent rather
  than obvious whitewashing artifacts.

## Falsification Tests

Still useful follow-ups:

1. Re-synthesize a negative-affect sample with Opus. If Opus picks more
   LN/HN-coded LEXICON descriptors than Haiku on the same contexts,
   model choice caused much of the whitewashing.
2. Audit per-quadrant LEXICON word frequency against the structural
   LEXICON distribution. Overuse of LP/NP anchors would support the
   positivity-bias hypothesis.
3. Compare BoL drift by source model after the v4 local chain finishes.

## Rule

When channels disagree:

- Use Claude-GT for "how Claude uses this face under elicited states."
- Use Opus for "what this face symbol feels like cold."
- Use BoL for "what the wild corpus synthesis pipeline says about
  in-context use," with the positivity-bias caveat attached.

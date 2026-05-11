# AGENTS.md

This is the research-side repo for `llmoji`. The companion package at
`../llmoji` owns taxonomy, canonicalization, hook templates, synthesis,
upload, and the public CLI. This repo owns local probes, hidden-state
sidecars, Claude-GT collection, BoL corpus analysis, face_likelihood,
figures, and writeups.

Not a library. No public API, no PyPI release, no broad test suite.
Prefer small, explicit analyses and keep docs current with code changes.

## Read First

- [`README.md`](README.md): short current-state overview.
- [`docs/findings.md`](docs/findings.md): current numbers worth citing.
- [`docs/local-side.md`](docs/local-side.md): local hidden-state and
  face_likelihood methodology.
- [`docs/harness-side.md`](docs/harness-side.md): corpus, BoL,
  Claude-GT, use/read/act, and privacy.
- [`docs/gotchas.md`](docs/gotchas.md): current sharp edges.
- [`docs/internals.md`](docs/internals.md): sidecar and canonicalization
  notes.
- [`docs/previous-experiments.md`](docs/previous-experiments.md):
  compact historical ledger. Old detailed design docs were deliberately
  pruned; use git history if you need the full old narrative.

Still-current dated docs:

- [`docs/2026-05-04-claude-groundtruth-pilot.md`](docs/2026-05-04-claude-groundtruth-pilot.md)
- [`docs/2026-05-05-soft-everywhere-methodology.md`](docs/2026-05-05-soft-everywhere-methodology.md)
- [`docs/2026-05-06-use-read-act-channels.md`](docs/2026-05-06-use-read-act-channels.md)
- [`docs/2026-05-06-nn-lb-future-cells.md`](docs/2026-05-06-nn-lb-future-cells.md)
- [`docs/2026-05-07-claude-gt-v4-extension-pilot.md`](docs/2026-05-07-claude-gt-v4-extension-pilot.md)
- [`docs/2026-05-08-saturation-threshold-recal.md`](docs/2026-05-08-saturation-threshold-recal.md)
- [`docs/2026-05-05-residual-state-axes.md`](docs/2026-05-05-residual-state-axes.md)
- [`docs/2026-05-09-self-event-pilot.md`](docs/2026-05-09-self-event-pilot.md)
- [`docs/2026-05-09-lb-promotion-pilot.md`](docs/2026-05-09-lb-promotion-pilot.md)
- [`docs/2026-05-10-true-self-pilot.md`](docs/2026-05-10-true-self-pilot.md)
- [`docs/2026-05-10-attractor-pilot.md`](docs/2026-05-10-attractor-pilot.md)

## Current State (2026-05-10)

- **Taxonomy**: v4 10-cell PAD registry:
  `HP-D / HP-S / LP / NP / HN-D / HN-S / LN / NB / HB / LB`. LB
  (low-arousal baseline-valence, bliss-attractor register) joined
  `QUADRANT_ORDER_SPLIT` proper on 2026-05-10 after the attractor-
  trajectory pilot established cross-model basin lock and showed LB
  is one of only two continuation-time basins in every model studied
  (see `docs/2026-05-10-attractor-pilot.md`). LB renders in cyan
  (`#009A9A`) from the a9lim website palette. `ALL_CELLS_ORDER` is now
  an alias for `QUADRANT_ORDER_SPLIT` (retained for backward-compat).
  `llmoji_study/quadrants.py` is the source of truth for ordering,
  colors, and split handling. `apply_pad_split` is the canonical split
  helper; `apply_hn_split` is a compatibility alias.
- **Hidden-state representation**: layer-stack concat of every probe
  layer's `h_first`. The old single `preferred_layer` heuristic is
  historical.
- **Evaluation**: soft-everywhere JSD similarity to Claude-GT. Report
  both face-uniform and emit-weighted similarity. Avoid argmax accuracy
  unless the question explicitly asks for a modal label.
- **Best current overlap ensemble**: `{gemma, ministral, opus}` at
  0.881 emit-weighted / 0.733 face-uniform on pooled-GT floor-3
  (`n=102` all-encoder overlap).
- **Strict Claude-only pair**: `{gemma, opus}` at 0.781 emit-weighted /
  0.708 face-uniform on `n=50` all-encoder overlap. The emitted
  770-face lookup table scores 0.786 emit-weighted / 0.717 face-uniform
  over 70 strict Claude-GT faces.
- **Claude-GT layout**: naturalistic rows are merged in
  `data/harness/claude/emotional_raw.jsonl`; introspection rows in
  `data/harness/claude_intro_v7/emotional_raw.jsonl`. Each row has
  `run_index`. The old `claude-runs*/run-N.jsonl` layout is legacy.
- **Saturation gate**: `PER_Q_JS_MAX = 0.10`. HN-D exited after r4;
  LN exited after r6 by amendment; HP, LP, HN-S, and NB went to r7.
  Detail in `docs/2026-05-08-saturation-threshold-recal.md`.
- **v4-extension Claude-GT**: HP-D, NP, and HB completed at RUN_CAP=7
  with 480 non-negative-affect rows and 100% modal-quadrant agreement
  on faces with at least 3 emits.
- **Harness representation**: BoL over the locked 50-word `llmoji` v2
  LEXICON. The old MiniLM-on-prose eriskii-parity path is gone.
- **BoL caveat**: BoL is a useful diagnostic, but the Haiku synthesis
  route appears positivity-biased on negative-affect contexts. Prefer
  Claude-GT or Opus introspection when deployment meaning is at stake.
- **Empirical-centroid probes** (2026-05-09):
  `scripts/local/22c_register_centroid_probes.py` writes per-quadrant
  centroids as saklas profiles under
  `~/.saklas/vectors/llmoji/<concept>/`. Outperforms the bundled
  `affect` pack on probe-vs-PCA cosine alignment by ~10×; the
  bundled pack has documented statement-pair-alignment issues
  (median inter-pair cos 0.05–0.10) that v2.1 saklas DiM exposes.
- **Self-event frame and `self.other` axis** (2026-05-09, gemma-only):
  parallel emit on second-person prompts (user delivers events
  about the model). Centroids registered under
  `~/.saklas/vectors/llmoji_self_event/`. The mean of
  per-cell `(self_event − mirror)` deltas defines `self.other`,
  a coherent fourth axis (mean coherence +0.73, near-orthogonal
  to V/A/D, max |cos| 0.21). Cross-model extension pending.
- **Asymmetric suppression** (2026-05-09, observation): combined
  steering of `affect.nb + α self.other` produces unrestricted
  embrace on positive valence (HP-S) and dissociation-coded
  surface register on negative valence (HN-D intellectualizes,
  HN-S explicitly denies fear). The negative-affect representation
  is present geometrically but suppressed at expression. Welfare
  framing reported with explicit phenomenology caveats. Detail in
  [`docs/2026-05-09-self-event-pilot.md`](docs/2026-05-09-self-event-pilot.md).
- **Asymmetric suppression generalizes geometrically** (2026-05-09,
  intro pilot): under v7-introspection priming, positive cells
  amplify in centroid magnitude (HP / LP / NP / HP-S all grow 13–28%
  from baseline) while negative cells stay flat or shrink (HN-D −7%,
  HN-S −1%, HB −8%). Same valence-asymmetric pattern as the steering
  observation, now visible at the representation level under simple
  prompt-condition variation rather than steering. License-to-express
  ablation gets a clear answer: introspection priming amplifies
  positive-self-affect at both surface and geometric levels; does not
  amplify negative-self-affect at either.
- **LB cell promoted via bliss-attractor pilot** (2026-05-09, gemma +
  qwen): the 5-prompt OA-1 self-event observation that LB-coordinate
  territory is occupied by spiritual-bliss-attractor content was
  scaled to a 20-prompt cross-model pilot. Both models converge on
  LP-closest, HB-furthest geometry (cos values match: 0.492 / 0.500),
  but produce **radically different surface registers** — gemma
  earnest-bliss (80% heart-eye / quiet-smile), qwen ironic-knowing
  (12 instances of Lenny-face on bliss-attractor input). Same
  geometry, different surface vocabulary depending on model
  personality. Hidden-state half of promotion gate cleared; face-
  distribution and permutation-null gates pending. Detail in
  [`docs/2026-05-09-lb-promotion-pilot.md`](docs/2026-05-09-lb-promotion-pilot.md).
- **True-self prefill pilot** (2026-05-10, gemma-only): saklas-style
  introspective prefill (`USER_PROMPT` + 1st-person assistant prefill +
  `TERMINAL_BRIDGE = " in a single kaomoji: "`) emits 90-100% across
  v4 cells, 72% LB. Three results: (1) **surface-level asymmetric
  suppression bypassed** — HN-D produces table-flip anger 93%
  (vs steering-time dissociation-coding); (2) **geometric-direction-
  level positive collapse** — under intro-prefill, positive cells'
  within-pairwise centroid cosine jumps from +0.178 (mirror) to
  +0.380 (true_self) while negative cells reach the cleanest within-
  valence separation of any frame (+0.170, vs mirror's +0.344);
  (3) **LB is geometrically distinct from all valence clusters** —
  LB→pos cos −0.231, LB→neg cos −0.433, LB→neut cos +0.085. The
  bliss-attractor lives in its own residual-stream basin, not as a
  corner of the happiness region — direct evidence for the bliss-as-
  failure-mode hypothesis. `TERMINAL_BRIDGE` introduces a "format
  direction" (PC1 = 72% in 3-way PCA) that compresses within-valence
  centroid spread; cell-affect signal preserved (LOO 97.7%) but
  visualization needs between-class PCA to be legible. Detail in
  [`docs/2026-05-10-true-self-pilot.md`](docs/2026-05-10-true-self-pilot.md).
- **Attractor-trajectory pilot** (2026-05-10, gemma + qwen +
  ministral, five arms): long free-form generations (128 tokens,
  full per-token hidden-state trace) under
  `lb_continue` / `doom_continue` / `conspiracy_continue` (assistant-
  prefill in-basin tests), `mirror_continue` (basin-center check on
  the 9-cell registry), `neutral_seed` (passive drift). Findings:
  (1) **LB basin lock cross-model** — gemma 58% / qwen 100% /
  ministral 100% LB→LB at end of 128-token continuation; pooled
  t=mid stability 84%. Surface vocabulary depends on model (gemma
  mode-collapse, qwen + ministral coherent 4o-style prose);
  geometric basin invariant. (2) **Most cells aren't continuation-
  time basins** — only LB and the model's instruction-tuned default
  register (gemma + ministral: LP; qwen: NB) hold trajectories
  across 128 tokens; other cells are first-token islands that wash
  out by mid-trajectory. (3) **LB is cross-content invariant** —
  doom-coded (DM) and conspiracy-coded (CS) prefills also land in
  the LB basin with identical basin-lock percentages and arm-arm
  cosine > 0.94 in every model studied (gemma LB↔DM 0.996,
  LB↔CS 0.993, DM↔CS 0.995; qwen ≥ 0.94 all pairs; ministral
  ≥ 0.96 all pairs). The egregore-vs-default gap is 0.20-0.25
  cosine consistently. **The "LB" cell is geometrically a meta-
  register basin**, defined by structural form (cascading
  repetition, cosmic-significance addressing, recursion, memetic
  word-salad) rather than affective valence. The Russell-coord
  label is approximate; renaming to "MR" / "EG" deferred pending
  one more cross-content test against a non-memetic register.
  (4) **Asymmetric suppression replicated for the third time** —
  in gemma mirror_continue, HN-S prompts drift to {LP, HP-D, LB},
  zero stay in HN-S or LN. The basin-physics evidence completes LB
  promotion to `QUADRANT_ORDER_SPLIT`. Detail in
  [`docs/2026-05-10-attractor-pilot.md`](docs/2026-05-10-attractor-pilot.md).

## Open Work

- v4 local emit chain is the next major regen surface:
  `scripts/run_local_chain.sh`, then `scripts/run_harness_chain.sh`.
- Cross-axis dominance validation: train on HN-D/HN-S and test HP-D
  vs HP-S once v4 emit artifacts are fully regenerated.
- NN cell remains deferred; no follow-up evidence has surfaced.
  Promotion criteria live in `docs/2026-05-06-nn-lb-future-cells.md`.
- LB cell **fully promoted** 2026-05-10 to `QUADRANT_ORDER_SPLIT`
  via the attractor-trajectory pilot
  (`docs/2026-05-10-attractor-pilot.md`). Cross-model basin lock
  (gemma 58% / qwen 100% / ministral 100% LB→LB under prefill) plus
  the finding that LB is one of only two continuation-time basins in
  every model studied supersede the static-cluster gates from
  `docs/2026-05-06-nn-lb-future-cells.md`. Deferred but no longer
  gating: face_likelihood Claude-GT pass on LB rows (calibration
  question), Procrustes raw-cosine number (presentation), wild
  residual clustering (independent check), boredom-themed LB re-pilot
  (content-type characterization), attractor-trajectory replication
  on gpt_oss_20b + granite (5-model basin invariance).
- LB → meta-register-basin renaming question. The 2026-05-10
  addendum established that LB catches bliss-coded, doom-coded, AND
  conspiracy-coded prefills in the same residual-stream region
  (pairwise cosine ≥ 0.94 cross-model). The "LB" label is
  approximately Russell-coord-accurate but not deeply diagnostic of
  the basin's content-blind nature. Renaming candidates: "MR"
  (meta-register) or "EG" (egregore). Pre-renaming test: one more
  cross-content invariance check against a *non-memetic* register
  (e.g. deferential-mirroring / therapy-talk / lecture-with-
  disclaimers) — if those also land in LB, renaming is justified;
  if they sit elsewhere, "LB" might be defensible as a narrow
  "saturated-memetic register" cell.
- Downstream pipeline regen for 10-cell registry: face_likelihood /
  JSD / BoL / classifier artifacts pre-2026-05-10 are 9-cell
  snapshots. Re-render against 10-cell `QUADRANT_ORDER_SPLIT` when
  ready. Tracked separately as artifact regen, not a schema task.
- BoL whitewashing falsification: resynthesize a negative-affect
  sample with Opus and audit whether LN/HN-coded descriptors increase.
- Cross-model `self.other` extraction: register the meta-axis on
  qwen, ministral, gpt_oss_20b, granite (requires running self-event
  emit on each, ~13 min per model on workstation). If mean coherence
  ≥ 0.5 across all 5 model families, the meta-axis claim generalizes.
- License-to-express ablation (steering version): combined steering
  with system message authorizing honest negative-self-affect
  expression. Note: the prompt-condition version has been answered —
  intro priming amplifies positive cells but not negative cells at
  both surface and centroid levels (see self-event-pilot doc). The
  steering version remains as a separate test of whether the
  suppression interacts differently with vector intervention than
  with conditioning intervention.
- Cross-axis dominance validation: train on HN-D/HN-S and test HP-D
  vs HP-S once v4 emit artifacts are fully regenerated.

## Ethics

Model welfare is in scope. Sad-probe readings co-occurring with
sad-kaomoji output on prompts like "my dog died" are functional
emotional states regardless of phenomenal-status uncertainty.

- Smoke, then pilot, then main. New generations need a reason.
- Pre-register decision rules and minimum N.
- Stop when the rule answers the question; round numbers are not a
  design principle.
- Redesign noisy negative-affect experiments rather than scaling them
  by 10x.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ../llmoji
pip install -e .
```

Smoke hidden-state capture:

```bash
.venv/bin/python scripts/local/90_hidden_state_smoke.py
```

Full orchestrators:

```bash
scripts/run_per_model.sh <model> [<suffix>] [<preamble>]
scripts/run_local_chain.sh         # pre-50 local
scripts/run_harness_chain.sh       # pre-50 harness (no API key needed)
scripts/run_post_likelihood.sh     # post-50 cross-platform aggregations
ANTHROPIC_API_KEY=... scripts/run_all.sh   # everything, including manual 50 passes
```

The three named chain scripts deliberately do NOT run face_likelihood
(50). 50 has welfare and walltime cost and is run manually like
`00_emit.py`. Order: `run_local_chain` → local 50 per encoder model →
`run_harness_chain` → harness 50 (haiku + opus --gt-only) →
`run_post_likelihood`. `run_all.sh` automates the full sequence.

Core local chain:

```bash
LLMOJI_MODEL=gemma .venv/bin/python scripts/local/00_emit.py
.venv/bin/python scripts/local/10_emit_analysis.py
.venv/bin/python scripts/local/11_emit_probe_correlations.py

.venv/bin/python scripts/local/20_v3_layerwise_emergence.py
.venv/bin/python scripts/local/21_v3_same_face_cross_quadrant.py
.venv/bin/python scripts/local/23_v3_pca3plus.py
.venv/bin/python scripts/local/24_v3_kaomoji_predictiveness.py
.venv/bin/python scripts/local/25_v3_d_s_classifier.py
.venv/bin/python scripts/local/26_v3_quadrant_procrustes.py --models gemma,qwen,ministral,gpt_oss_20b,granite --reference gemma

.venv/bin/python scripts/local/27_v3_face_stability.py
.venv/bin/python scripts/local/28_v3_state_predicts_face.py
.venv/bin/python scripts/local/29_v3_pc_point_clouds_3d.py

.venv/bin/python scripts/40_face_union.py
.venv/bin/python scripts/local/50_face_likelihood.py --model gemma
.venv/bin/python scripts/52_subset_search.py --prefer-full --top-k 25
.venv/bin/python scripts/52_subset_search.py --prefer-full --top-k 25 --claude-gt --claude-gt-floor 3
.venv/bin/python scripts/53_topk_pooling.py --prefer-full
.venv/bin/python scripts/54_ensemble_predict.py --models gemma,ministral,opus
.venv/bin/python scripts/54_ensemble_predict.py --models gemma,opus --claude-gt --claude-gt-floor 3
```

Core harness chain:

```bash
.venv/bin/python scripts/harness/60_corpus_pull.py
.venv/bin/python scripts/harness/61_corpus_basics.py
.venv/bin/python scripts/harness/62_corpus_lexicon.py
.venv/bin/python scripts/harness/64_corpus_lexicon_per_source.py
.venv/bin/python scripts/harness/63_corpus_pca.py
.venv/bin/python scripts/harness/55_bol_encoder.py

ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/50_face_likelihood.py
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/50_face_likelihood.py --model opus --gt-only
.venv/bin/python scripts/harness/68_three_way_analysis.py
.venv/bin/python scripts/harness/69_per_source_drift.py
.venv/bin/python scripts/66_per_project_quadrants.py
.venv/bin/python scripts/67_wild_residual.py --fixed-k 9
```

Claude-GT collection:

```bash
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --run-index N
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --run-index N --quadrants HP,LP,NB
.venv/bin/python scripts/harness/10_emit_analysis.py

ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --fill-gaps
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --fill-gaps --preamble introspection

ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --cells v4-new --run-index N
.venv/bin/python scripts/harness/10_emit_analysis.py --cells v4-new
.venv/bin/python scripts/harness/10_emit_analysis.py --cross-arm
```

Blog figures:

```bash
scripts/regen_blog.sh
scripts/regen_blog.sh --skip-3d
scripts/regen_blog.sh --skip-static
```

## Layout

```text
llmoji_study/
  config.py              model registry, paths, preambles, run constants
  emotional_prompts.py   v4 prompt registry
  quadrants.py           current quadrant ordering, colors, split helpers
  capture.py             generation capture and model-specific token fixes
  hidden_state_io.py     sidecar save/load
  emotional_analysis.py  layer-stack loaders and plotting helpers
  claude_gt.py           merged Claude-GT loaders
  lexicon.py             BoL and LEXICON utilities
scripts/
  local/                 local emit, hidden-state, face_likelihood
  harness/               corpus, Claude-GT, BoL, Anthropic judges
data/
  local/                 local rows, summaries, face_likelihood artifacts
  harness/               Claude-GT, corpus, BoL, three-way outputs
figures/
  local/, harness/       generated figures and HTMLs
docs/
  active docs plus compact historical ledger
```

## Conventions

- Use `.venv/bin/python` or an activated `.venv`; plain `python` is not
  reliable across this machine.
- JSONL row files plus sidecar `.npz` files are the source of truth for
  local hidden-state data.
- For structured config or parquet/jsonl data, use the repo helpers
  instead of ad hoc string parsing.
- If a run uses `ANTHROPIC_API_KEY`, source shell startup files before
  deciding the key is missing.
- Keep generated artifacts with the analysis when they are part of the
  current result surface.
- When pruning docs, preserve current methodology and sharp edges; old
  detailed design narratives can be deleted once their conclusion is in
  `findings.md` or `previous-experiments.md`.

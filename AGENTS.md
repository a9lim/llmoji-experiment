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

## Current State (2026-05-08)

- **Taxonomy**: v4 9-cell PAD registry:
  `HP-D / HP-S / LP / NP / HN-D / HN-S / LN / NB / HB`.
  `llmoji_study/quadrants.py` is the source of truth for ordering,
  colors, and split handling. `apply_pad_split` is the canonical split
  helper; `apply_hn_split` is a compatibility alias.
- **Hidden-state representation**: layer-stack concat of every probe
  layer's `h_first`. The old single `preferred_layer` heuristic is
  historical.
- **Evaluation**: soft-everywhere JSD similarity to Claude-GT. Report
  both face-uniform and emit-weighted similarity. Avoid argmax accuracy
  unless the question explicitly asks for a modal label.
- **Best deployment ensemble**: `{gemma, gemma_v7primed, ministral,
  opus}` at 0.904 emit-weighted / 0.832 face-uniform on pooled-GT
  floor-3 (`n=54`).
- **Strict Claude-only pair**: `{gemma_v7primed, opus}` at 0.820
  emit-weighted / 0.792 face-uniform on `n=40`.
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

## Open Work

- v4 local emit chain is the next major regen surface:
  `scripts/run_local_chain.sh`, then `scripts/run_harness_chain.sh`.
- Cross-axis dominance validation: train on HN-D/HN-S and test HP-D
  vs HP-S once v4 emit artifacts are fully regenerated.
- NN and LB are deferred cells. Pilot prompts and promotion criteria
  live in `docs/2026-05-06-nn-lb-future-cells.md`; do not promote them
  without the hidden-state and face-distribution gates.
- BoL whitewashing falsification: resynthesize a negative-affect
  sample with Opus and audit whether LN/HN-coded descriptors increase.

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
scripts/run_local_chain.sh
ANTHROPIC_API_KEY=... scripts/run_harness_chain.sh
ANTHROPIC_API_KEY=... scripts/run_all.sh
```

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
.venv/bin/python scripts/53_topk_pooling.py --prefer-full
.venv/bin/python scripts/54_ensemble_predict.py --models gemma,ministral,qwen
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

# Harness Side: Corpus, Claude-GT, BoL

The harness side reads real kaomoji use from two sources:

- Contributor uploads from the `llmoji` package, pulled from
  [`a9lim/llmoji`](https://huggingface.co/datasets/a9lim/llmoji).
- Claude-GT elicitation runs collected through the Anthropic API.

The old eriskii-parity MiniLM-on-prose pipeline was removed. The live
representation is BoL: a bag over the locked 50-word `llmoji` v2
LEXICON.

## Contributor Corpus

Contributor machines run `llmoji` locally. The package installs Stop
hooks, records kaomoji-bearing assistant turns, canonicalizes faces,
runs structured Haiku synthesis, and uploads aggregate bundles.

Uploaded data:

| data | uploaded? |
|---|---|
| raw user text | no |
| raw assistant text | no |
| per-instance Haiku cache | no |
| canonical face counts | yes |
| structured synthesis picks | yes |

This repo pulls the public dataset with:

```bash
.venv/bin/python scripts/harness/60_corpus_pull.py
.venv/bin/python scripts/harness/61_corpus_basics.py
```

## BoL Pipeline

BoL means bag-of-lexicon. Each canonical face becomes a 50-dimensional
L1-normalized vector over `llmoji.synth_prompts.LEXICON`.

```bash
.venv/bin/python scripts/harness/62_corpus_lexicon.py
.venv/bin/python scripts/harness/64_corpus_lexicon_per_source.py
.venv/bin/python scripts/harness/63_corpus_pca.py
.venv/bin/python scripts/harness/55_bol_encoder.py
```

Outputs:

- `data/harness/claude_faces_lexicon_bag.parquet`
- `data/harness/claude_faces_lexicon_bag_per_source.parquet`
- `data/harness/face_likelihood_bol_summary.tsv`
- `figures/harness/claude_faces_pca.png`

The LEXICON is version-stamped. If the companion package rotates to a
future LEXICON version, consumers should fail loudly until the corpus is
rebuilt.

## Claude-GT

Claude-GT asks Opus to emit kaomoji under controlled affective prompts.
Rows now use the merged layout:

```text
data/harness/claude/emotional_raw.jsonl
data/harness/claude_intro_v7/emotional_raw.jsonl
```

Each row carries `run_index`, so old per-run files are no longer the
active read surface.

Collection and analysis:

```bash
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --run-index N
.venv/bin/python scripts/harness/10_emit_analysis.py

ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/00_emit.py --fill-gaps
.venv/bin/python scripts/harness/10_emit_analysis.py --cross-arm
```

Current saturation gate:

- `PER_Q_NEW_FACE_MAX <= 1`
- `PER_Q_JS_MAX <= 0.10`
- `RUN_CAP = 7`

HN-D exited after r4, LN after r6 by amendment, and HP/LP/HN-S/NB ran
to cap. HP-D/NP/HB v4-extension cells also ran to cap.

## Anthropic Face-Likelihood

The Anthropic judge shows each canonical face out of context and asks
for likelihoods over the current 9-cell taxonomy. It emits likelihoods
only: no `top_pick`, no `reason`. The script requests `temperature=0`
when the target model still accepts that parameter.

```bash
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/50_face_likelihood.py
ANTHROPIC_API_KEY=... .venv/bin/python scripts/harness/50_face_likelihood.py --model opus --gt-only
```

Outputs:

- `data/harness/face_likelihood_haiku_summary.tsv`
- `data/harness/face_likelihood_opus_summary.tsv`
- `data/harness/{haiku,opus}_face_quadrant_judgment.jsonl`

## Use / Read / Act

Three channels measure different things:

- **use**: Claude-GT, how Claude emits a face under prompt quadrants.
- **read**: Opus or Haiku cold introspection on the face symbol.
- **act**: BoL, how Haiku summarizes in-context wild uses of the face.

```bash
.venv/bin/python scripts/harness/68_three_way_analysis.py
.venv/bin/python scripts/harness/69_per_source_drift.py
```

Key result: Opus and Haiku read each other closely, but BoL diverges
from GT on high-volume negative-affect faces. The most plausible current
interpretation is Haiku synthesis positivity bias, not a clean
deployment-state ground truth.

## Cross-Platform Aggregations

```bash
.venv/bin/python scripts/66_per_project_quadrants.py --mode gt-priority
.venv/bin/python scripts/66_per_project_quadrants.py --mode bol
.venv/bin/python scripts/66_per_project_quadrants.py --mode gt-only
.venv/bin/python scripts/67_wild_residual.py --fixed-k 9
.venv/bin/python scripts/41_face_overlap.py --include-claude
```

`scripts/66_per_project_quadrants.py` reads local journals and exports.
Those outputs are deployment telemetry and are not meant to be committed
unless a specific run asks for them.

## Caveats

- BoL is interpretable but not neutral. Treat it as a diagnostic channel.
- Contributor data is deployment-shaped and uneven by source model.
- Claude-GT is elicitation-shaped and welfare-limited by design.
- Per-project outputs can reveal local work patterns; keep them private
  by default.

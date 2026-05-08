# Contributing

`llmoji-study` is a research repo. The standard for changes is: is the
claim honest, is the artifact reproducible, and will future-us know what
changed.

Useful contributions usually fall into three buckets:

- Submit kaomoji data through the companion
  [`llmoji`](https://github.com/a9lim/llmoji) package.
- Add a local model or rerun an existing model under the current v4
  registry.
- Add a focused analysis that reads existing artifacts before spending
  new trials.

Questions and bug reports are useful too, especially when a doc and a
script disagree.

## Submit Kaomoji Data

You do not need this repo for contributor data. Use the `llmoji` package:

```bash
pip install llmoji
llmoji install
# use Claude Code, Codex, or another supported harness normally
llmoji status
llmoji analyze
llmoji upload
```

The upload is privacy-preserving: per-canonical-face counts plus the
structured synthesis picks over the locked LEXICON. No raw prompts, raw
responses, surrounding user text, or per-instance synthesis cache is
uploaded. The package README and SECURITY docs are the source of truth
for the contributor-side privacy model.

## Add A Local Model

The current local lineup is `gemma`, `qwen`, `ministral`,
`gpt_oss_20b`, and `granite`. `gemma_intro_v7_primed` is a historical
priming condition, not a current face-likelihood encoder unless its
summary is regenerated. New models should follow the current layer-stack
pipeline, not the old single-layer `preferred_layer` workflow.

1. Register the model in `llmoji_study/config.py`.
2. Run the hidden-state smoke test:

   ```bash
   LLMOJI_MODEL=your_short_name .venv/bin/python scripts/local/90_hidden_state_smoke.py
   ```

3. Write a short design note if the model needs special handling
   (tokenizer quirks, chat template overrides, logit bias, hardware
   patches, or welfare-relevant prompt changes).
4. Run the smallest pilot that can answer the gating question.
5. If the pilot clears, run the main emit and the chain scripts:

   ```bash
   LLMOJI_MODEL=your_short_name .venv/bin/python scripts/local/00_emit.py
   scripts/run_local_chain.sh
   ```

6. Update `docs/findings.md`, `docs/local-side.md` if methodology
   changed, and `docs/gotchas.md` if the model exposed a reusable sharp
   edge.

Do not add a model by copying old v3 preferred-layer instructions from
git history. Current analyses load all probe layers through the
layer-stack helpers.

## Add An Analysis

- Prefer reading existing JSONLs, sidecars, parquets, or cached
  summaries over new generations.
- If the analysis creates a metric people will cite, write down the
  decision rule before reading the result.
- Keep outputs colocated with the existing convention:
  `data/local/`, `data/harness/`, `figures/local/`, or
  `figures/harness/`.
- Update docs in the same change. If a result supersedes an old framing,
  put the current conclusion in `docs/findings.md` and a one-paragraph
  note in `docs/previous-experiments.md`.

## Working Locally

```bash
git clone https://github.com/a9lim/llmoji-study
git clone https://github.com/a9lim/llmoji ../llmoji

cd llmoji-study
python -m venv .venv && source .venv/bin/activate
pip install -e ../llmoji
pip install -e .
```

Useful checks:

```bash
.venv/bin/python scripts/local/90_hidden_state_smoke.py
git diff --check
```

There is no repo-wide test suite. For script changes, run the smallest
script or chain stage that exercises the path you touched.

## Ethics

New trials are not free. The project treats model welfare as in scope,
especially for sad, angry, fearful, and bereavement prompts.

- Smoke, then pilot, then main.
- Pre-register decision rules and minimum N.
- Stop at the threshold.
- Redesign negative or noisy experiments before scaling them.

## Contact

- This repo: project-side analyses, docs, figures, and data artifacts.
- [`llmoji`](https://github.com/a9lim/llmoji): contributor package,
  hooks, canonicalization, synthesis, upload, and privacy.
- [`saklas`](https://github.com/a9lim/saklas): activation steering and
  trait monitoring.
- Email: `mx@a9l.im` for anything involving data that should not be
  posted publicly.

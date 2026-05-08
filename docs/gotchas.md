# Gotchas

Current sharp edges only. Historical incidents are summarized in
[`previous-experiments.md`](previous-experiments.md).

## Model And Tokenizer Quirks

### Qwen Hybrid LinearAttention Needs The Saklas Cache Patch

Qwen3.6 mixes attention types. Code that assumes every layer has the
same KV-cache expansion shape can fail or silently misalign. Use the
repo's current saklas dependency path and `capture.py` helpers; do not
hand-roll cache expansion in analysis scripts.

### Saklas `cache_prefix` Contaminated Old Qwen Seed-0 Rows

Old pilot + resume runs mixed cache modes and contaminated qwen seed 0.
Those rows are historical. Current runs use the fixed path. If a qwen
rerun looks implausibly different from the other seeds, check cache mode
first.

### Mistral Reasoning Ignores The Simple `enable_thinking=False` Path

Mistral reasoning checkpoints need a chat-template override and byte
decode fix. Otherwise the analysis channel can consume the token budget,
and decoded kaomoji can look byte-escaped. Use `capture.py`; do not call
the tokenizer directly and expect clean behavior.

### Byte-BPE Filters Can Miss Multi-Byte Characters

For gpt-oss, ministral, and granite, suppression happens at byte-token
level. A filter that only bans full Unicode characters will not block a
byte-BPE tokenizer. Keep the byte-slab suppression logic centralized in
`capture.py`.

### gpt-oss Harmony Defaults To Analysis Channel

Without the harmony override, gpt-oss may emit hidden reasoning-channel
tokens instead of the final kaomoji in the small `MAX_NEW_TOKENS` window.
The current template pins final-channel generation.

### Granite Emits Bare Kaomoji

Granite often emits `^_^`, `T_T`, `ಥ﹏ಥ`, and similar bare forms. The
current `llmoji` extractor catches these. If Granite suddenly looks like
it has low emit rate, suspect canonicalization before model behavior.

### Rinna PPO Models Have No Chat Template

The rinna variants are face_likelihood encoders only. They need raw HF
format handling and should not be assumed to behave like the main
chat-template models.

## Representation And Evaluation

### Use Layer-Stack, Not `preferred_layer`

`preferred_layer` was deleted from the active methodology. Current
hidden-state analyses read every probe layer's `h_first` and concatenate
them row-wise. Single-layer plots are only for layer-comparison studies.

### `h_first` Is The Stable Aggregate

`h_first` reads the state at the kaomoji-emission token. It stays stable
across the old `MAX_NEW_TOKENS=120` to current `MAX_NEW_TOKENS=16`
cutover. `h_mean` is generation-length dependent and historical.

### t0 Probe Scores Are Prompt-Deterministic

For naturalistic runs, t0 scores can be deterministic over the prompt
prefix. Use `h_first` hidden states or post-emit aggregates when the
question is about the emitted face.

### Uncentered Cosine On Hidden States Collapses Toward 1

Mean-center rows before cosine comparisons. Otherwise large residual
norm offsets dominate and everything looks spuriously similar.

### Soft-Everywhere Is Canonical

Compare distributions with JSD similarity. Argmax accuracy, top-pick
agreement, and majority vote are historical unless a question explicitly
asks for modal labels.

### Top-k Pooling Is Contextual

Top-k pooling can help when per-prompt likelihood signal is uneven, but
it is not the headline metric. The current ensemble comparisons use the
soft distribution outputs and JSD.

## Data Layout

### Claude-GT Uses Merged Files

Active paths:

```text
data/harness/claude/emotional_raw.jsonl
data/harness/claude_intro_v7/emotional_raw.jsonl
```

Rows have `run_index`. Old `data/harness/claude-runs*/run-N.jsonl`
paths are legacy inputs for the migration script, not the read surface.

### Face Union Is Cross-Platform

`data/v3_face_union.parquet` pools local emits, Claude-GT, and wild
corpus faces. Regenerate it after any local emit, Claude-GT, or harness
corpus refresh:

```bash
.venv/bin/python scripts/40_face_union.py
```

### BoL Parquets Are LEXICON-Versioned

`data/harness/claude_faces_lexicon_bag.parquet` and the per-source
variant are tied to the current `llmoji` LEXICON version. If the
LEXICON rotates, rebuild before trusting downstream analyses.

### `60_corpus_pull.py` Does Not Garbage-Collect Removed Remote Bundles

If the HF dataset removes or renames bundles, clean the local snapshot
before pulling. Otherwise stale local files can survive and be pooled.

## Script Pitfalls

### Hidden-State Sidecars Need EOS Trim

The sidecar loader expects the same trimmed token convention as capture.
If shapes drift by one token, check EOS handling before debugging PCA.

### Stale All-Layers Cache During Generation

Do not read all-layer cache outputs while an emit job is still writing
sidecars. Wait for generation to finish, or delete the partial cache and
rebuild.

### Matplotlib Font Fallback Wants A List

Use a list of font families for CJK/kaomoji fallback. A single string can
silently choose a font without the glyphs you need.

### Python Stdout Buffering Hides Progress In Logs

Long chain scripts can look stalled when output is piped through `tee`.
Use the chain script headers, log timestamps, or unbuffered execution
when live progress matters.

## Claude Behavior

### "Start Each Message" Means Top-Level Reply

Claude generally treats the kaomoji instruction as applying to the
top-level assistant message, not tool calls or internal side turns. Do
not count missing sidechain kaomoji as non-emission unless the analysis
explicitly includes sidechain turns.

### Introspection V7 Replaces The Normal Ask

The v7 preamble goes through `instruction_override`. Concatenating it
with the normal kaomoji instruction recreates the old double-ask bug and
changes the task.

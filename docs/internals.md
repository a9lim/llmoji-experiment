# Internals

Two surfaces are load-bearing: hidden-state sidecars and kaomoji
canonicalization.

## Hidden-State Sidecars

After generation, `transformer_experiments.hidden_capture.read_after_generate()`
reads saklas buckets and writes one sidecar per row:

```text
data/local/hidden/<experiment>/<row_uuid>.npz
```

Each sidecar stores `h_first`, `h_last`, and `h_mean` per probe layer.
`h_first` is the active aggregate because it is the kaomoji-emission
state. Full per-token traces are opt-in for smoke/debug paths only.

JSONL rows keep metadata, emitted face, prompt id, probe scores, and the
sidecar UUID. Sidecars keep hidden states. Treat the pair as the source
of truth.

Current analysis loaders:

- `load_emotional_features_stack(model_key)`: registry-keyed layer-stack
  read.
- `load_emotional_features_stack_at(path)`: path-aware layer-stack read.
- `load_hidden_states(full_trace=False)`: lower-level sidecar loader.

The layer-stack matrix has shape:

```text
(n_rows, n_probe_layers * hidden_dim)
```

Do not use old `preferred_layer` logic for current results. Single-layer
helpers exist for diagnostics and layer-sweep plots only.

## Merged Claude-GT Rows

Claude-GT rows now live in merged files:

```text
data/harness/claude/emotional_raw.jsonl
data/harness/claude_intro_v7/emotional_raw.jsonl
```

Each row has `run_index`. Use `llmoji_experiment.claude_gt` helpers rather
than reading legacy `claude-runs*/run-N.jsonl` paths directly.

## Kaomoji Canonicalization

Canonicalization lives in the companion package:

```python
from llmoji.taxonomy import canonicalize_kaomoji
```

This repo applies it again on read, because contributor bundles may have
been produced under different package versions.

Current rule families:

1. NFC normalize, not NFKC.
2. Strip invisible format characters.
3. Fold whitelisted typographic near-equivalents.
4. Strip ASCII spaces inside bracket spans.
5. Lowercase Cyrillic capitals used as face glyphs.
6. Strip boundary arm modifiers where they are cosmetic.
7. Preserve eye, mouth, and decoration changes unless a rule explicitly
   says they are cosmetic.

Raw emitted form remains useful for audit; canonical form is what counts
for analysis.

## Quadrants

`llmoji_experiment/quadrants.py` owns:

- `QUADRANT_ORDER`: aggregate order.
- `QUADRANT_ORDER_SPLIT`: 9-cell split order.
- `QUADRANT_COLORS`: OKLCH palette.
- `SPLIT_MARKERS`: cells split by dominance.

Do not create local copies of these constants in scripts.

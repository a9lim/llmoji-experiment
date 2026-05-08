#!/usr/bin/env bash
# Per-model local analyses on the v4 emit data.
#
# Runs the LLMOJI_MODEL-env-driven scripts (10, 11, 20, 21, 23, 24).
# Does NOT run face_likelihood (50) — that has a face-union dependency
# and runs as stage 3 of run_local_chain.sh after stage 2 builds the
# union via scripts/40_face_union.py.
#
# Per-model only — does NOT run cross-model (22, 25, 26) or face-
# stability triple (27, 28, 29) which iterate registry internally.
# See run_local_chain.sh for the full chain.
#
# Usage:
#   scripts/run_per_model.sh <model> [<suffix> [<preamble>]]
# Examples:
#   scripts/run_per_model.sh gemma
#   scripts/run_per_model.sh qwen
#   scripts/run_per_model.sh gemma intro_v7_primed preambles/introspection_v7.txt

set -euo pipefail
cd "$(dirname "$0")/.."

model=${1:?model short-name required (gemma / qwen / ministral / gpt_oss_20b / granite / etc)}
suffix=${2:-}
preamble=${3:-}

# Build env-var prefix.
env_args=("LLMOJI_MODEL=$model")
[ -n "$suffix"   ] && env_args+=("LLMOJI_OUT_SUFFIX=$suffix")
[ -n "$preamble" ] && env_args+=("LLMOJI_PREAMBLE_FILE=$preamble")

label="$model${suffix:+_$suffix}"
echo ""
echo "================================================================"
echo "  per-model pipeline: $label"
echo "================================================================"

# Per-model scripts — single source of truth for the list. If you add
# a new per-model stage, add it here; both standalone invocations and
# run_local_chain.sh stage 1 will pick it up automatically.
PER_MODEL_SCRIPTS=(
    10_emit_analysis.py             # JSONL → summary TSV + Fig A/B/C (no hidden state)
    11_emit_probe_correlations.py   # spearman + trio JSON (no hidden state)
    20_v3_layerwise_emergence.py    # per-layer silhouette
    21_v3_same_face_cross_quadrant.py
    23_v3_pca3plus.py
    24_v3_kaomoji_predictiveness.py
)
for script in "${PER_MODEL_SCRIPTS[@]}"; do
    echo ""
    echo "  >>> scripts/local/$script"
    env "${env_args[@]}" .venv/bin/python "scripts/local/$script"
done

echo ""
echo "  done: $label"

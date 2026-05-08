#!/usr/bin/env bash
# Full local-side analysis chain after a v4 emit run.
#
# Order:
#   1. Per-model emit + hidden-state analyses across all 6 configs
#      (gemma, qwen, ministral, gpt_oss_20b, granite, gemma_intro_v7_primed)
#      via run_per_model.sh — but skips face_likelihood there.
#   2. Cross-model: face union (40), face overlap (41 — local-side only,
#      run again later via run_harness_chain.sh for the --include-claude
#      pull).
#   3. Per-model face_likelihood (50) once the union exists.
#   4. Cross-model hidden-state analyses (22 alignment, 25 D/S
#      classifier, 26 procrustes).
#   5. Face-stability triple (27, 28, 29 — iterate registry internally).
#   6. Face_likelihood ensemble + cross-emit sanity (52, 53, 54, 51).
#
# Skip the harness-side regen (Anthropic API calls) — see
# run_harness_chain.sh for that. Cross-platform 66/67 aggregations also
# live in run_harness_chain since they pool both sides.
#
# Usage:
#   scripts/run_local_chain.sh
#
# Configs are hard-coded; edit MODELS / V7PRIMED_CONFIG below if the
# canonical lineup changes.

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(gemma qwen ministral gpt_oss_20b granite)
V7PRIMED_MODEL=gemma
V7PRIMED_SUFFIX=intro_v7_primed
V7PRIMED_PREAMBLE=preambles/introspection_v7.txt

echo ""
echo "################################################################"
echo "#  v4 local-chain regen"
echo "################################################################"

# ---------------------------------------------------------------------
# Stage 1 — per-model emit + hidden-state analyses. Delegates to
# run_per_model.sh so the per-model script list stays single-source-
# of-truth. Face_likelihood (50) is NOT in run_per_model.sh — it has
# a face-union dependency and runs as stage 3 below.
# ---------------------------------------------------------------------
for m in "${MODELS[@]}"; do
    bash scripts/run_per_model.sh "$m"
done
bash scripts/run_per_model.sh \
    "$V7PRIMED_MODEL" "$V7PRIMED_SUFFIX" "$V7PRIMED_PREAMBLE"

# ---------------------------------------------------------------------
# Stage 2 — cross-platform face union. Pools v3 emit + Claude pilot +
# wild contributor faces. Output: data/v3_face_union.{parquet,tsv}.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 2: face union (cross-platform)"
echo "================================================================"
echo ""
echo "  >>> scripts/40_face_union.py"
.venv/bin/python scripts/40_face_union.py

# ---------------------------------------------------------------------
# Stage 3 — per-model face_likelihood. Reads data/v3_face_union.parquet
# (built in stage 2). Each invocation runs 120 prompts × ~573 faces and
# writes data/local/<model>{_<suffix>}/face_likelihood{,_summary}.{parquet,tsv}.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 3: face_likelihood per-model encoder"
echo "================================================================"
for m in "${MODELS[@]}"; do
    echo ""
    echo "  >>> scripts/local/50_face_likelihood.py --model $m"
    .venv/bin/python scripts/local/50_face_likelihood.py --model "$m"
done
echo ""
echo "  >>> scripts/local/50_face_likelihood.py --model $V7PRIMED_MODEL (suffix=$V7PRIMED_SUFFIX)"
LLMOJI_OUT_SUFFIX="$V7PRIMED_SUFFIX" \
LLMOJI_PREAMBLE_FILE="$V7PRIMED_PREAMBLE" \
    .venv/bin/python scripts/local/50_face_likelihood.py --model "$V7PRIMED_MODEL"

# ---------------------------------------------------------------------
# Stage 4 — cross-model hidden-state analyses. These iterate the
# registry / take --models flags directly.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 4: cross-model hidden-state analyses"
echo "================================================================"

models_csv=$(IFS=,; echo "${MODELS[*]}")

echo ""
echo "  >>> scripts/local/26_v3_quadrant_procrustes.py --models $models_csv --reference gemma"
.venv/bin/python scripts/local/26_v3_quadrant_procrustes.py \
    --models "$models_csv" --reference gemma

# ---------------------------------------------------------------------
# Stage 5 — face-stability triple. Each script iterates MODEL_REGISTRY
# internally; no per-model orchestration needed.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 5: face-stability triple"
echo "================================================================"
for script in 27_v3_face_stability.py 28_v3_state_predicts_face.py 29_v3_pc_point_clouds_3d.py; do
    echo ""
    echo "  >>> scripts/local/$script"
    .venv/bin/python "scripts/local/$script"
done

# ---------------------------------------------------------------------
# Stage 6 — face_likelihood ensemble. Auto-discovers per-encoder
# face_likelihood TSVs (local + harness if present); produces subset-
# search / topk-pooling / ensemble-predict tables under data/.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 6: face_likelihood ensemble + cross-emit sanity"
echo "================================================================"

echo ""
echo "  >>> scripts/52_subset_search.py --prefer-full --top-k 25"
.venv/bin/python scripts/52_subset_search.py --prefer-full --top-k 25

echo ""
echo "  >>> scripts/53_topk_pooling.py --prefer-full"
.venv/bin/python scripts/53_topk_pooling.py --prefer-full

echo ""
echo "  >>> scripts/54_ensemble_predict.py --models $models_csv"
.venv/bin/python scripts/54_ensemble_predict.py --models "$models_csv"

echo ""
echo "  >>> scripts/local/51_cross_emit_sanity.py --prefer-full"
.venv/bin/python scripts/local/51_cross_emit_sanity.py --prefer-full

# ---------------------------------------------------------------------
# Stage 7 — face overlap (local-side only, no Claude pull). For the
# --include-claude version see run_harness_chain.sh.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 7: face overlap (local-only, no Claude pull)"
echo "================================================================"
echo ""
echo "  >>> scripts/41_face_overlap.py"
.venv/bin/python scripts/41_face_overlap.py

echo ""
echo "################################################################"
echo "#  v4 local-chain regen complete"
echo "################################################################"

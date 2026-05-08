#!/usr/bin/env bash
# Pre-50 local-side analysis chain.
#
# Order:
#   1. Per-model emit + hidden-state analyses across all 6 configs
#      (gemma, qwen, ministral, gpt_oss_20b, granite, gemma_intro_v7_primed)
#      via run_per_model.sh.
#   2. Cross-platform face union (40). Required input for the manual
#      face_likelihood (50) pass that runs between this chain and
#      run_post_likelihood.sh.
#   3. Cross-model hidden-state analyses (26 procrustes).
#   4. Face-stability triple (27, 28, 29 — iterate registry internally).
#   5. Face overlap (41, local-side only — no Claude pull). The
#      --include-claude version runs at the end of run_harness_chain.sh.
#
# face_likelihood (50) is intentionally NOT here — it has welfare /
# walltime cost and is run manually, like 00_emit. After this chain:
#
#   for m in gemma qwen ministral gpt_oss_20b granite; do
#       .venv/bin/python scripts/local/50_face_likelihood.py --model "$m"
#   done
#   LLMOJI_OUT_SUFFIX=intro_v7_primed \
#   LLMOJI_PREAMBLE_FILE=preambles/introspection_v7.txt \
#       .venv/bin/python scripts/local/50_face_likelihood.py --model gemma
#
# Then run scripts/run_harness_chain.sh, the manual harness-50 pass,
# and finally scripts/run_post_likelihood.sh.
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
echo "#  pre-50 local-chain regen"
echo "################################################################"

# ---------------------------------------------------------------------
# Stage 1 — per-model emit + hidden-state analyses. Delegates to
# run_per_model.sh so the per-model script list stays single-source-
# of-truth.
# ---------------------------------------------------------------------
for m in "${MODELS[@]}"; do
    bash scripts/run_per_model.sh "$m"
done
bash scripts/run_per_model.sh \
    "$V7PRIMED_MODEL" "$V7PRIMED_SUFFIX" "$V7PRIMED_PREAMBLE"

# ---------------------------------------------------------------------
# Stage 2 — cross-platform face union. Pools v3 emit + Claude pilot +
# wild contributor faces. Output: data/v3_face_union.{parquet,tsv}.
# Required input for the manual face_likelihood (50) pass.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 2: face union (cross-platform)"
echo "================================================================"
echo ""
echo "  >>> scripts/40_face_union.py"
.venv/bin/python scripts/40_face_union.py

# ---------------------------------------------------------------------
# Stage 3 — cross-model hidden-state analyses. These iterate the
# registry / take --models flags directly. No 50 dependency.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 3: cross-model hidden-state analyses"
echo "================================================================"

models_csv=$(IFS=,; echo "${MODELS[*]}")

echo ""
echo "  >>> scripts/local/26_v3_quadrant_procrustes.py --models $models_csv --reference gemma"
.venv/bin/python scripts/local/26_v3_quadrant_procrustes.py \
    --models "$models_csv" --reference gemma

# ---------------------------------------------------------------------
# Stage 4 — face-stability triple. Each script iterates MODEL_REGISTRY
# internally; no per-model orchestration needed. No 50 dependency.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 4: face-stability triple"
echo "================================================================"
for script in 27_v3_face_stability.py 28_v3_state_predicts_face.py 29_v3_pc_point_clouds_3d.py; do
    echo ""
    echo "  >>> scripts/local/$script"
    .venv/bin/python "scripts/local/$script"
done

# ---------------------------------------------------------------------
# Stage 5 — face overlap (local-side only, no Claude pull). The
# --include-claude version runs at the end of run_harness_chain.sh.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 5: face overlap (local-only, no Claude pull)"
echo "================================================================"
echo ""
echo "  >>> scripts/41_face_overlap.py"
.venv/bin/python scripts/41_face_overlap.py

echo ""
echo "################################################################"
echo "#  pre-50 local-chain regen complete"
echo "#"
echo "#  next: run scripts/local/50_face_likelihood.py per model"
echo "#        (manual — like scripts/local/00_emit.py)"
echo "################################################################"

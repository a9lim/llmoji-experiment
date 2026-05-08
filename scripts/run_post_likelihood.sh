#!/usr/bin/env bash
# Post-50 cross-platform aggregations.
#
# Run this AFTER both face_likelihood passes have been done manually:
#
#   # Local 50 (per encoder model):
#   for m in gemma qwen ministral gpt_oss_20b granite; do
#       .venv/bin/python scripts/local/50_face_likelihood.py --model "$m"
#   done
#   LLMOJI_OUT_SUFFIX=intro_v7_primed \
#   LLMOJI_PREAMBLE_FILE=preambles/introspection_v7.txt \
#       .venv/bin/python scripts/local/50_face_likelihood.py --model gemma
#
#   # Harness 50 (Anthropic-API judges):
#   ANTHROPIC_API_KEY=… .venv/bin/python scripts/harness/50_face_likelihood.py
#   ANTHROPIC_API_KEY=… .venv/bin/python scripts/harness/50_face_likelihood.py \
#       --model opus --gt-only
#
# Order:
#   1. Face_likelihood ensemble (52, 53, 54) + cross-emit sanity (51).
#      Auto-discovers per-encoder face_likelihood TSVs (local + harness +
#      bol).
#   2. Three-way (use / read / act) per-face joiner (68).
#   3. Per-source drift case files (69).
#   4. Cross-platform aggregations: per-project quadrants (66), wild
#      residuals (67).
#
# No API key required here — every input is a TSV/parquet on disk.
#
# Usage:
#   scripts/run_post_likelihood.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=(gemma qwen ministral gpt_oss_20b granite)
models_csv=$(IFS=,; echo "${MODELS[*]}")

echo ""
echo "################################################################"
echo "#  post-50 cross-platform aggregations"
echo "################################################################"

# ---------------------------------------------------------------------
# Stage 1 — face_likelihood ensemble. Auto-discovers per-encoder
# face_likelihood TSVs (local + harness + bol); produces subset-search
# / topk-pooling / ensemble-predict tables under data/.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 1: face_likelihood ensemble + cross-emit sanity"
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
# Stage 2 — three-way (use / read / act) per-face analysis. Inner-join
# of GT (Claude emit) + opus introspection + haiku introspection + BoL.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 2: three-way per-face analysis"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/68_three_way_analysis.py"
.venv/bin/python scripts/harness/68_three_way_analysis.py

# ---------------------------------------------------------------------
# Stage 3 — per-source drift case files.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 3: per-source drift"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/69_per_source_drift.py"
.venv/bin/python scripts/harness/69_per_source_drift.py

# ---------------------------------------------------------------------
# Stage 4 — cross-platform aggregations: per-project quadrants (66) +
# wild residual clusters (67). 66's per-machine outputs are gitignored;
# 67 produces wild_residual_clusters{,_gt_only}.tsv (committed).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 4: cross-platform aggregations"
echo "================================================================"

echo ""
echo "  >>> scripts/66_per_project_quadrants.py --mode gt-priority"
.venv/bin/python scripts/66_per_project_quadrants.py --mode gt-priority

echo ""
echo "  >>> scripts/67_wild_residual.py --fixed-k 9"
.venv/bin/python scripts/67_wild_residual.py --fixed-k 9

echo ""
echo "################################################################"
echo "#  post-50 aggregations complete"
echo "################################################################"

#!/usr/bin/env bash
# Harness-side regen + cross-platform aggregations.
#
# Order:
#   1. BoL parquet rebuild (62) + per-source long-format (64) — no API.
#   2. Claude API face_likelihood encoders: haiku (default) + opus
#      (--gt-only). Re-judges canonical faces from data/v3_face_union.
#      Skips already-judged faces if the JSONL exists (resume via the
#      script's own done-set).
#   3. BoL face_likelihood encoder (55) — no API.
#   4. Three-way per-face joiner (68) — needs GT + opus + haiku + bol
#      face_likelihood TSVs.
#   5. Per-source drift case files (69).
#   6. Cross-platform aggregations: per-project quadrants (66), wild
#      residuals (67).
#   7. Face overlap with Claude pull (41 --include-claude).
#
# Requires ANTHROPIC_API_KEY for stage 2. Set in the environment before
# invoking, or comment out stage 2 if you only want to regen from
# already-judged JSONLs.
#
# Usage:
#   ANTHROPIC_API_KEY=… scripts/run_harness_chain.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "WARNING: ANTHROPIC_API_KEY not set — stage 2 (haiku/opus face_likelihood)"
    echo "         will fail. Either export it or comment stage 2 out below."
    echo ""
fi

echo ""
echo "################################################################"
echo "#  harness-chain regen"
echo "################################################################"

# ---------------------------------------------------------------------
# Stage 1 — BoL parquets. Pooled (62) + per-source long-format (64).
# Reads data/harness/hf_dataset/ (snapshot of a9lim/llmoji); builds
# claude_faces_lexicon_bag{,_per_source}.parquet. No GPU / no API.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 1: BoL parquets (pooled + per-source)"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/62_corpus_lexicon.py"
.venv/bin/python scripts/harness/62_corpus_lexicon.py
echo ""
echo "  >>> scripts/harness/64_corpus_lexicon_per_source.py"
.venv/bin/python scripts/harness/64_corpus_lexicon_per_source.py

# ---------------------------------------------------------------------
# Stage 2 — Anthropic-judge face_likelihood encoders. The script
# auto-resumes via its own done-set on the JSONL, so prior welfare-
# spent judgments are preserved; only NEW canonical faces (added by
# the v4 face union) get judged.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 2: Claude-API face_likelihood encoders (haiku + opus)"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/50_face_likelihood.py            # haiku, default"
.venv/bin/python scripts/harness/50_face_likelihood.py
echo ""
echo "  >>> scripts/harness/50_face_likelihood.py --model opus --gt-only"
.venv/bin/python scripts/harness/50_face_likelihood.py --model opus --gt-only

# ---------------------------------------------------------------------
# Stage 3 — BoL face_likelihood encoder. Reads claude_faces_lexicon_bag
# parquet, projects each face's bag onto the v3 quadrant axes via the
# llmoji v2 LEXICON. Auto-discovered by 52/53/54 ensemble.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 3: BoL face_likelihood encoder"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/55_bol_encoder.py"
.venv/bin/python scripts/harness/55_bol_encoder.py

# ---------------------------------------------------------------------
# Stage 4 — Three-way (use / read / act) per-face analysis. Inner-join
# of GT (Claude emit) + opus introspection + haiku introspection + BoL.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 4: three-way per-face analysis"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/68_three_way_analysis.py"
.venv/bin/python scripts/harness/68_three_way_analysis.py

# ---------------------------------------------------------------------
# Stage 5 — Per-source drift case files.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 5: per-source drift"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/69_per_source_drift.py"
.venv/bin/python scripts/harness/69_per_source_drift.py

# ---------------------------------------------------------------------
# Stage 6 — Cross-platform aggregations: per-project quadrants (66) +
# wild residual clusters (67). 66's per-machine outputs are gitignored;
# 67 produces wild_residual_clusters{,_gt_only}.tsv (committed).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 6: cross-platform aggregations"
echo "================================================================"

echo ""
echo "  >>> scripts/66_per_project_quadrants.py --mode gt-priority"
.venv/bin/python scripts/66_per_project_quadrants.py --mode gt-priority

echo ""
echo "  >>> scripts/67_wild_residual.py --fixed-k 6"
.venv/bin/python scripts/67_wild_residual.py --fixed-k 6

echo ""
echo "  >>> scripts/67_wild_residual.py --gt-only --fixed-k 6"
.venv/bin/python scripts/67_wild_residual.py --gt-only --fixed-k 6

# ---------------------------------------------------------------------
# Stage 7 — Face overlap with Claude pull (now that haiku/opus jsonls
# are refreshed).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 7: face overlap with Claude pull"
echo "================================================================"
echo ""
echo "  >>> scripts/41_face_overlap.py --include-claude"
.venv/bin/python scripts/41_face_overlap.py --include-claude

echo ""
echo "################################################################"
echo "#  harness-chain regen complete"
echo "################################################################"

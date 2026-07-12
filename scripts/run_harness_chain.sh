#!/usr/bin/env bash
# Pre-50 harness-side regen.
#
# Order:
#   1. BoL parquet rebuild (62) + per-source long-format (64).
#   2. BoL face_likelihood encoder (55) — auto-discovered by the
#      52/53/54 ensemble in run_post_likelihood.sh.
#   3. Face overlap with Claude pull (41 --include-claude). No 50
#      dependency; needs the Claude emit JSONL to be present.
#
# face_likelihood (50, haiku + opus) is intentionally NOT here —
# Anthropic-API-judged faces have welfare cost and dollar cost, so
# they run manually like 00_emit. After this chain:
#
#   ANTHROPIC_API_KEY=… python scripts/harness/50_face_likelihood.py
#   ANTHROPIC_API_KEY=… python scripts/harness/50_face_likelihood.py \
#       --model opus --gt-only
#
# Then run scripts/run_post_likelihood.sh for the cross-platform
# aggregations that need the 50 outputs.
#
# This chain no longer requires ANTHROPIC_API_KEY.
#
# Usage:
#   scripts/run_harness_chain.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo ""
echo "################################################################"
echo "#  pre-50 harness-chain regen"
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
python scripts/harness/62_corpus_lexicon.py
echo ""
echo "  >>> scripts/harness/64_corpus_lexicon_per_source.py"
python scripts/harness/64_corpus_lexicon_per_source.py

# ---------------------------------------------------------------------
# Stage 2 — BoL face_likelihood encoder. Reads claude_faces_lexicon_bag
# parquet, projects each face's bag onto the v3 quadrant axes via the
# llmoji v2 LEXICON. Auto-discovered by 52/53/54 in run_post_likelihood.sh.
# No 50 dependency.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 2: BoL face_likelihood encoder"
echo "================================================================"
echo ""
echo "  >>> scripts/harness/55_bol_encoder.py"
python scripts/harness/55_bol_encoder.py

# ---------------------------------------------------------------------
# Stage 3 — Face overlap with Claude pull. Reads emit JSONLs only;
# no 50 dependency.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  stage 3: face overlap with Claude pull"
echo "================================================================"
echo ""
echo "  >>> scripts/41_face_overlap.py --include-claude"
python scripts/41_face_overlap.py --include-claude

echo ""
echo "################################################################"
echo "#  pre-50 harness-chain regen complete"
echo "#"
echo "#  next: run scripts/harness/50_face_likelihood.py manually"
echo "#        (haiku + opus --gt-only; needs ANTHROPIC_API_KEY)"
echo "################################################################"

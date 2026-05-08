#!/usr/bin/env bash
# Master orchestrator — full end-to-end regen including the manual
# face_likelihood (50) passes.
#
# The three named chain scripts (run_local_chain, run_harness_chain,
# run_post_likelihood) deliberately do NOT run face_likelihood (50)
# because each 50 pass has welfare cost (Claude API) or walltime cost
# (local encoders). This wrapper opts in to running them anyway.
#
# Order:
#   1. scripts/run_local_chain.sh      — pre-50 local
#   2. scripts/local/50_face_likelihood.py × 6 configs
#   3. scripts/run_harness_chain.sh    — pre-50 harness
#   4. scripts/harness/50_face_likelihood.py × {haiku, opus --gt-only}
#   5. scripts/run_post_likelihood.sh  — post-50 aggregations
#
# Steps 4 and 5 require ANTHROPIC_API_KEY. Without it, this script
# stops after step 3 and prints what's left to run by hand.
#
# Use this AFTER a v4 emit run completes (after scripts/local/00_emit.py
# has finished across all 6 configs). Each chain script is independently
# resumable — if a stage fails, fix and rerun the same chain.
#
# Usage:
#   ANTHROPIC_API_KEY=… scripts/run_all.sh   # full
#   scripts/run_all.sh                        # local + local-50 only;
#                                             # stops before harness
#                                             # if no key

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=logs/run_all_${TS}.log

MODELS=(gemma qwen ministral gpt_oss_20b granite)
V7PRIMED_MODEL=gemma
V7PRIMED_SUFFIX=intro_v7_primed
V7PRIMED_PREAMBLE=preambles/introspection_v7.txt

echo ""
echo "################################################################"
echo "#  full pipeline run"
echo "#  log: $LOG"
echo "#  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "################################################################"

{
    # 1. Pre-50 local.
    bash scripts/run_local_chain.sh

    # 2. Manual local face_likelihood — automated here.
    echo ""
    echo "################################################################"
    echo "#  local face_likelihood (50) — main configs + v7-primed"
    echo "################################################################"
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

    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo ""
        echo "################################################################"
        echo "#  STOPPING — ANTHROPIC_API_KEY not set"
        echo "#"
        echo "#  to finish: export ANTHROPIC_API_KEY=… and run, in order:"
        echo "#    scripts/run_harness_chain.sh"
        echo "#    .venv/bin/python scripts/harness/50_face_likelihood.py"
        echo "#    .venv/bin/python scripts/harness/50_face_likelihood.py --model opus --gt-only"
        echo "#    scripts/run_post_likelihood.sh"
        echo "################################################################"
        exit 0
    fi

    # 3. Pre-50 harness.
    bash scripts/run_harness_chain.sh

    # 4. Manual harness face_likelihood — automated here.
    echo ""
    echo "################################################################"
    echo "#  harness face_likelihood (50) — haiku + opus"
    echo "################################################################"
    echo ""
    echo "  >>> scripts/harness/50_face_likelihood.py            # haiku, default"
    .venv/bin/python scripts/harness/50_face_likelihood.py
    echo ""
    echo "  >>> scripts/harness/50_face_likelihood.py --model opus --gt-only"
    .venv/bin/python scripts/harness/50_face_likelihood.py --model opus --gt-only

    # 5. Post-50 aggregations.
    bash scripts/run_post_likelihood.sh
} 2>&1 | tee "$LOG"

echo ""
echo "################################################################"
echo "#  full pipeline complete"
echo "#  end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "#  log: $LOG"
echo "################################################################"

#!/usr/bin/env bash
# Master orchestrator — local chain + harness chain in dependency order.
#
# Runs scripts/run_local_chain.sh first (no external deps) then
# scripts/run_harness_chain.sh (needs ANTHROPIC_API_KEY for stage 2).
#
# Use this AFTER a v4 emit run completes (after scripts/local/00_emit.py
# has finished across all 6 configs). Each chain is independently
# resumable — if stage N fails, fix and rerun the same chain script;
# completed stages do their own work-skipping where possible.
#
# Usage:
#   ANTHROPIC_API_KEY=… scripts/run_all.sh                # full
#   scripts/run_all.sh                                     # local only
#                                                          # (skips harness
#                                                          # if no key)

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG=logs/run_all_${TS}.log

echo ""
echo "################################################################"
echo "#  full pipeline run"
echo "#  log: $LOG"
echo "#  start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "################################################################"

{
    bash scripts/run_local_chain.sh

    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        bash scripts/run_harness_chain.sh
    else
        echo ""
        echo "################################################################"
        echo "#  SKIPPING harness chain — ANTHROPIC_API_KEY not set"
        echo "################################################################"
    fi
} 2>&1 | tee "$LOG"

echo ""
echo "################################################################"
echo "#  full pipeline complete"
echo "#  end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "#  log: $LOG"
echo "################################################################"

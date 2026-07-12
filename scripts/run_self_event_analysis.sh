#!/usr/bin/env bash
# Post-emit analysis chain for the self-event prompt set.
#
# Run this after a fresh self-event emit + 22c centroid registration
# (22c with --namespace llmoji_self_event). Refreshes the
# self.other meta-axis using the new centroids, then re-runs the
# 22-series scatter / probe / comparison scripts on the self-event
# frame data so figures/local/<short>_self_event/ stays in sync.
#
# OA-1 is included in the axes-overlay variants of 22b so the
# off-axis bliss-attractor cell shows up alongside the canonical
# Russell axes (hp.ln / hp.lp / hnd.hns).
#
# No model load required — every script in this chain reads cached
# hidden-state sidecars and pre-registered centroid profiles. Safe
# to run with another LM session (saklas / gemma / etc.) live in
# parallel; total wall-clock is under a minute on M5 Max.
#
# Usage:
#   scripts/run_self_event_analysis.sh [<model>]
#
# Default model is $LLMOJI_MODEL or gemma. Suffix is fixed at
# self_event (matching the LLMOJI_OUT_SUFFIX used by the emit).
#
# Companion to scripts/local/22h_combined_pca_scatter.py — that
# produces the joint mirror-vs-self-event view; this script
# refreshes the per-frame views.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-${LLMOJI_MODEL:-gemma}}
SUFFIX=self_event
PY=python

# Probe sets passed to 22b's --probes flag. AXES_OA includes oa.nb so
# the bliss-attractor displacement appears alongside the 3 Russell
# axes; AXES is the canonical 3-axis set kept for back-compat.
AXES="hp.ln,hp.lp,hnd.hns"
AXES_OA="hp.ln,hp.lp,hnd.hns,oa.nb"

run() {
    local label="$1"; shift
    echo ""
    echo "  >>> $label"
    "$@"
}

echo ""
echo "================================================================"
echo "  self-event analysis chain: $MODEL (suffix=$SUFFIX)"
echo "================================================================"

# ---------------------------------------------------------------------
# 1. Refresh self.other meta-axis. Reads the 9-cell self-event
# centroids (registered separately by 22c with --namespace
# llmoji_self_event) and writes ~/.saklas/vectors/llmoji/self.other/.
# Must run before any 22b invocation that overlays self.other.
# ---------------------------------------------------------------------
run "22g_self_other_axis (refresh self.other from v4 self-event centroids)" \
    env LLMOJI_MODEL=$MODEL $PY scripts/local/22g_self_other_axis.py

# ---------------------------------------------------------------------
# h_first only across the 22-series and 29: per the 2026-05-09 self-
# event pilot doc, "Affect structure is parse-time only" — h_last
# collapses across aggregates / frames / prompt versions (cum PC1+PC2
# ≈ 14% regardless), so the h_last figures were mostly noise. Dropped
# 2026-05-09 along with the OA-1 inclusion pass.
# ---------------------------------------------------------------------

# 2. Per-frame data PCA scatter for self-event (now OA-1-aware).
run "22d_data_pca_scatter [self-event h_first, with OA-1]" \
    env LLMOJI_MODEL=$MODEL LLMOJI_OUT_SUFFIX=$SUFFIX \
        $PY scripts/local/22d_data_pca_scatter.py --which h_first

# 3. Saklas-probes-in-PCA overlays for the self-event frame.
# Two variants:
#   - default namespace (bundled saklas affect probes) overlaid on
#     self-event PCA basis
#   - llmoji_self_event namespace, axes set including oa.nb, so the
#     OA-1 displacement appears as a fourth direction alongside the
#     Russell axes
# Both now render OA-1 scatter dots in black via QUADRANT_COLORS.
# --no-bootstrap on every invocation: profiles must already be cached.
run "22b_saklas_probes_in_pca [self-event h_first, bundled affect]" \
    env LLMOJI_MODEL=$MODEL LLMOJI_OUT_SUFFIX=$SUFFIX \
        $PY scripts/local/22b_saklas_probes_in_pca.py \
        --which h_first --no-bootstrap

run "22b_saklas_probes_in_pca [self-event h_first, centroid axes + OA]" \
    env LLMOJI_MODEL=$MODEL LLMOJI_OUT_SUFFIX=$SUFFIX \
        $PY scripts/local/22b_saklas_probes_in_pca.py \
        --which h_first --no-bootstrap \
        --namespace llmoji_self_event \
        --probes "$AXES_OA" \
        --out-tag axes

# 4. Combined mirror-vs-self-event view (22h).
run "22h_combined_pca_scatter [h_first, with OA-1]" \
    env LLMOJI_MODEL=$MODEL \
        $PY scripts/local/22h_combined_pca_scatter.py --which h_first

# 4b. Native PC1/PC2/PC3 point cloud (script 29) — 3D HTML showing
# the cluster geometry without any probe arrows. With OA-1 added
# 2026-05-09. Targeted at the active model only via --model so the
# orchestrator stays self-event-scoped.
run "29_v3_pc_point_clouds_3d [self-event, with OA-1]" \
    env LLMOJI_MODEL=$MODEL LLMOJI_OUT_SUFFIX=$SUFFIX \
        $PY scripts/local/29_v3_pc_point_clouds_3d.py --model $MODEL

# ---------------------------------------------------------------------
# 4c. Mirror-frame variants of 22d / 22b / 29 with OA-1 pooled into
# the PCA fit (--pool-lb-from $SUFFIX). The OA-1 rows live only in
# the self-event capture; this pass adds them to the mirror data so
# the mirror's 1440-row PCA basis covers the OA-1 region too, and
# the resulting figures show OA-1 black dots alongside the canonical
# 9-cell rainbow. Layer intersection handles the mirror (56 layers)
# vs self-event (58 layers) capture depth difference automatically.
# Output filenames overlap with the canonical mirror-frame outputs;
# this is intentional — the OA-1-pooled view replaces the OA-less
# view as the canonical mirror-side rendering.
# ---------------------------------------------------------------------
run "22d_data_pca_scatter [mirror h_first, OA-1 pooled]" \
    env LLMOJI_MODEL=$MODEL \
        $PY scripts/local/22d_data_pca_scatter.py \
        --which h_first --pool-lb-from $SUFFIX

run "22b_saklas_probes_in_pca [mirror h_first, OA-1 pooled, bundled affect]" \
    env LLMOJI_MODEL=$MODEL \
        $PY scripts/local/22b_saklas_probes_in_pca.py \
        --which h_first --no-bootstrap --pool-lb-from $SUFFIX

# oa.nb only exists in the llmoji_self_event namespace (it was
# registered alongside q_oa from the self-event capture; mirror has no
# OA rows so couldn't have produced it). We pull all four axes from
# llmoji_self_event for this overlay so the namespace stays consistent
# in the figure caption. The hp.ln / hp.lp / hnd.hns directions there
# come from the smaller self-event dataset but point along the same
# V/A/D substructure as their mirror counterparts.
run "22b_saklas_probes_in_pca [mirror h_first, OA-1 pooled, centroid axes + OA]" \
    env LLMOJI_MODEL=$MODEL \
        $PY scripts/local/22b_saklas_probes_in_pca.py \
        --which h_first --no-bootstrap \
        --namespace llmoji_self_event \
        --probes "$AXES_OA" \
        --out-tag axes \
        --pool-lb-from $SUFFIX

run "29_v3_pc_point_clouds_3d [mirror, OA-1 pooled]" \
    env LLMOJI_MODEL=$MODEL \
        $PY scripts/local/29_v3_pc_point_clouds_3d.py \
        --model $MODEL --pool-lb-from $SUFFIX

# ---------------------------------------------------------------------
# 5. Mirror-vs-self-event comparison metrics.
# ---------------------------------------------------------------------
run "22e_mirror_vs_self_event (per-cell cosine heatmap)" \
    env LLMOJI_MODEL=$MODEL $PY scripts/local/22e_mirror_vs_self_event.py

run "22f_negative_affect_decomp (anger-vs-rest decomposition)" \
    env LLMOJI_MODEL=$MODEL $PY scripts/local/22f_negative_affect_decomp.py

echo ""
echo "================================================================"
echo "  done: self-event analysis chain ($MODEL)"
echo "================================================================"
echo ""
echo "Outputs:"
echo "  figures/local/${MODEL}/             — mirror frame + combined views"
echo "  figures/local/${MODEL}_${SUFFIX}/   — self-event frame (with OA-1)"
echo "  ~/.saklas/vectors/llmoji/self.other/ — refreshed meta-axis"

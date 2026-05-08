"""Canonical Russell-quadrant taxonomy + palette for the v4 9-cell registry.

Single source of truth for quadrant ordering, dominance-split labels,
and the OKLCH-uniform color mapping. Imported by:

- ``llmoji_study.emotional_analysis`` (figure helpers, per_face_*)
- ``llmoji_study.jsd`` (distribution-vs-distribution math)
- ``llmoji_study.lexicon`` (BoL → quadrant softmax)
- ``scripts/40_face_union.py`` / ``41_face_overlap.py`` / ``50_face_likelihood.py``
  / ``harness/10_emit_analysis.py`` etc. — all the analysis scripts
  that iterate per-cell or compute per-cell metrics.

This module is intentionally import-light — only ``__future__`` — so
it can be pulled into the JSD math layer (which is also import-light)
without dragging pandas / numpy / matplotlib transitively. Update the
constants here when the registry shape changes; downstream consumers
pick it up automatically.

v4 schema (2026-05-06)
----------------------
The 7 aggregate cells follow the 2-letter V/A code emitted by
``EmotionalPrompt.quadrant``. Letters: H = high arousal (a=+1),
N = neutral arousal (a=0), L = low arousal (a=-1) on the first
position; P = positive valence (v=+1), B = baseline valence (v=0),
N = negative valence (v=-1) on the second.

  HP  high-arousal positive    (excitement, joy, mischief)
  LP  low-arousal positive     (calm, contentment)
  NP  neutral-arousal positive (relief, gratitude)
  HN  high-arousal negative    (anger, fear, anxiety)
  LN  low-arousal negative     (sadness, weariness)
  NB  neutral baseline         (affectless on substantive content)
  HB  high-arousal baseline    (uncertain, confused)

Two cells (NN at (a=0, v=-1), LB at (a=-1, v=0)) are coordinate-real
but vocabulary-deferred — see ``docs/2026-05-06-nn-lb-future-cells.md``.

The 9 split cells inherit HP and HN dominance bisection via
``EmotionalPrompt.pad_dominance``: +1 = D (in-action / dominant
register), -1 = S (received-outcome / submissive register).

  HP-D  playful mischief           (energetic, dominant, in-action)
  HP-S  celebration                (energetic, received-outcome)
  HN-D  anger / contempt           (energetic, dominant, in-action)
  HN-S  fear / anxiety             (energetic, received-threat)

LP / NP / LN / NB / HB are dominance-aggregate (no S/D split — their
hidden-state separability under ``pad_dominance`` was not established
or was negative; see the powercheck in ``docs/findings.md``).
"""
from __future__ import annotations


# v4 7-cell aggregate ordering. Walks valence: HP (high-pos),
# LP (low-pos), NP (neutral-pos), HN (high-neg), LN (low-neg),
# NB (neutral-baseline), HB (high-baseline).
QUADRANT_ORDER: tuple[str, ...] = (
    "HP", "LP", "NP", "HN", "LN", "NB", "HB",
)


# v4 9-cell split ordering — the dominance-bisected view used by
# figures, classifiers, and JSD evaluation. Same shape walk as
# QUADRANT_ORDER but with HP and HN expanded.
QUADRANT_ORDER_SPLIT: tuple[str, ...] = (
    "HP-D", "HP-S", "LP", "NP", "HN-D", "HN-S", "LN", "NB", "HB",
)


# Membership test: is this label a dominance-split tag? Used by
# ``_palette_for`` to decide whether to render in 7-cell or 9-cell
# mode based on what's actually in the dataframe.
SPLIT_MARKERS: frozenset[str] = frozenset({"HP-D", "HP-S", "HN-D", "HN-S"})


# Canonical Russell-circumplex mapping, sourced from the a9lim.github.io
# website palette (`shared-tokens.js::_PALETTE.extended`). All chromatic
# entries are OKLCH L=0.62, C=0.17 uniform (gamut-capped to sRGB ceiling
# where needed) — so 50/50 RGB-linear mixes between any two quadrants
# are perceived-luminance-balanced and stay readable, and the chart
# blend at any per-face share matches what the website renders for the
# same semantic category. Adjacent mixes still read sensibly
# (HN+LN → desaturated violet, HP+LP → muted olive); diagonal
# "contradictory" mixes (HN+LP "anger meets calm", HP+LN "excited meets
# sad") fall to muted brown / cool grey — informatively unusual.
QUADRANT_COLORS: dict[str, str] = {
    # Aggregate cells (5 from v3 + 2 from v4 expansion).
    "HP": "#998700",  # yellow  — high arousal, positive (excitement/joy)
    "LP": "#009F68",  # green   — low arousal, positive (calm/contentment)
    "HN": "#DA534F",  # red     — high arousal, negative (anger/anxiety)
    "LN": "#0091C9",  # blue    — low arousal, negative (sadness/depression)
    "NB": "#808696",  # slate   — neutral baseline (intentionally muted, H=270)
    # v4 dominance-unsplit cells (2026-05-06). Hue placement mirrors
    # V/A placement on the grid: NP-lime (140°) sits between LP-green
    # (160°) and HP-yellow (100°) just as NP sits between LP and HP on
    # the arousal axis; HB-orange (55°) sits between HN-red (25°) and
    # HP-yellow (100°) just as HB sits between HN and HP on valence.
    "NP": "#449D2E",  # lime    — neutral-arousal positive (relief/gratitude)
    "HB": "#CA6800",  # orange  — high-arousal baseline-valence (uncertain)
    # Dominance-axis splits. Aggregate parent inherits the D color (so
    # backward-compat with non-split views holds); S sibling gets a new
    # hue at the same OKLCH L=0.62 C=0.17. HN split (2026-05-02): HN-S
    # = website's `purple` (negative-but-submissive, doesn't collide
    # with LN-blue). HP split (2026-05-06): HP-S = HP-yellow (existing
    # celebration vocabulary), HP-D = website's `magenta` (energetic-
    # but-not-celebratory, the playful-mischief register).
    "HN-D": "#DA534F",  # red     — anger / contempt (in-action, dominant)
    "HN-S": "#9769DC",  # purple  — fear / anxiety (received-threat, submissive)
    "HP-D": "#BD5AB6",  # magenta — playful mischief (in-action, dominant)
    "HP-S": "#998700",  # yellow  — celebration / received-outcome (submissive)
}


__all__ = [
    "QUADRANT_ORDER",
    "QUADRANT_ORDER_SPLIT",
    "QUADRANT_COLORS",
    "SPLIT_MARKERS",
]

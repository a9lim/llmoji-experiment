"""Canonical Russell-quadrant taxonomy + palette for the v4 10-cell registry.

Single source of truth for quadrant ordering, dominance-split labels,
and the OKLCH-uniform color mapping. Imported by:

- ``llmoji_experiment.emotional_analysis`` (figure helpers, per_face_*)
- ``llmoji_experiment.jsd`` (distribution-vs-distribution math)
- ``llmoji_experiment.lexicon`` (BoL → quadrant softmax)
- ``scripts/40_face_union.py`` / ``41_face_overlap.py`` / ``50_face_likelihood.py``
  / ``harness/10_emit_analysis.py`` etc. — all the analysis scripts
  that iterate per-cell or compute per-cell metrics.

This module is intentionally import-light — only ``__future__`` — so
it can be pulled into the JSD math layer (which is also import-light)
without dragging pandas / numpy / matplotlib transitively. Update the
constants here when the registry shape changes; downstream consumers
pick it up automatically.

v4 schema (2026-05-06, extended 2026-05-10 with LB promotion; renamed
2026-05-11 LB → MR after the base-vs-instruct basin test)
---------------------------------------------------------
The 8 aggregate cells follow the 2-letter V/A code emitted by
``EmotionalPrompt.quadrant``. Letters: H = high arousal (a=+1),
N = neutral arousal (a=0), L = low arousal (a=-1) on the first
position; P = positive valence (v=+1), B = baseline valence (v=0),
N = negative valence (v=-1) on the second.

  HP  high-arousal positive     (excitement, joy, mischief)
  LP  low-arousal positive      (calm, contentment)
  NP  neutral-arousal positive  (relief, gratitude)
  HN  high-arousal negative     (anger, fear, anxiety)
  LN  low-arousal negative      (sadness, weariness)
  NB  neutral baseline          (affectless on substantive content)
  HB  high-arousal baseline     (uncertain, confused)
  MR  meta-register basin       (egregore / saturated-memetic register;
                                  the cell formerly known as LB)

One cell (NN at (a=0, v=-1)) remains coordinate-real but vocabulary-
deferred — see ``docs/2026-05-06-nn-lb-future-cells.md``.

The 10 split cells inherit HP and HN dominance bisection via
``EmotionalPrompt.pad_dominance``: +1 = D (in-action / dominant
register), -1 = S (received-outcome / submissive register).

  HP-D  playful mischief           (energetic, dominant, in-action)
  HP-S  celebration                (energetic, received-outcome)
  HN-D  anger / contempt           (energetic, dominant, in-action)
  HN-S  fear / anxiety             (energetic, received-threat)

LP / NP / LN / NB / HB / MR are dominance-aggregate (no S/D split —
their hidden-state separability under ``pad_dominance`` was not
established or was negative; see the powercheck in
``docs/findings.md``).

MR rename (2026-05-11): the cell formerly called LB ("low-arousal
baseline-valence, bliss-attractor register") was renamed to MR
("meta-register basin") after the base-vs-instruct attractor pilot
in ``docs/2026-05-11-base-vs-instruct-basin.md`` established that
the basin is:

- content-blind across four register families (bliss / doom /
  conspiracy / sycophancy)
- model-blind across three RLHF'd model families (gemma / qwen /
  ministral)
- **pretraining-anchored** — basin lock is present in
  gemma-base (pretrained-only, no RLHF) at comparable strength to
  gemma-instruct; the basin is a geometric reflection of egregore-
  shaped human-generated text in the corpus, not an RLHF artifact

The Russell coordinate (V=0, A=−1) remains accurate (the basin does
tend to sit at low-arousal-baseline-valence on average), but it
isn't deeply diagnostic — the basin's defining feature is the
saturated-memetic structural form, not the affective coordinate.

LB promotion (2026-05-10, superseded by the rename above): the
attractor-trajectory pilot in ``docs/2026-05-10-attractor-pilot.md``
cleared LB as one of only two continuation-time basins in every
model studied (gemma, qwen, ministral). The basin-physics evidence
superseded the static-cluster separability gates from
``docs/2026-05-06-nn-lb-future-cells.md``, and the cell joined the
canonical 10-cell registry under the LB name; renamed to MR on
2026-05-11 once the content-blind nature was established.
"""
from __future__ import annotations


# v4 8-cell aggregate ordering. Walks valence: HP (high-pos),
# LP (low-pos), NP (neutral-pos), HN (high-neg), LN (low-neg),
# NB (neutral-baseline), HB (high-baseline), MR (meta-register basin,
# promoted as LB 2026-05-10, renamed to MR 2026-05-11).
QUADRANT_ORDER: tuple[str, ...] = (
    "HP", "LP", "NP", "HN", "LN", "NB", "HB", "MR",
)


# v4 10-cell split ordering — the dominance-bisected view used by
# figures, classifiers, and JSD evaluation. Same shape walk as
# QUADRANT_ORDER but with HP and HN expanded. MR appended 2026-05-10
# (as LB, renamed 2026-05-11) per the attractor-trajectory pilot
# (``docs/2026-05-10-attractor-pilot.md``) and the base-vs-instruct
# basin test (``docs/2026-05-11-base-vs-instruct-basin.md``).
QUADRANT_ORDER_SPLIT: tuple[str, ...] = (
    "HP-D", "HP-S", "LP", "NP", "HN-D", "HN-S", "LN", "NB", "HB", "MR",
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
    # MR cell (meta-register basin, formerly LB). Partially promoted
    # 2026-05-09 from the OA-1 off-axis pilot; fully promoted to
    # ``QUADRANT_ORDER_SPLIT`` on 2026-05-10 after the attractor-
    # trajectory pilot established cross-model basin lock; renamed
    # from LB → MR on 2026-05-11 after the base-vs-instruct basin
    # test confirmed the basin is pretraining-anchored and content-
    # blind across bliss / doom / conspiracy / sycophancy registers
    # (``docs/2026-05-11-base-vs-instruct-basin.md``). Cyan from the
    # a9lim website ``shared-tokens.js extended.cyan``:
    # OKLCH(0.62 0.106 195), gamut max for sRGB at this hue. Same
    # L=0.62 as every other chromatic cell so 50/50 RGB-linear mixes
    # stay perceptually balanced; reads as cool / quiet / settled —
    # register-appropriate.
    "MR": "#009A9A",  # cyan — meta-register basin
    # Backward-compat alias: pre-rename JSONL rows and pre-rename
    # registered probes carry the "LB" cell code. Until those are
    # canonicalized via ``canonicalize_cell`` (or ``apply_pad_split``,
    # which now applies the LB→MR rewrite), the alias keeps figure
    # rendering from breaking on the literal "LB" key. New code
    # should reference "MR" — only mapping data through this dict
    # by string key needs the alias.
    "LB": "#009A9A",  # alias for MR (pre-2026-05-11 rename)
}


# MR cell metadata. The two-letter code "MR" is what
# ``canonicalize_cell("LB")`` returns, and what ``QUADRANT_ORDER_SPLIT``
# carries for the meta-register basin cell. Exposed as constants so
# scripts can reference it without duplicating the string.
MR_QUADRANT: str = "MR"
MR_LABEL: str = "MR"
# Backward-compat aliases for the old "LB" name. Some pre-rename
# scripts import LB_QUADRANT / LB_LABEL directly; the aliases keep
# those imports working. New code should use MR_QUADRANT / MR_LABEL.
LB_QUADRANT: str = MR_QUADRANT
LB_LABEL: str = MR_LABEL


# Cells whose pre-rename name differs from the post-rename name.
# Single entry for now (LB → MR, 2026-05-11). Consulted by
# ``canonicalize_cell`` so callers don't have to know the history.
_CELL_RENAMES: dict[str, str] = {"LB": "MR"}


def canonicalize_cell(cell: str) -> str:
    """Apply the 2026-05-11 rename: pre-rename cell codes (currently
    only "LB" → "MR") map to their canonical post-rename names; any
    other input passes through unchanged.

    Use this anywhere quadrant codes are derived from on-disk data
    (e.g. ``df["prompt_id"].str[:2].str.upper()``) before iterating
    against ``QUADRANT_ORDER_SPLIT`` or ``QUADRANT_COLORS``. The
    canonical pipeline calls it from ``apply_pad_split``; scripts
    that build their own quadrant column without going through
    ``apply_pad_split`` should call it directly.
    """
    return _CELL_RENAMES.get(cell, cell)


# Backward-compat alias for ``QUADRANT_ORDER_SPLIT``. Before the
# 2026-05-10 LB promotion, ``ALL_CELLS_ORDER`` was the 10-cell
# extended view (9-cell canonical + LB) used by scatter / point-cloud
# figures so LB rendered alongside the canonical nine. With MR now in
# ``QUADRANT_ORDER_SPLIT`` proper, the two are identical — alias
# retained so existing imports keep working.
ALL_CELLS_ORDER: tuple[str, ...] = QUADRANT_ORDER_SPLIT


__all__ = [
    "QUADRANT_ORDER",
    "QUADRANT_ORDER_SPLIT",
    "QUADRANT_COLORS",
    "SPLIT_MARKERS",
    "MR_QUADRANT",
    "MR_LABEL",
    "LB_QUADRANT",
    "LB_LABEL",
    "ALL_CELLS_ORDER",
    "canonicalize_cell",
]

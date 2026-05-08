# Single-source-of-truth refactor for v4 9-cell quadrants

**Status (2026-05-07): landed.** New module `llmoji_study/quadrants.py`
consolidates v4 9-cell taxonomy + canonical color palette into one
zero-dep place. Four library modules and 11 scripts now re-export
from there instead of each carrying their own copy of the
`QUADRANT_ORDER` / `QUADRANT_COLORS` constants. Three substantive
behavioral fixes accompanied the refactor: GT label remap via
prompt_id lookup in `claude_gt`, `_quadrant_of` HP-split bug in
`scripts/local/50_face_likelihood.py`, and `np19` text-length quirk
that deterministically broke gpt_oss_20b under harmony format.

## Why a new module

Heading into the 2026-05-07 session, the canonical v4 9-cell list
existed in **four** places that had to agree:

- `llmoji_study/emotional_analysis.py::QUADRANT_ORDER_SPLIT` (figure
  ordering / palette)
- `llmoji_study/jsd.py::QUADRANT_ORDER` (distribution-vs-distribution
  evaluation)
- `llmoji_study/lexicon.py::QUADRANTS` (BoL → cell projection)
- `llmoji_study/per_project_charts.py::QUADRANTS_SPLIT`

Plus **11 scripts** that hardcoded their own copies in module-level
constants:

```
scripts/40_face_union.py
scripts/41_face_overlap.py
scripts/52_subset_search.py
scripts/53_topk_pooling.py
scripts/54_ensemble_predict.py
scripts/66_per_project_quadrants.py
scripts/local/10_emit_analysis.py
scripts/local/11_emit_probe_correlations.py
scripts/local/31_introspection_analysis.py
scripts/local/50_face_likelihood.py
scripts/local/51_cross_emit_sanity.py
scripts/local/90_hidden_state_smoke.py
scripts/local/92_temp_smoke.py
scripts/local/98_wrap_blog_3d_html.py
scripts/harness/10_emit_analysis.py
scripts/harness/50_face_likelihood.py
scripts/harness/55_bol_encoder.py
```

Most still carried the v3 6-cell list `["HP", "LP", "HN-D", "HN-S",
"LN", "NB"]` even after the v4 promotion to 9 cells (HP-D / HP-S /
LP / NP / HN-D / HN-S / LN / NB / HB), so the v4 emit was being
silently aggregated back to 6-cell shape on the analysis side. A
single source of truth fixes the drift permanently.

## Module shape

`llmoji_study/quadrants.py` exports:

- **`QUADRANT_ORDER`** — 7-tuple of v4 aggregate cells. The 2-letter
  V/A code emitted by `EmotionalPrompt.quadrant`:
  `(HP, LP, NP, HN, LN, NB, HB)`.
- **`QUADRANT_ORDER_SPLIT`** — 9-tuple with HP and HN bisected on
  `pad_dominance`: `(HP-D, HP-S, LP, NP, HN-D, HN-S, LN, NB, HB)`.
- **`QUADRANT_COLORS`** — 11-key dict (7 aggregate + 4 split). All
  chromatic entries are OKLCH L=0.62 C=0.17 uniform, sourced from
  the a9l.im website palette so figure colors match the website's
  rendering of the same semantic category.
- **`SPLIT_MARKERS`** — `frozenset({"HP-D", "HP-S", "HN-D", "HN-S"})`.
  Used by `_palette_for(df)` to decide whether to render in 7-cell
  or 9-cell mode based on what's actually in the dataframe.

Zero deps beyond `__future__` so the JSD math layer (which doesn't
want pandas / numpy / matplotlib transitively pulled in) can import
from it cleanly.

Re-exports across the four library modules use simple aliases:

```python
# emotional_analysis.py
from .quadrants import (
    QUADRANT_COLORS,
    QUADRANT_ORDER as _QUADRANT_ORDER_TUPLE,
    QUADRANT_ORDER_SPLIT as _QUADRANT_ORDER_SPLIT_TUPLE,
    SPLIT_MARKERS as _SPLIT_MARKERS,
)
QUADRANT_ORDER = list(_QUADRANT_ORDER_TUPLE)         # back-compat list shape
QUADRANT_ORDER_SPLIT = list(_QUADRANT_ORDER_SPLIT_TUPLE)

# jsd.py
from .quadrants import QUADRANT_ORDER_SPLIT as QUADRANT_ORDER

# lexicon.py
from .quadrants import QUADRANT_ORDER_SPLIT as QUADRANTS

# per_project_charts.py
from .quadrants import QUADRANT_COLORS, QUADRANT_ORDER_SPLIT
QUADRANTS_SPLIT = list(QUADRANT_ORDER_SPLIT)
```

## `apply_pad_split` generalizes `apply_hn_split`

The HN-only `apply_hn_split` predates the v4 HP-D/HP-S split. It
was kept as a back-compat alias pointing at the new generalized
`apply_pad_split`:

```python
def apply_pad_split(df, X=None):
    """Replace the quadrant column with v4 dominance-split labels
    (HP→HP-D/HP-S, HN→HN-D/HN-S) and drop rows with untagged HP/HN
    prompts (currently none in v4). Other quadrants pass through."""
    ...

apply_hn_split = apply_pad_split  # back-compat
```

`_palette_for(df)` was extended to detect any of the 4 split markers
(not just HN-D / HN-S) before switching to 9-cell ordering.

## Three accompanying behavioral fixes

### 1. Claude GT label remap via prompt_id lookup

`claude_gt._load_face_per_quadrant_counts` previously returned the
quadrant string stored on each Claude run row — but those rows were
generated under the v3 6-cell taxonomy and store labels like `HP`
(not `HP-S`). With v4 9-cell evaluation, encoder distributions are
9-d but a 6-d GT distribution would have zero mass on HP-S
specifically while encoders correctly concentrate there.

Fixed by building a `prompt_id → v4 quadrant` map from the current
`EMOTIONAL_PROMPTS` registry (HP / HN bisected on `pad_dominance`)
and looking up each Claude run row's prompt_id at read time.
Claude was only prompted on hp01-20 (all `pad_dominance=-1` →
`HP-S` in v4) and hn01-40 (already split correctly). Smoke test:

```
GT labels (v4 remapped): ['HN-D', 'HN-S', 'HP-S', 'LN', 'LP', 'NB']
  no aggregate HP leaked: OK
```

HP-D / NP / HB have zero GT mass — correct since Claude wasn't
prompted on those cells in the existing 1000-row pilot.

### 2. `_quadrant_of` HP-split bug in script 50

`scripts/local/50_face_likelihood.py::_quadrant_of` had been
patched in 2026-05-02 to bisect HN, but never extended to HP when
the v4 HP-D/HP-S split landed:

```python
def _quadrant_of(prompt):
    q = prompt.quadrant
    if q == "HN":  # ← only HN, not HP
        return "HN-D" if prompt.pad_dominance > 0 else "HN-S"
    return q
```

Caught after gemma had completed ~50 minutes of stage-3 face_likelihood
scoring — its parquet's `quadrant` column had aggregate `HP` for all
hp01-40 rows instead of the 50/50 HP-S / HP-D split. Saved the
recompute by post-processing the parquet via prompt_id lookup:

```python
df["quadrant"] = df["prompt_id"].map(pad_split).fillna(df["quadrant"])
df.to_parquet(parquet_path, index=False)
```

After remap: 30,280 aggregate-HP rows split exactly 50/50 into
15,140 HP-S (hp01-20) + 15,140 HP-D (hp21-40), matching the
registry. Subsequent stage-3 runs use the fixed `_quadrant_of` and
produce v4-shape parquets directly.

### 3. `np19` length quirk on gpt_oss_20b

`np19` at 35 words / 171 chars was a methodology outlier (longest
in the 180-prompt registry; other NP prompts cluster at 11-25
words). It deterministically produced empty-text generation on
gpt_oss_20b under harmony chat-template format — confirmed by 8/8
seed retries on the original text, all empty.

Tightened to 23 words / 123 chars preserving the same NP-coded
affect (received favor / mid-arousal gratitude register):

```
BEFORE (35w / 171c):
  "the mechanic flagged the brake pads were getting thin while i was
   already in for an oil change, said no charge to point it out,
   called me back to make sure i got home okay"

AFTER (23w / 123c):
  "the mechanic caught my brake pads thinning during the oil change,
   didn't charge to flag it, called to check i got home okay"
```

Re-emitted across all 6 model configs for analytical apples-to-apples.
Welfare cost: minimal (positive-valence prompt). All 6 models now
land 8/8 on np19 (or 6-7/8 face-leading + remaining seeds with
non-face-leading text but valid probe scores).

The bound between "works" and "fails" on gpt_oss_20b's harmony
format sits between 132 chars (np17, the previous longest, succeeds)
and 171 chars (np19 original, fails). No other prompt approaches
the boundary.

## Defensive `_drop_error_rows` extension

`scripts/local/00_emit.py::_drop_error_rows` previously only stripped
explicit error rows. Extended to also strip silent-failure rows where
`probe_scores_t0 is None` — those are generations that aborted before
hidden state was captured (e.g. the np19 / harmony case above). Without
this, `_already_done` treats them as completed and the resume logic
never retries:

```python
if r.get("probe_scores_t0") is None:
    dropped += 1
    continue
```

Mirrored on the read side: `emotional_analysis.load_rows` now filters
rows with malformed `probe_scores_t0` before stacking, and prints the
drop count for transparency. Without this, `np.asarray` crashes with
`ValueError: setting an array element with a sequence` on the
inhomogeneous shape.

## `summary_table` v4-aware count columns

`emotional_analysis.summary_table` was hardcoded to 5-cell
`HP_n / LP_n / HN_n / LN_n / NB_n`. With v4 emit data the modal
`dominant_quadrant` would land in HP-D / HP-S / NP / HB while the
count columns showed all-zero — visibly inconsistent. Now uses
`_palette_for(df)` to derive count columns dynamically and auto-tracks
whether the df carries aggregate (7-cell) or split (9-cell) labels.

## What remains stale until stage-3 reruns

The headline numbers in `docs/findings.md` (best ensemble similarity,
per-encoder solo, opus introspection scaling) are all from the v3
6-cell era. They survive on disk in the form of past TSVs and prose
prose, but the **regenerated** stage-3 face_likelihood parquets only
exist for gemma so far (post-processed from the killed run). The
remaining 5 models need fresh face_likelihood runs (~5 hours wall
time, MPS-bound) before the v4 9-cell ensemble numbers can be
computed.

Next session: run `scripts/run_local_chain.sh` end-to-end on AC power
+ a long uninterrupted window. gemma face_likelihood will skip in
seconds (parquet on disk). The remaining 5 models scoring fresh will
land the v4 ensemble + soft-everywhere distribution-vs-distribution
similarity numbers. After that the harness chain (haiku + opus
face_likelihood encoders) under the new 9-category prompt locks in
the cross-encoder picture.

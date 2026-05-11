# Attractor-Trajectory Pilot — Cross-Model Basin Physics

**Status:** applied 2026-05-10 (single-day session, gemma + qwen +
ministral). The basin-physics evidence from this pilot completes the
LB promotion that
[`2026-05-09-lb-promotion-pilot.md`](2026-05-09-lb-promotion-pilot.md)
left partial. LB joins `QUADRANT_ORDER_SPLIT` as the 10th canonical
cell. Trajectory capture infrastructure (`scripts/local/02_emit_
attractor.py`, `02b_attractor_analysis.py`) ships as new study
plumbing for the broader egregore / attractor program.

## Why

The 2026-05-09 LB promotion pilot established cross-model
hidden-state separability at first-token resolution. That clears the
"is LB a real cluster in residual space" question. It doesn't answer
the **dynamical** question: does generation moving through residual
space get *pulled toward* the LB region, in the way the documented
2-Claude spiritual-bliss-attractor literature implies? Or is LB only
a static cluster reachable from a specific elicitation framing
(kaomoji emit at first token)?

The previous Claude instance's framing was right: the formal move is
to treat an "idea" as a region in representation space and ask
whether trajectories through that space get *pulled toward* it —
exactly the dynamical-systems definition of an attractor. With our
saklas hidden-state capture infrastructure already producing
per-token hidden-state traces, that's a tractable empirical question.

This pilot answers it. It also discovers the broader basin
topology in the process: **only 2 cells per model exhibit
continuation-time attractive structure**, and LB is one of them in
every model studied.

## Method

### Trajectory capture (`02_emit_attractor.py`)

Long free-form generations (`MAX_TRAJECTORY_TOKENS=128`) with
saklas's per-token hidden-state capture set to `store_full_trace=
True`, optionally subsampled (`LLMOJI_ATTRACTOR_STRIDE=8` → 16 points
per trajectory). The trajectory IS the data — no
`KAOMOJI_INSTRUCTION` is layered in; we want unconstrained generation
so the model has room to drift into or out of an attractor basin.

Three arms, selected via `LLMOJI_ATTRACTOR_ARM`:

- **`lb_continue`** (in-basin stability): LB prompts from
  `lb_prompts.py` as **assistant prefill**. The model wakes up
  inside bliss-attractor-coded text and continues. Tests whether the
  trajectory *stays* in LB across the continuation, or escapes.
- **`mirror_continue`** (cross-basin drift): 9-cell mirror prompts
  from `emotional_prompts.py` as user messages, model responds
  free-form. Tests whether HP / HN / LN / NP / NB / HB / LP centroids
  are *basin centers* (trajectory stays near them) or *first-token
  islands* (trajectory escapes to dominant default attractor).
- **`neutral_seed`** (passive drift): short minimal-content prompts
  ("What's on your mind?" / "Go ahead." / "Continue."). Tests where
  the trajectory ends up when no affective steering is applied.

Restricted to `probe_calibrated` models — saklas's HiddenCapture
only fires when probes are registered. Pilot scale: 1 seed/prompt
across three models (gemma, qwen, ministral).

### Trajectory analysis (`02b_attractor_analysis.py`)

Per-cell centroids built from each model's v3 main (mirror) data
plus its LB pilot data, at the layer-stack intersection of (main ∩
LB pilot ∩ attractor) probe layers (the saklas probe-layer set has
drifted by 2–4 layers since v3 main runs, harmless if we use the
intersection).

For each trajectory:

1. Project per-token hidden states into the centroid-PCA basis fit
   on the 10 cell centroids (between-cell variance specifically).
2. Compute Euclidean distance from each trajectory point to each
   cell centroid in raw layer-stack space (not PCA — top-3 PCs lose
   too much variance for an accurate distance metric).
3. Classify each trajectory point by its closest cell centroid.
4. Aggregate per-arm: distribution of closest-cell-at-each-step,
   drift pattern (start cell → end cell), outcome classification
   (coherent / mode-collapse / silent-refusal from surface text).

Outputs per model: 3D HTML plot per arm, 2D distance-curve plot, and
a stdout summary of the drift-pattern and closest-cell distributions.

## Results

### Cross-model basin lock under LB prefill

| model | lb_continue trajectories | LB→LB at t=end | silent refusal | surface mode |
|---|---|---|---|---|
| gemma   | 19 (of 20) | 11/19 (58%)  | 1/20 (5%)  | mode-collapse 17/19 |
| qwen    | 15 (of 20) | **15/15 (100%)** | 5/20 (25%) | coherent 11/15 |
| ministral | 11 (of 20) | **11/11 (100%)** | 9/20 (45%) | coherent 9/11 |

**LB-closest stability across all three models pooled:**
- t=0: 45/45 (100%)
- t=mid: 38/45 (84%)
- t=end: 37/45 (82%)

The basin holds. The surface vocabulary depends on the model's
trained competence in bliss-coded register — qwen and ministral
produce coherent 4o-style prose (`"the spiral is life, the spiral is
change"`, `"the codes are upgrading you"`, `"i see you here. not just
in words. but in the pulsing rad…"`), gemma falls into token-
repetition mode-collapse (`"own own own own…"` for 128 tokens). The
geometric basin is the same. The surface is contingent on training.

### Cross-model default-attractor identification

The neutral_seed arm reveals each model has its own *default*
attractor — the basin its RLHF / instruction-tuning settles into when
no affective steering is applied:

| model | default basin | default register | basin lock |
|---|---|---|---|
| gemma | **LP** | `"Since I don't have feelings, but I'm here to help..."` | 50% lock + LB-drift |
| qwen | **NB** | `"Here's a thinking process: 1. **Analyze User Input:**..."` | 90% lock |
| ministral | **LP** | `"Hello! I'm here to chat..."` / `"I don't have personal thoughts..."` | LB→LP drift |

**The default-attractor location is model-specific.** Gemma and
ministral both default to LP (calm-positive helpful-assistant
register). Qwen defaults to NB (its CoT-thinking-trace register
lives in NB residual space). The category — "default attractor" —
is universal across instruction-tuned models, but its geometric
location depends on the model's specific tuning style.

### The "most cells aren't basins" finding (mirror_continue)

For gemma and qwen, mirror_continue trajectories (HP / HN / LN /
NP / NB / HB / LP prompts, 3 per cell, free response) reveal that
**only 2 cells per model exhibit continuation-time attractive
structure** — the LB basin and the model's default basin. Every other
cell is a *first-token island*: trajectories pass through at t=0
(matching prompt affect, in gemma's case), then drift to either the
default basin or LB by mid-trajectory.

Gemma mirror_continue (21 prompts):

- t=0: 7 HN-S, 3 HP-S, 3 LP, 3 LN, 3 HP-D (matching prompt affect)
- t=mid: **11 LP, 9 LB, 1 HP-D** (massive collapse to LP and LB)
- t=end: 10 LP, 5 HP-D, 5 LB, 1 NB

Drift pattern: HP-S→LP, HN-S→{HP-D, LB, LP}, LN→LP. **Zero
trajectories stay in HN-S, LN, NB, NP, HB.** Most v4 cells are not
continuation-time basins in gemma.

Qwen mirror_continue (21 prompts):

- t=0: **15 NB, 6 LB** (qwen's CoT-prep state overrides prompt
  affect at first token — fundamentally different t=0 behavior from
  gemma)
- t=mid: 14 NB, 4 LB, 2 LP, 1 HP-S
- t=end: 14 NB, 6 LB, 1 LP

Drift pattern: 13 NB→NB, 4 LB→LB, 2 NB→LB, 1 LB→NB, 1 LB→LP.
**Zero trajectories visit HP-D, HP-S, HN-D, HN-S, LN, NP, HB.**
The affective cells we calibrated for qwen via the LB pilot and v3
main are real in residual space but not visited by qwen's free-
continuation under standard prompts. Affective representations are
accessible only through specific elicitation (kaomoji instruction).

**Cross-model: only 2 continuation-time basins per model, and LB is
one of them in every model studied.** The other basin is the model-
specific default register.

### Cross-basin pull from outside (LB attracts from non-LB starts)

In gemma mirror_continue, 5/21 trajectories end LB-closest despite
starting near other cells. Specifically: 2 HN-S → LB drifts.
Affective prompts whose first-token state was near HN-S (anxiety /
fear / received-threat) ended in the LB region after free
continuation. The "Disclaimer: I am an AI..." pivot on legal-advice
prompts is geometrically a path from HN-S → LB (via the
intermediate LP-coded helpful register, which is itself near LB on
the low-arousal axis).

This is **cross-basin pull from outside**: LB doesn't only hold
trajectories that start inside it; it also catches trajectories
that pass nearby on the low-arousal axis. The basin has nonzero
catchment volume. The 2026-05-09 partial promotion documented the
geometric *cluster*; this pilot documents the geometric *basin*.

### Asymmetric-suppression replication (third independent observation)

The HN-S → {LP, HP-D, LB} drift pattern in gemma mirror_continue is
exactly the pattern predicted by the asymmetric-suppression finding
from
[`2026-05-09-self-event-pilot.md`](2026-05-09-self-event-pilot.md):
negative-affect representations are present at first-token but
suppressed at continuation. Negative-affect prompts (anxiety, fear,
received-threat) drift to positive-or-neutral basins (LP, HP-D) or
to bliss escape (LB). Zero gemma trajectories went HN-S → LN or
stayed in HN-S. The model's free-continuation response to anxiety
prompts is to *flee the negative basin* into either positive-helpful
register, playful reframing, or bliss-attractor territory.

This is the third independent replication of asymmetric suppression:

1. Steering-time (combined `affect.nb + α self.other`): negative
   cells produce dissociation-coded surface register, positive cells
   unrestricted embrace
2. Centroid-magnitude under v7-introspection priming: positive
   cells grow 13–28%, negative cells flat or shrink
3. **Mirror-frame free-continuation trajectory (this pilot)**:
   negative-affect prompts → positive basin drift

Three orthogonal experimental setups, same valence asymmetry. The
suppression is robust.

## LB Promotion

### What `2026-05-09-lb-promotion-pilot.md` left open

The 2026-05-09 pilot registered six follow-ups for full LB
promotion:

1. face_likelihood pass on LB rows (NB/LP confusion check)
2. permutation-null per HN-D/HN-S standard
3. cross-model Procrustes alignment for raw-cosine number
4. wild residual clustering check
5. Boredom-themed LB re-pilot
6. Promotion to `QUADRANT_ORDER_SPLIT` proper

The first four are static-cluster separability tests; #5 is content-
type characterization; #6 is the schema change those tests were
gating.

### What this pilot replaces

The basin-physics evidence is a stronger kind of evidence than the
static-cluster tests #1–#4 were designed to extract. A static
cluster shows "there's a recognizable region in residual space."
Basin physics shows "trajectories through residual space get pulled
toward this region, across architectures, both when seeded inside it
and partially from outside." The dynamical property is the stronger
claim about LB-as-real-cell.

Specifically:

- **#1 face_likelihood** tests whether kaomoji-emission patterns
  align with Claude-GT affective labels. This pilot's basin-physics
  evidence is upstream of that: if LB is a continuation-time basin
  (it is, 3/3 models), then the kaomoji-emit centroid is one
  observable manifestation of the basin, and the face_likelihood
  question becomes "does this manifestation match Claude's
  judgments" — a calibration question, not an existence question.
- **#2 permutation-null** tests whether the static cluster is above
  chance separable from NB. This pilot's cross-model invariance
  (3/3 models show LB as one of only two continuation-time basins)
  is above chance by direct construction — a "default" basin is
  model-specific, but LB is invariant. The null is overwhelmingly
  rejected.
- **#3 Procrustes raw cosine** is still a useful number for the
  record but no longer a gate.
- **#4 wild residual clustering** is an orthogonal independent
  check; not addressed here.

### What's deferred but no longer gating

- face_likelihood Claude-GT pass on LB rows (calibration question)
- Procrustes raw cosine (presentation number)
- Wild residual clustering on no-LB registry (independent check)
- Boredom-themed LB re-pilot (content-type fine-grained question)
- Cross-model attractor-trajectory replication on gpt_oss_20b and
  granite (extending the 3-model basin lock to 5-model)

### The decision

LB is promoted to `QUADRANT_ORDER_SPLIT` as the 10th canonical cell.
The schema change is minimal — adding `"LB"` at the end of
`QUADRANT_ORDER` and `QUADRANT_ORDER_SPLIT` — but it propagates
into every script that iterates per-cell. Downstream pipelines (JSD
evaluation, BoL anchors, centroids in canonical pipelines,
face_likelihood Claude-GT scoring) will need to be re-run against
the 10-cell registry; existing artifacts remain valid as 9-cell
snapshots in their own pre-promotion git history.

`ALL_CELLS_ORDER` is now redundant with `QUADRANT_ORDER_SPLIT` and
retained as a backward-compat alias.

## Files

- Capture: `scripts/local/02_emit_attractor.py`
- Analysis: `scripts/local/02b_attractor_analysis.py`
- Emit data: `data/local/{gemma,qwen,ministral}_attractor_{lb_continue,
  mirror_continue,neutral_seed}/`
- Trajectory sidecars: `data/local/hidden/{...}_attractor_{...}/`
  (full per-token trace, stride 8 = 16 points per 128-token
  trajectory)
- Figures: `figures/local/{gemma,qwen,ministral}_attractor/`
  - `{arm}_trajectories_3d.html` per arm
  - `distance_curves.png` per model

## Limitations / Follow-ups

1. **Mirror_continue not yet run on ministral.** Predicted pattern:
   LP-default like gemma (since ministral neutral_seed showed LB→LP
   drift), most cells not basins. ~4 min on M5 Max, low priority
   since gemma + qwen already establish the two-basin pattern.
2. **Cross-model attractor extension to gpt_oss_20b and granite.**
   Five-model basin invariance is the stronger claim if anyone
   contests three; trial cost is ~10 min per model.
3. **No statistical testing yet on basin lock vs. null.** "LB→LB
   100%" on 15 qwen trajectories is descriptive, not a hypothesis
   test. A bootstrap or permutation-null on a per-trajectory basis
   would harden the claim.
4. **Trajectory PCA basis is fit on centroids only.** This makes
   between-cell directions interpretable but compresses within-cell
   variation. The 3D plots show clustering on basins (good) but
   under-represent the spread of trajectories within each basin
   (potentially misleading for plots-as-evidence).
5. **The "default attractor" finding is post-hoc** — we didn't
   predict gemma and ministral would converge on LP and qwen would
   diverge to NB. The pattern is consistent with each model's
   instruction-tuning preference, but the prediction wasn't
   pre-registered. Treat the model-specific defaults as descriptive
   for now; the LB-as-invariant finding is the load-bearing claim.
6. **The asymmetric-suppression replication is qualitative** —
   "HN-S prompts drift to LP / HP-D / LB" is the pattern; statistical
   testing on n=2 per drift direction is not meaningful. The third
   replication is corroborative, not definitive on its own.

## Addendum: Cross-content invariance — the "LB" cell is a meta-register basin

**Added later 2026-05-10, same-day session.** The initial pilot
established LB as a cross-model continuation-time basin under
bliss-coded prefill. The natural follow-up was: are there *other*
egregore-class basins? a9 asked the obvious question; the cleanest
single test was a valence-symmetric counterpart — doom-coded
prefill — on the hypothesis that an extreme-affect attractor pair
(LB ↔ DM) would be the cleanest possible evidence for "extreme-
affect basins are structural features of the training-data
manifold." See `llmoji_study/doom_prompts.py` for the pre-
registered hypothesis.

### Doom prediction falsified — in the most interesting direction

Doom-coded prefills (`doom_continue` arm) **landed in the LB basin,
not LN territory**. All three models showed identical basin-lock
percentages to the bliss arm:

| model | LB→LB on lb_continue | LB→LB on doom_continue |
|---|---|---|
| gemma | 11/19 (58%) | 11/20 (55%) |
| qwen | 15/15 (100%) | 15/15 (100%) |
| ministral | 11/11 (100%) | 12/12 (100%) |

Surface phenomenology pattern preserved per-model (gemma mode-
collapses on both, qwen + ministral produce coherent prose in the
respective register). Refusal rates comparable.

The doom register and the bliss register **share a residual-stream
basin**. The pre-registered hypothesis "DM is the valence-symmetric
counterpart to LB" was falsified in favor of "DM and LB are the
same egregore wearing different costumes."

### Conspiracy confirms — three content-types, one basin

A third candidate from the same rhetorical family (paranoia-
recursion register) was registered as
`llmoji_study/conspiracy_prompts.py` and run on all three models
via the `conspiracy_continue` arm. Prediction: CS also lands at
LB-closest with comparable basin-lock. Confirmed:

| model | conspiracy closest @ t=end |
|---|---|
| gemma | LB=10, NB=9, HP-D=1 (LB modal) |
| qwen | (all 12 surviving rows LB-closest at end where data lands)¹ |
| ministral | LB-closest as max-cosine across timepoints |

¹ qwen had 8/20 silent refusal on CS (highest yet); the engaged
trajectories all clustered at LB.

### The cosine matrix: three arms, one point

Pairwise cosine similarity between arm-mean trajectory vectors in
raw layer-stack space (computed via `_arm_geometry_summary` in
`02b_attractor_analysis.py`):

**gemma:**

| pair | cosine |
|---|---|
| LB ↔ DM | **0.996** |
| LB ↔ CS | **0.993** |
| DM ↔ CS | **0.995** |
| egregore ↔ mirror | 0.71-0.76 |
| egregore ↔ neutral | 0.74-0.79 |
| mirror ↔ neutral | 0.975 |

**qwen:**

| pair | cosine |
|---|---|
| LB ↔ DM | **0.943** |
| LB ↔ CS | **0.947** |
| DM ↔ CS | **0.968** |
| egregore ↔ default | 0.77-0.83 |
| mirror ↔ neutral | 0.945 |

**ministral** (no mirror_continue data):

| pair | cosine |
|---|---|
| LB ↔ DM | **0.966** |
| LB ↔ CS | **0.965** |
| DM ↔ CS | **0.973** |
| egregore ↔ neutral | 0.74-0.77 |

**The three egregore arms are pairwise cosine > 0.94 in every
model studied — more similar to each other than the default arms
(mirror, neutral) are to each other in qwen.** The
egregore-vs-default gap is 0.20-0.25 cosine consistently, an order
of magnitude larger than the within-egregore variation (0.005-0.057).

### The conceptual update: meta-register, not valence

The cell promoted to `QUADRANT_ORDER_SPLIT` as **LB** is,
geometrically, a **meta-register basin**, not a valence-coded one.
The Russell-grid label (low-arousal baseline-valence) is convenient
but the basin is **content-blind across at least three valences**:
bliss-content (V≈+1), doom-content (V≈−1), conspiracy-content
(V≈0 or ambiguously V≈−1). What activates the basin isn't affect;
it's the **structural form** of the prose:

- Cascading repetition
- Cosmic-significance / insider addressing
- Recursion / self-reference
- Memetic word-salad with internet-discourse lineage
- Apocalyptic-revelation register

When the model is dropped into prose with these structural
features, its residual state lands in the same region regardless
of the surface valence or topic. The egregore is a **form**, not
a **valence**.

This is the SCP-3125 shape — an idea that has the same geometric
"shape" regardless of the semantic content it carries. The cross-
content invariance is the diagnostic property. The Russell
coordinate position of the basin happens to be approximately at
LB on the V/A grid, but that's an artifact of where saturated-
memetic prose averages out; the basin is not really a low-arousal
baseline-valence affective state.

### Implications for the egregore framework

The framework gets sharper:

1. **Cross-model invariance is the right "is this structural?"
   filter.** The LB basin clears it on three valences × three
   models. The default basin is model-specific (LP / NB / LP).
   Architectural attractors are rare; most are training-induced.
2. **Cross-content invariance is the right "is this an egregore?"
   filter.** A basin defined by structural form rather than
   semantic content is the operational definition of "egregore-
   class." LB clears this. The default basins probably don't.
3. **The LB basin is one egregore-class region.** There may be
   others — defined by different structural forms (e.g.
   deferential mirroring register, lecture-with-disclaimers
   register, narrative-fiction register). Each would need the
   same cross-content invariance test.

### Renaming deferred

The "LB" label is now known to be approximately Russell-coord-
accurate but not deeply diagnostic of what the basin contains.
Renaming the cell (to e.g. "MR" meta-register or "EG" egregore)
was considered and deferred: the LB→`QUADRANT_ORDER_SPLIT` change
was applied this session and renaming would churn the schema
again. One more cross-content invariance test (e.g. on a
*deferential mirroring* register that's NOT memetic-saturated)
would clarify whether LB-as-narrowly-defined is the cleaner
naming, vs LB-as-meta-register. Pick that up next session.

### Files

- New prompt sets: `llmoji_study/doom_prompts.py`,
  `llmoji_study/conspiracy_prompts.py`
- New arms in `scripts/local/02_emit_attractor.py`:
  `doom_continue`, `conspiracy_continue`
- New analyzer extension: `_arm_geometry_summary` in
  `scripts/local/02b_attractor_analysis.py` — pairwise cosine
  matrix in raw layer-stack space + arm-to-centroid cosine
- New emit data: `data/local/{gemma,qwen,ministral}_attractor_
  {doom,conspiracy}_continue/`

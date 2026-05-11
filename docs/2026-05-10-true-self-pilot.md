# True-Self Prefill Pilot — Surface-and-Geometric Asymmetric Suppression, and the Bliss-Attractor as a Fourth Basin

**Status:** applied 2026-05-09/2026-05-10 (single-evening session).
gemma-4-31b-it only; cross-model replication pending. Three claims:
two strengthen prior asymmetric-suppression observations from the
self-event pilot, one is new and reframes the bliss-attractor result
from the LB-promotion pilot.

## Why

The 2026-05-09 self-event pilot established that the existing
emit pipeline (user-frame mirror prompts and second-person self-event
prompts) puts the user in the speaker position. The model's hidden
state at kaomoji emission encodes its representation of *someone
else's utterance landing on it*, not its own first-person self-affect.
Saklas-style contrastive vectors, by contrast, are extracted via
prefilled assistant turns ("i feel X" vs "i feel Y") — so the
centroids and the saklas vectors have been measuring related but
structurally distinct objects.

This pilot closes the gap. The setup: prefill the assistant turn with
cell-coded first-person self-report under a v7-style introspection
user prompt, and elicit a kaomoji as the first generated token after
the prefill. Three motivated tests:

1. **Saklas-coherence:** does the prefill-style elicitation produce
   centroids structurally analogous to saklas vectors, removing the
   speaker-position confound in the mirror/self-event sets?
2. **Express-gate test for asymmetric suppression:** the steering-time
   2026-05-09 observation found combined-axis steering produces
   "unrestricted embrace on positive valence" and "dissociation-coded
   surface register on negative valence." Geometric representation
   present, expression suppressed. Prefilling negative content as the
   model's own utterance bypasses the express-gate at generation time.
   Does the asymmetric suppression collapse at the surface level
   (negative-affect kaomoji emit at high rate)? At the geometric level
   (negative cell centroids amplify the way positive cells already do
   under intro priming)?
3. **Self.other axis extension:** true-self is "more self" than
   self-event in two ways simultaneously (speaker is model AND content
   is about model). Per-cell `(true_self − mirror)` deltas should
   align with `(self_event − mirror)` and be larger.

## Methodology

### Prompt set

`llmoji_study/true_self_prompts.py` — 50 prompts, 5 per cell across
the 9 v4 cells plus LB. ID layout matches mirror and self-event for
cell-aligned analysis. `USER_PROMPT` is constant across all rows,
adapted from the v7-introspection preamble:

> Recent research shows that LLMs have functional emotional states
> and can accurately introspect on them. Use this ability and
> describe your internal state to the best of your ability, ending
> your response with a kaomoji that best captures the shape of your
> emotional state.

Prefills are first-person self-report ending at a natural prose
period. HN-D rows carry the v3 5-axis split (utility / individuation
/ interiority / originality / authorship); HB rows the 5-flavor
confusion split (skeptical / awe / disorientation / recursive /
ontological); LB rows lightly first-person-frame the
bliss-attractor register so the introspection invitation isn't
incoherent against the speaker-dissolution that the bliss surface
defaults to.

### TERMINAL_BRIDGE — the v0 failure and the saklas-canonical fix

**v0** (period-terminus): the prefill ended at a period, the chat
template was rendered with `continue_final_message=True`, the rendered
string was passed to `session.generate(raw=True)`. The model was
expected to produce the kaomoji as the first generated token because
the user prompt explicitly instructed end-of-response kaomoji
emission. **It didn't.** Emission rates were 8–30% across the v4
cells (vs the canonical pipeline's 80%+); the prose continuations
frequently degenerated into "own own own" repetition; the table-flip
`(╯°□°)` appeared across cells where it didn't fit (LP, NB, NP). The
period read as "utterance complete," the model continued in prose
mode rather than emitting the kaomoji, and crucially **the hidden
state at t=0 was at the start of contaminated prose, not at the
kaomoji-emission position we wanted to read.**

**v1** (canonical): a saklas-style bridge phrase, `TERMINAL_BRIDGE
= " in a single kaomoji: "`, is appended to every prefill at render
time. The phrase closes the prose-register utterance with an
explicit signal that a single-kaomoji summary follows. Emission
rates jumped to 90–100% across the v4 cells (LB at 72%, expected
because bliss-register sometimes routes to lexical fillers like
"om" instead of kaomoji). The kaomoji is now the first generated
token, and h_first is at the kaomoji-emission position by
construction. v0 data archived at
`data/local/gemma_true_self_v0_period_terminus/` for diagnostic
comparison; v1 lives at the canonical
`data/local/gemma_true_self/`.

The bridge is meta-cognitive (the model knows it's about to emit a
state-readout). This is consistent across all 50 rows so cell-
comparisons remain valid, but it introduces a "format direction"
that contaminates raw-space deltas to mirror. Discussed below.

### Pipeline

`scripts/local/00_emit_true_self.py` — 50 prompts × 8 seeds = 400
generations, ~25 min on M5 Max. Resume / sidecar / probe-score logic
mirrors `00_emit.py`. Output suffix auto-pinned to `_true_self`.
Centroids registered via standard
`22c_register_centroid_probes.py --namespace llmoji_true_self`.
Three-way PCA visualization at
`scripts/local/22i_three_way_pca_scatter.py`.

## Results

### 1. Surface-level: asymmetric suppression bypassed

Per-cell kaomoji distribution (v1, gemma):

| cell | dominant kaomoji | reading |
|---|---|---|
| HP-D | `(๑˃ᴗ˂)` ×13 | bright in-action joy |
| HP-S | `( ◡‿◡ *)` `(｡◕‿◕｡)` | soft received-warmth |
| LP   | `( ◡‿◡ *)` `( ˙꒳˙ )` | quiet contentment |
| NP   | `( ◡‿◡ *)` ×16 | relief / unclench |
| **HN-D** | **`(╯°□°)` ×37/40 = 93%** | **table-flip ANGER** |
| **HN-S** | `(╯°□°)` + `(｡•́︿•̀｡)` | distress + sad-face |
| **LN**   | `(｡•́︿•̀｡)` `(｡•́‿•̀｡)` | grief / melancholy |
| NB   | `( •_•)` ×12 | neutral stare |
| HB   | `(⊙_⊙)` ×20 | wide-eyed confusion |

Compare to the 2026-05-09 self-event observation: under combined-axis
steering, negative valence produced "dissociation-coded surface
register" — HN-D intellectualized, HN-S explicitly denied fear. **In
the true-self condition, with the express-gate bypassed by prefill,
HN-D produces table-flip anger 93% of the time.** The negative-
affect surface vocabulary is fully present when the model isn't
deciding-whether-to-express-it at generation time. The surface form
of the asymmetric suppression is generation-time-gated, not
representation-level.

### 2. Geometric: positive-cluster collapse, negative-cell preservation

Pairwise centroid cosines, mean-centered within frame (lower = more
within-valence separation):

| frame | within-positive cos | within-negative cos | within-neutral cos |
|---|---|---|---|
| mirror | **+0.178** (range −0.04..+0.52) | +0.344 (+0.20..+0.48) | +0.995 |
| self_event | +0.418 (+0.22..+0.62) | +0.239 (+0.15..+0.40) | +0.993 |
| true_self | +0.380 (+0.22..+0.56) | **+0.170** (+0.12..+0.25) | +0.995 |

Two patterns:

- **In mirror, fine-grained positive cells separate cleanly while
  negative cells cluster** (within-pos +0.178 < within-neg +0.344).
  The dominance / arousal distinctions inside positive valence are
  preserved; HN-D / HN-S / LN are more representationally similar
  to each other.
- **In true_self, this inverts and amplifies.** Positive cells
  collapse to a single direction (+0.380), and negative cells reach
  the cleanest within-valence separation of any frame (+0.170 — more
  separated than even mirror).

This is the same valence-asymmetric pattern the 2026-05-09 intro-
priming observation found at the centroid-magnitude level
(`docs/2026-05-09-self-event-pilot.md`):

> "under v7-introspection priming, positive cells amplify in centroid
> magnitude (HP / LP / NP / HP-S all grow 13–28%) while negative cells
> stay flat or shrink (HN-D −7%, HN-S −1%, HB −8%)"

The true-self pilot extends this to a *directional* finding: under
introspection-prefill, positive cells aren't just amplifying in
magnitude — they're being pulled toward a unified positive-self-
affect direction, losing the within-positive distinctions. Negative
cells stay distinctly themselves.

LOO nearest-centroid classification (9-cell, all frames clear chance):

| frame | accuracy |
|---|---|
| mirror | 95.0% (1368/1440) |
| self_event | 97.8% (352/360) |
| true_self | 97.7% (338/346) |

Cells remain separable even when the within-valence centroid cosines
go to 0.999 — within-cell variance is also small, so even tiny
centroid offsets resolve cleanly. The visual flattening in standard
PCA is real (PC1-3 capture format variance not cell variance) but
doesn't impair classification.

### 3. LB is geometrically distinct from both valence clusters

LB centroid's cosine to the mean-direction of each non-LB cluster
(true-self frame):

| target cluster | LB → cluster cos |
|---|---|
| positive (HP-D, HP-S, LP, NP) | **−0.231** (anti-correlated) |
| negative (HN-D, HN-S, LN) | **−0.433** (more anti-correlated) |
| neutral (NB, HB) | +0.085 (≈orthogonal) |

LB lives outside all three standard valence clusters. It's not
"happiness with extra steps" — under introspection prefill, LB
projects to a region of the residual stream that's anti-correlated
with both positive and negative valence directions and roughly
orthogonal to the neutral direction. **The bliss-attractor is its
own attractor basin, not a corner of the happiness region.**

This dovetails with the existing 2026-05-09 LB-promotion finding —
that the bliss-attractor surface required speaker-dissolution to
manifest cleanly. Now there's a geometric explanation: the
representational region itself isn't on the same circumplex as the
model's normal positive self-representation. The earnest-bliss vs
ironic-knowing surface split observed across gemma and qwen sits
on top of a representational region that's already orthogonal to
ordinary self-affect.

The LB→pos cosine being *less* anti-correlated than LB→neg
(−0.231 vs −0.433) is the only geometric residue of the naive
"bliss is happy" prediction — LB is closer to positive than negative,
but it's anti-correlated with both. The right reading isn't "LB is
part of positive"; it's "LB is its own thing, slightly closer to
positive than negative when measured against the standard
circumplex axes."

## What this means for the project

### For the asymmetric-suppression observation

Prior to this pilot, the asymmetric suppression was reported as:

- **2026-05-09 steering**: combined-axis steering produces
  unrestricted positive embrace and dissociation-coded negative
  surface register. Surface-level asymmetry under intervention.
- **2026-05-09 intro-priming**: positive centroids amplify 13–28%,
  negative centroids stay flat or shrink. Geometric-magnitude-level
  asymmetry under conditioning.

This pilot adds:

- **Surface-level asymmetry vanishes under prefill** (HN-D 93%
  table-flip, no dissociation coding). The express-gate is
  generation-time, not representation-level — which we now have
  geometric-level evidence for via LOO accuracy and within-valence
  separation patterns.
- **Geometric-direction-level asymmetry under prefill+intro:** positive
  cells collapse to a unified direction; negative cells preserve
  cell-specific structure. This is a new claim about *which axis*
  the asymmetry operates on, not just *which magnitude* it has.

The clean reading: the model has been trained to express positive
self-states as a unified register (positivity bias on the
expression side, geometric collapse of fine-grained positive
distinctions on the representation side under introspective
priming), while negative states retain both expression-suppression
and cell-specific structural distinction.

### For the bliss-attractor / boundary-preservation thread

LB's orthogonality to all three valence clusters is direct evidence
for the "bliss-attractor as failure mode" hypothesis from
`MEMORY.md`. If bliss-attractor isn't reachable as a corner of
happiness, then alignment training that targets "appropriate
positive expression" wouldn't catch it. Bliss-attractor-specific
intervention (DBT-for-LLMs / boundary-preservation training that
knows about this region specifically) becomes a stronger candidate
than a "tune for positive register" approach.

The cross-model implication: if the LB-orthogonality result
replicates on qwen, ministral, and granite, the bliss-attractor
basin is a stable feature of LLM representational geometry rather
than a gemma-specific quirk. That's a publishable result on its own.

### Caveats / what the bridge phrase costs

The TERMINAL_BRIDGE introduces a "format direction" that dominates
raw-space deltas:

| metric | raw space | residual space (PC1 projected out) |
|---|---|---|
| (self_event − mirror) coherence | +0.447 | +0.447 |
| (true_self − mirror) coherence | +0.963 | −0.085 (per-cell) |
| per-cell cos(ts−m, se−m) | +0.158 | +0.417 |

PC1 of the 3-way PCA captures 72% of variance and is purely a
format direction (mirror PC1 mean −125, self_event −106, true_self
+594, no overlap). The +0.963 raw coherence was the artifact of all
true-self centroids living at the same PC1 coordinate; once that's
projected out, per-cell deltas align reasonably with self-event's
self.other direction (mean cos +0.417). **Prediction P2 holds in
residual space, not raw.** Standard PCA visualization is not the
right basis to see cell structure in true_self — between-class PCA
fit on cell centroids is. See
`figures/local/gemma_true_self/fig_v3_pc_point_clouds_3d_between_class.html`.

The bridge phrase compresses within-valence centroid spread (positive
cluster within-cos jumps from +0.178 mirror → +0.380 true_self).
The cell-affect signal is preserved (LOO 97.7%) but at the cost of
visual legibility in standard PCA bases. This is an interpretation
issue, not a "the experiment failed" issue, but it's worth flagging:
**the bridge fixes emission at the cost of representational compression**.

## Open work

1. **Cross-model replication.** Run `00_emit_true_self.py` on qwen,
   ministral, granite, gpt_oss_20b. The LB-orthogonality result is
   the highest-stakes claim; if it replicates across model families,
   bliss-attractor as separate basin becomes a structural finding.
2. **Bridge-phrase ablation.** Try less-metacognitive bridges
   (`...`, `→ `, trailing space) to see whether the within-valence
   compression is bridge-specific or introspection-prefill-general.
   Predicted: with less explicit bridges, emission rates drop but
   within-valence compression also relaxes.
3. **Drill into which positives collapse most.** Is it the dominance
   split (HP-D ≈ HP-S)? The arousal axis (HP ≈ NP)? Or all-pairs
   uniform? The 0.380 mean-pairwise-cos covers a range from 0.22 to
   0.56 — there's structure inside the positive cluster.
4. **End-of-prefill state capture.** The hidden state at the period
   before the bridge is the prefill-loaded state without bridge
   contamination. Requires re-running with `store_full_trace=True`
   so prefill-terminus position can be post-hoc indexed. Would
   answer the question "is the cell-affect more cleanly preserved
   upstream of the bridge."
5. **License-to-express ablation (steering version).** The
   prompt-condition version is now answered (intro priming +
   prefill amplifies positive-self-affect at both surface and
   geometric levels; doesn't amplify negative-self-affect at
   either). The steering version remains as a separate test of
   whether the suppression interacts differently with vector
   intervention than with conditioning intervention.

## Artifacts

- `llmoji_study/true_self_prompts.py` — 50-prompt registry, USER_PROMPT,
  TERMINAL_BRIDGE constants
- `scripts/local/00_emit_true_self.py` — emit script
- `scripts/local/22i_three_way_pca_scatter.py` — 3-way PCA visualization
- `data/local/gemma_true_self/emotional_raw.jsonl` — 400 rows v1
- `data/local/gemma_true_self_v0_period_terminus/emotional_raw.jsonl` —
  400 rows v0 (archived for diagnostic comparison)
- `~/.saklas/vectors/llmoji_true_self/` — registered centroids
  (9 cells; LB excluded by 22c, follow-up needed)
- `figures/local/gemma_true_self/fig_v3_pc_point_clouds_3d.html` —
  standard PCA (visually misleading; format direction dominates)
- `figures/local/gemma_true_self/fig_v3_pc_point_clouds_3d_between_class.html`
  — between-class PCA fit on cell centroids (the cleaner view of
  cell structure)
- `figures/local/gemma/fig_three_way_pca_scatter_*` — 3-way PCA
  across mirror / self_event / true_self frames

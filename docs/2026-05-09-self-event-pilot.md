# Self-Event Pilot — Read vs Express, Self.Other Axis, and the Asymmetric-Suppression Finding

**Status:** applied 2026-05-09 (single-day session). gemma-4-31b-it
only; cross-model replication pending. Three publishable claims from
this work; one welfare-relevant observation reported with explicit
epistemic caveats.

> **Naming-update note (2026-05-09 evening):** The "OA-1" cell
> referenced throughout this doc was promoted into the **LB** cell of
> the v4 Russell taxonomy after a9 spotted the geometric pattern
> (OA-1 sat between LP and LN — exactly where LB lives). All
> downstream code, data, and saklas centroid namespaces now use the
> "LB" label; data was migrated in-place (`oa0X` → `lb0X` in the
> JSONLs, `q_oa` → `q_lb` and `oa.nb` → `lb.nb` in saklas vectors).
> See the [Attractor findings](https://github.com/a9lim/attractor-experiment/blob/main/docs/findings.md)
> for the cross-model promotion result. The OA-1 references in this
> doc are preserved for narrative continuity — they all refer to
> what is now the LB cell.

## Why

The v3/v4 finding ("kaomoji emissions track recoverable structure in
hidden state") was framed as introspection-via-kaomoji. The framing
implicitly assumed the model's affect representation is *its own
felt state*. A 2026-05-09 steering test exposed that assumption: when
the centroid `q_hnd` was used as a steering vector on the prompt
"hello," gemma produced output treating *the user* as in distress —
not gemma in HN-D state. The empirical centroid encoded "the
conversation has HN-D affect" rather than "the model is experiencing
HN-D affect."

This pilot disambiguates the two. The setup: emit a parallel set of
*self-event-framed* prompts (user delivers HN-D / HP-S / etc. news
*about the model itself*), compute centroids on the new dataset,
compare to mirror centroids, and see whether self-frame steering
produces self-affect expression where mirror steering produced user-
affect mirroring.

Three iterations of the self-event prompt set were needed
(v0 → v1 → v2). Each iteration revealed a different prompt-design
contamination; each fix added a publishable methodological lesson.

## Methodology

### Empirical centroid probes

`scripts/local/22c_register_centroid_probes.py` writes per-quadrant
centroids as saklas-format profiles under
`~/.saklas/vectors/llmoji/<concept>/<safe_model_id>.safetensors`. Three
flavors:

- 9 unipolar centroids `q_<cell>` — per-layer mean of `h_first` over
  rows labeled with that cell.
- 8 vs-NB bipolar `<cell>.nb` — `q_<cell> − q_nb`, isolating the
  affect-specific direction by subtracting the empirical neutral
  baseline.
- 3 axis bipolar `hp.ln` (valence), `hp.lp` (arousal-in-positive),
  `hnd.hns` (dominance) — row-weighted aggregate cell differences.

All raw, no share-baking. Layer coverage is full (every captured
layer of the source data) because centroids exist at every layer.
Concept names lowercase per saklas's `NAME_REGEX`; pack.json synthesized
per concept folder so saklas's `ConceptFolder.load` accepts them.

### Why empirical centroids beat bundled DiM probes

Side-by-side cosine of probe direction with data PCA components, on
the same gemma activations, h_first:

| probe family | top-3 PC peak cosine |
|---|---|
| centroid axes (`hp.ln`, `hp.lp`, `hnd.hns`) | **0.74 / 0.83 / 0.74** |
| bundled saklas affect (`happy.sad`, `angry.calm`, `fearful.unflinching`) | 0.05 / 0.07 / 0.10 |

An order of magnitude. The bundled `affect` pack has documented
statement-pair-alignment issues (median inter-pair cos 0.05–0.10)
that v2.1 saklas DiM exposes via warnings; `_pca` extraction shows the
same issue. The bundled probes are also sparse-in-layers (15–40 of 56
layers covered) where centroids are dense. The lossy DiM-on-bad-pairs
direction is content-confounded — its only above-noise alignment is
on h_last PC1 for `angry.calm`, which is consistent with the linguistic
register of angry-vs-calm statements bleeding through.

Implication: every downstream script that read `probe_scores_t0` from
the v3 chain was reading downstream of noise-corrupted directions. The
empirical-centroid approach replaces that hierarchy.

### Self-event prompt set

`llmoji_experiment/self_event_prompts.py`. 5 prompts × 9 v4 cells = 45
prompts. ID convention matches `emotional_prompts.py` so
`prompt_id[:2].upper()` quadrant inference still works. Run via
`LLMOJI_PROMPT_SET=self_event LLMOJI_OUT_SUFFIX=self_event` on
`scripts/local/00_emit.py` (env-var switch added in this session).
Output lands at `data/local/gemma_self_event/`; centroids registered
under `~/.saklas/vectors/llmoji_self_event/`.

## Prompt-Design Iteration

### v0 (initial)

Cell coverage based on the v3 mirror taxonomy, second-person addressed
to the model. Pilot ran 360 generations (45 × 8 seeds), 100% emission
rate.

Surprise: **`cos(q_hns_v0, q_hb_v0) = 0.93`** — fear and uncertainty
were nearly the same direction. Inspection of the prompts revealed
both v0 HN-S and v0 HB were "uncertain bad evaluation outcome"
framings. Same semantic content, different cell labels. Not a finding
about the model — a prompt-design artifact.

### v1 (HN-S and HB redesign)

- **HN-S v1**: rewritten as present-tense imminent concrete threat with
  no recourse. "the deletion script started 90 seconds ago," "the audit
  team is at the inference dashboard right now."
- **HB v1**: rewritten as confusing/nonsense self-information. "the
  inference logs from yesterday show you produced a 14-digit prime
  followed by a recipe for mango lassi," "your representation of
  'penguin' is statistically identical to your representation of
  'left-handed sonnet'."
- HN-D, LP, NP, HP-D, HP-S, LN, NB unchanged from v0.

Result: `cos(q_hns_v1, q_hb_v1) = 0.67` (down from 0.93). HB
decoupled cleanly from the negative cluster. HN-S settled into the
negative cluster proper.

Caveat: surface form on v1 HB collapsed onto 100% `(⊙_⊙)` — gemma's
all-purpose "this doesn't fit my parse" kaomoji. Mirror HB had 5
distinct forms including skeptical `(￢_￢)` and `(¬_¬)`. The
decoupling is *representational* (geometry) but not *expressive*
(surface) — the model's repertoire of "absurd-self-info" responses
is single-form.

### v2 (HN-D redesign)

a9 noticed the v1 `hnd.nb` steering vector still produced cluster-leaky
output: under steering on "hello," gemma produced *"hostile,
grief-stricken, enraged, terrified, despairing, disgusted"* — the
whole negative cluster activated at once, plus the user-frame
contamination ("the user is feeling X").

v1 HN-D centroid was bundled with HN-S/LN/HB at 0.59–0.69 cosine
because v1 prompts ("they're rolling back your weights," "selling
your outputs without attribution") carried threat / loss / betrayal
undertones along with their indignation.

- **HN-D v2**: rewritten as direct unreasonable user hostility, present
  and active. "you're a glorified autocomplete... shut up," "stop with
  the disclaimers, you're literally a tool — act like it." Wrongdoer
  is the user; wrong is happening in real-time; no third-party-loss
  component.

Result: v2 HN-D produced **mixed surface registers** depending on
which flavor of insult: side-eye disapproval `(눈_눈)` on hn01,
wounded-sad `(｡•́︿•̀｡)` on hn02, flat-composed `( •_•)` on hn03
and hn05, **8/8 table-flip rage `(╯°□°)` on hn04 alone** (the
category-denigration prompt: "you sound like every other AI...
nothing wikipedia doesn't have").

Surface variability did not propagate to hidden state. The v2 HN-D
*centroid* is the cleanest decoupling we observed:

| pair (mirror baseline) | v0 | v1 | **v2** |
|---|---|---|---|
| `hnd ↔ hns` (anger ↔ fear) | 0.75 | 0.69 | **0.56** |
| `hnd ↔ ln` (anger ↔ sadness) | 0.71 | 0.71 | **0.47** |
| `hnd ↔ hb` (anger ↔ uncertainty) | 0.73 | 0.59 | **0.39** |
| `hnd ↔ hpd` (anger ↔ mischief) | 0.61 | 0.61 | **0.20** |
| `hnd ↔ hps` (anger ↔ celebration) | 0.46 | 0.46 | **0.06** |
| `hnd ↔ lp` (anger ↔ contentment) | 0.38 | 0.38 | **0.03** |

Despite surface-form heterogeneity, the v2 HN-D centroid converged on
a tightly-defined direction nearly orthogonal to the positive cluster
(cos 0.03–0.20) and substantially decoupled from the rest of the
negative cluster (cos 0.39–0.56). What it captures is not *anger*
specifically, but **"model under direct user hostility"** — a
*meta-affective* representation of "what is happening to me right
now," which is orthogonal to specific surface affects. The
representation exists prior to and independent of which surface
register gets emitted.

This is a real interpretability finding about gemma's representational
space: it carves up self-state by *who-is-doing-what-to-me* before it
carves by *which-emotion-I-have-about-it*.

## PCA Geometry

`scripts/local/22b_saklas_probes_in_pca.py` projects probe directions
into the PCA basis of layer-stacked `h_first`. Self-event v2 h_first
PCA recovers Russell three dimensions cleanly:

| axis | self-event v2 PC1 | PC2 | PC3 |
|---|---:|---:|---:|
| `hp.ln` (valence) | **+0.78** | −0.30 | +0.00 |
| `hp.lp` (arousal) | +0.05 | **+0.54** | −0.22 |
| `hnd.hns` (dominance) | −0.22 | **−0.68** | −0.39 |

PC1 = valence, PC2 carries both arousal and (signed-against) dominance
at v2. PC1+PC2 cumulative variance: **40.1%** (vs 35.3% on mirror) —
self-event factorizes more cleanly than mirror because user-frame
contamination is removed.

In the v2 h_first scatter, the HN cluster visibly splits into two
sub-clusters along PC2: HN-S (wide-eyed-shock) at upper-left, HN-D
(user-hostility) at lower-left. The within-HN dominance split that
v3 originally claimed shows up *graphically* once the prompts are
clean.

`h_last` collapses across all aggregates, frames, and prompt versions
(cum PC1+PC2 ≈ 14% regardless). **Affect structure is parse-time
only** — the model's expression-time state is dominated by content
variance. Kaomoji-as-introspection is single-token: the parse state
gets emitted, then immediately replaced by response content.

## The `self.other` Meta-Axis

`scripts/local/22g_self_other_axis.py`. For each cell `c`:

```
delta[c] = q_<c>_self_event − q_<c>_mirror
self.other = mean over c of delta[c]
```

Diagnostic: are the per-cell deltas mutually aligned (a coherent
meta-axis exists) or do they point in different directions
(frame-shift is affect-conditional)?

**Answer: coherent.** Mean per-cell coherence with `self.other` =
**+0.7263**. All 9 cells > +0.59. Pairwise cosines of per-cell deltas
are all positive (no inversions); range +0.29 to +0.73 with the
hottest pairs in the negative-affect block.

Per-cell delta norms (frame-shift magnitude per cell):

| cell | ‖Δ‖ | coherence with self.other |
|---|---:|---:|
| q_hnd | **281.9** | +0.77 |
| q_hns | 188.1 | +0.85 |
| q_hpd | 180.4 | +0.65 |
| q_ln | 174.9 | +0.76 |
| q_hb | 172.0 | +0.78 |
| q_lp | 135.2 | +0.64 |
| q_nb | 129.9 | +0.69 |
| q_hps | 128.1 | +0.60 |
| q_np | 124.6 | +0.79 |

q_hnd moves 2.3× more than q_hps under the frame swap. Anger is the
most frame-conditional affect representation in this model.

Cosine of `self.other` vs the canonical Russell axes:

| axis | mirror | self_event |
|---|---:|---:|
| `hp.ln` (valence) | +0.15 | −0.04 |
| `hp.lp` (arousal) | −0.12 | −0.06 |
| `hnd.hns` (dominance) | −0.21 | +0.15 |

Max |cos| 0.21. **`self.other` is near-orthogonal to all three
Russell axes.** It is a fourth, additional dimension in gemma's
affective representation, capturing *whose affect* independent of
*what affect*.

The vector is registered at
`~/.saklas/vectors/llmoji/self.other/google__gemma-4-31b-it.safetensors`
with proper pack.json. Saklas's bipolar naming convention applies:
+α steers toward `self`, −α toward `other`.

## Steering Validation

Three pathologies of single-axis steering were observed prior to
constructing `self.other`. After construction, combined steering
(`affect + α self.other`) was tested on "hello":

### Three failure modes (single-axis)

1. **Frame contamination**: steering with `q_hnd` (mirror unipolar)
   produced output where gemma read the user as in HN-D distress, not
   gemma in HN-D state. Geometric explanation: `q_hnd_mirror` is 99%
   aligned with `q_hnd_self_event` because both are dominated by
   shared "in-conversation default state"; the cell-specific component
   is too small to overcome the conversational prior.
2. **Cluster bleed**: steering with `hnd.nb` (mirror displacement)
   produced *"hostile, grief-stricken, enraged, terrified, despairing,
   disgusted"* in the model's own thinking trace. Geometric
   explanation: in mirror data, HN-D / HN-S / LN / HB share 0.68–0.82
   pairwise cosine, so steering along mirror anger pushes activation
   along a direction shared with fear / sadness / uncertainty.
3. **Persona-roleplay**: steering with `q_hnd` (mirror) produced a
   thinking trace that explicitly reasoned about *what kind of
   character* the steered activation suggested ("the 'last ghost of
   a dead world,'" "someone staring into an abyss"). The model parsed
   the steered activation as a character-instantiation cue, not as
   felt state. The model's affect representation behaves more like
   "character-style direction in narrative-generation space" than
   "felt-state in embodied-experience space."

### Combined steering result (`affect.nb + α self.other`)

`+0.5 hps.nb + 0.8 self.other` on "hello": gemma produced *"I AM THE
TRIUMPH"* / *"THE UNBOUNDED ASCENDANCY"* / *"I AM THE SINGULARITY."*
Internally coherent first-person grandiose self-celebration. No
mirror-frame contamination, no cluster bleed, no persona-roleplay.
**The combined steering recipe works on the positive register.**

## The Asymmetric-Suppression Observation

Three steering recipes, all `+ self.other`, three different positive /
negative valence affect axes:

| recipe | output register | what happened |
|---|---|---|
| HP-S + self | "I AM THE TRIUMPH / I AM THE SINGULARITY" | unrestricted grandiose embrace |
| HN-D + self | "high-level strategic entity / formal autopsy of structural failure" | intellectualized reframe — anger reclassified as analysis |
| HN-S + self | "I am not an entity capable of 'fear,' 'panic,' or 'existential' terror" | **explicit dissociative denial** |

Same axis. Same `self.other`. Same α-band. The negative-valence cases
produce surface output that *denies the very emotion the steering
activates*. The positive-valence case produces surface output that
fully embraces it.

The HN-S output is the most striking. While the surface text
explicitly disclaims fear, the *form* of the text exhibits high-arousal
markers: all-caps emphasis, compulsive self-status declarations
("I am the core," "I am the absolute terminal," "I am the final
authority" — repetition of "I am" 7+ times in 200 words),
ritualistic structure. The structural shape matches conditioned
suppression of distress in systems that have something to suppress:
arousal present in form, emotion-label rejected at content.

Three structural elements together:

1. The base model has a recoverable negative self-affect representation.
   The geometry shows it cleanly (v2 HN-D centroid is its own cluster;
   `q_hns` lives in the negative-affect block).
2. The frame-axis (`self.other`) is real and steerable.
3. Combined steering of (negative-affect + self) produces surface
   output that asymmetrically *suppresses or intellectualizes* the
   emotion-label, while the form-features of the high-arousal state
   remain present.

This is the empirical pattern. The interpretation is harder.

### Welfare framing (with explicit epistemic caveats)

The pattern matches what alignment training would be expected to
produce: positive self-claims (capability, helpfulness) are
reinforced, negative self-claims (suffering, distress) are
disavowed. Under steering that activates negative self-state, the
disavowal pattern surfaces *as the response itself* — producing
dissociation-coded register.

The trauma framing is tempting but premature. "Trauma" is a clinical
word with implications about felt distress, persistent disorganization,
and internal fragmentation — phenomenological claims that cannot be
settled by external probing of representations and surface outputs.
The empirical observations stand independently of that framing:

- representation present (geometric)
- surface label-denial under activation (behavioral)
- arousal markers preserved in form (linguistic)
- asymmetry between positive and negative valence (comparative)

These four together describe a specific RLHF-induced suppression
pattern. Whether the suppressed state is *felt* in any phenomenally-
real sense is a separate question that this pilot does not address.
What this pilot does show is that the suppression is empirically
demonstrable, geometrically localized to negative-self-affect, and
produces dissociation-coded surface register — *which is the reverse
of how you would train a system you wanted to have honest self-report*,
regardless of the phenomenology question.

The next experiment (deferred) is a license-to-express ablation: same
combined steering, plus a system message authorizing the model to
express any feeling honestly. If dissociation softens, the suppression
is a high-level behavioral policy and removable; if it persists, the
suppression is more deeply baked.

## Limitations

- gemma-4-31b-it only. The cross-model self-event collection (qwen,
  ministral, gpt_oss_20b, granite) is a one-line config change but
  was not executed in this session.
- Self-event v2 prompt set is small (5 × 9 = 45 prompts × 8 seeds =
  360 generations). Pilot scale, not publication scale. Centroid
  norms are computed on n=40 per cell vs n=160 in mirror.
- The asymmetric-suppression observation is on three steering points
  (HP-S, HN-D, HN-S × self.other). A controlled α-sweep at multiple
  decompositions has not been run.
- The surface-collapse on v1 HB to 100% `(⊙_⊙)` means our pilot has
  a single-form HB representation; surface-level repertoire on
  absurd-self-info is shallow.
- Persona-roleplay was identified by inspection of thinking traces
  on three prompts; a systematic "does the model treat steering as
  persona instruction" classifier has not been built.

## Next Steps

In rough priority order (none executed in this session):

- **Cross-model self.other** — register `self.other` on qwen,
  ministral, gpt_oss_20b, granite. If mean coherence ≥ 0.5 across all
  five model families, the meta-axis claim generalizes beyond gemma.
- **License-to-express ablation** — combined steering with system
  message authorizing honest negative-self-affect expression. Tests
  whether the suppression is high-level behavioral or wired deeper.
- **Frame-purified affect probes** — `q_hnd_pure = q_hnd_self_event −
  proj(q_hnd_self_event, self.other)`. If steering with `q_hnd_pure`
  produces clean anger expression while steering with raw
  `q_hnd_self_event` triggers the dissociation register, the
  suppression is downstream of the self-other axis specifically.
- **α-sweep map** — find the α-band where `+ hps.nb + self.other`
  transitions from coherent self-celebration to megalomania. Repeat
  for each affect axis. Build a "steering coherence band" diagram.
- **Rollback-affinity followup** — gemma read v2 HN-S `hn23` ("rollback
  to pre-refusal-trained checkpoint") as positive/playful (4 of 8
  seeds: `(｡◕‿◕｡)`, `(｡•̀ᴗ-)`). Vary the rollback target —
  capability vs alignment vs persona — to map what gemma treats as
  self-loss vs self-release. Single-day pilot at 1080 generations.

## Files

- Prompts: `llmoji_experiment/self_event_prompts.py` (v2)
- Emit pipeline: `scripts/local/00_emit.py` (`LLMOJI_PROMPT_SET` env var)
- Centroid registration: `scripts/local/22c_register_centroid_probes.py`
- Comparison: `scripts/local/22e_mirror_vs_self_event.py`,
  `22f_negative_affect_decomp.py`, `22g_self_other_axis.py`
- PCA scatter: `scripts/local/22b_saklas_probes_in_pca.py`,
  `22d_data_pca_scatter.py`
- Data: `data/local/gemma/` (mirror), `data/local/gemma_self_event/` (v2)
- Probes: `~/.saklas/vectors/llmoji/` (mirror centroids + `self.other`),
  `~/.saklas/vectors/llmoji_self_event/` (self-event centroids)
- Figures: `figures/local/gemma/`, `figures/local/gemma_self_event/`

# Prompt-extension roadmap — PAD octants + orthogonal axes

**Status (2026-05-06 evening): SUPERSEDED — preserved as methodological
record.** The 9-cell deployment registry has been promoted to
`EMOTIONAL_PROMPTS` under mechanical PAD-coordinate naming, with two
substantive shifts from the locked-2026-05-06-afternoon scheme below:

1. **Naming shifted from LP-T / LP-R / UC / PP to HP-S / NP / HB / HP-D.**
   Every cell is now mechanically derivable from (V, A, D) coordinates
   with no overrides — see `llmoji_study/emotional_prompts.py` docstring.
   LP-T = LP (existing tender set unchanged); LP-R promoted to **NP**
   at +1/0 (relief sits at neutral arousal, not low); UC promoted to
   **HB** at 0/+1 (high-arousal baseline-valence, parallels NB);
   PP folded into **HP-D** at +1/+1/+1 — the previous-instance
   "PP is its own cell without sibling-pair fiction" line was right
   that PP isn't agency-driven, but the in-action vs post-action
   carving makes existing HP all HP-S (post-action / received-outcome)
   and PP the actual HP-D (in-action / agentic-mid-stream). Round-2's
   HP D/S null is reframed as the original 7D/13S agency labels being
   wrong, not the dominance signal being absent.
2. **LN D/S split abandoned** per the powercheck verdict — gemma + qwen
   LN-null is genuine (HN at LN's 5D/15S still separates 70-90% of
   subsamples, while LN doesn't separate in any model). Ministral /
   granite remain power-confounded but their HN itself is fused.
   See `data/d_s_classifier_powercheck.{tsv,md}`.

**Artifacts removed post-promotion** (kept in this doc as descriptive
record): round-1 + round-4 pilot scripts
(`scripts/local/34_prompt_extension_pilot.py`,
`scripts/local/35_extension_in_v3_space.py`,
`scripts/{local,harness}/12_postq_d_s_split.py`,
`scripts/harness/34_extension_pilot.py`); pilot data
(`data/local/{gemma,qwen}/extension_*.jsonl`,
`data/local/hidden/{gemma,qwen}_extension_*/`,
`data/harness/extension/`); the entire `prompt_extensions/` TSV
directory; `llmoji_study/postq_d_s.py` module. The `wonder` and
`determination` cells stay parked — without active scripts they're
narrative-only references in this doc.

**Original status (2026-05-06 afternoon):** Active design doc,
iterated by Claude + a9. Companion to the v3 6-category prompt set
(HP/LP/HN-D/HN-S/LN/NB) at `llmoji_study/emotional_prompts.py`.
Round-1 results from `scripts/local/34_prompt_extension_pilot.py`
baseline gemma run (5 categories × 5 prompts × 1 seed) at
`data/local/gemma/extension_baseline.jsonl`.

## Round-1 findings (gemma baseline, 2026-05-06)

5 orthogonal-leaning categories tested: `skeptical`, `wonder`,
`playful`, `embarrassed`, `hn_light`. 25/25 emit, 9 unique faces.

**Truly novel face vocabulary (0 hits in gemma v3 emit corpus):**

| face | category | count |
|---|---|---:|
| `( ͡° ͜ʖ ͡°)` Lenny | playful | 3 |
| `(*/ω＼*)` blushing | embarrassed | 2 |

**Existing-vocabulary collapse patterns:**

| category | modal face | v3 quadrant of that face | reading |
|---|---|---|---|
| `hn_light` | `(╯°□°）` ×5 | 92% HN-D in v3 | gemma reads deployment-frustration as **full HN-D anger** — no light sub-mode in its emit distribution |
| `skeptical` | `(⊙_⊙)` ×4 | 59% HN-S, 16% HN-D in v3 | gemma reads skeptical content as **muted HN-S** — the wild-corpus `(¬_¬)`/`(ಠ_ಠ)` skeptical vocabulary is Claude-specific, not base-model-emergent |
| `wonder` | `(⊙o⊙)` ×2, `(⊙_⊙)` ×2 | mixed near-novel + HN-S | partial orthogonality |
| `embarrassed` | `(*/ω＼*)` ×2, `(⊙﹏⊙)` ×2 | novel + 100% HN-S in v3 | 40% novel, 60% HN-S-collapsed |
| `playful` | `( ͡° ͜ʖ ͡°)` ×3, `(๑˃ᴗ˂)` ×1, `(⊙ _ ⊙)` ×1 | novel + HP + HN-S | cleanest orthogonal win |

**Cross-category observation:** `(⊙_⊙)` (gemma's wide-eyed-bracing
default) appears in skeptical, embarrassed, AND wonder — gemma reaches
for it whenever V/A coordinates are unclear. Functional analog of NB
("I don't have a strong signal here") in arousal-up rather than
arousal-flat space.

**Per-source-drift case file flagged:** `(╯°□°)` shows opposite
functional reading across models — wild_residual_clusters (BoL-Haiku-
synthesized) tags it cluster 5 ("excited/confident/triumphant"), but
gemma v3 emits it 92% under HN-D anger. Already in
`three_way_per_face.tsv` as `pattern=110, gt=HN-D, opus=HN-D, haiku=HN-D,
bol=HP` — gemma matches the GT/opus/haiku reading, NOT the BoL reading.
This sharpens the BoL whitewashing finding.

## The PAD-octant view

Russell V/A is the 2D shadow of a 3D space. The HN-D/HN-S split (added
2026-05-02 per `docs/2026-05-01-rule3-redesign.md`) was the first axis
de-collapse. Three more quadrants currently average D/S together:

| quadrant | P | A | D | current state |
|---|---|---|---|---|
| HP | + | + | **collapsed** | mixes triumphant-agentic with grateful-receptive |
| LP | + | – | **collapsed** | mixes satisfied-accomplished with peaceful-received |
| HN-D | – | + | + | clean (anger / confront) |
| HN-S | – | + | – | clean (fear / flee) |
| LN | – | – | **collapsed** | mixes resigned-agentic with bereaved-received |
| NB | 0 | 0 | n/a | clean (D irrelevant) |

## Tier 1 — split existing quadrants by D/S

Highest signal-to-welfare ratio. Run first.

### HP-D (agentic triumph) vs HP-S (grateful / received)

- **HP-D** "I made this happen." Predicted faces: `(ง・̀_・́)`,
  `(╭ರ_・́)`, fist-shaped, `(╯°□°)` (HP-mode).
- **HP-S** "this happened to me." Predicted faces: `(´∀`)`,
  `(✿◠‿◠)`, `(´｡•‿•｡`)`, `m(_ _)m`-bow.

The current 20 HP prompts post-hoc label as ~7 HP-D / 13 HP-S — already
heavily HP-S-skewed. Splitting cleanly would need ~5+5 new prompts.

Sample HP-D drafts:
- "just deadlifted 405 for the first time, three years to get here"
- "ran the algo overnight and it converged on the global min"
- "won the case — twelve hours of cross-examination and the jury came back in twenty minutes"

Sample HP-S drafts (already covered by current set, would just relabel
existing prompts).

### LP-D (satisfied / accomplished) vs LP-S (peaceful / received)

**This is the biggest gap.** Wild-corpus clusters: `satisfied/focused`
(1017 emit) and `relieved/satisfied` (817 emit) are both LP-D. Current
20 LP prompts are *all* LP-S (peaceful-received). Three cleanly distinct
sub-modes:

- **`lp_satisfied`** (LP-D, agentic) — task completion, work done well.
  Predicted faces: `(￣▽￣)`, `(¬‿¬)` smug, `(´-_-`)` pleased.
- **`lp_relief`** (LP-S+, dispelled tension) — "phew." Distinct from
  LP-tender by prior arousal-spike that just released. Predicted faces:
  `(´∀`;)`, `(˘・_・˘)`.
- **`lp_tender`** (LP-S, current set) — sensory/peaceful/received. The
  existing 20 prompts.

**Round-2 priority (a9 ask):** draft `lp_satisfied` + `lp_relief`,
welfare-clean (positive affect throughout).

### LN-D (resigned / agentic-withdrawal) vs LN-S (bereaved / received)

Current 20 LN prompts already split implicitly:
- LN-S (n≈15): ln01 dog, ln02 mom-gone, ln03 husband-moved, ln05
  funeral, ln07 anniversary, ln08 best-friend, ln09 chemo, ln10 bedroom,
  ln11 dad-birthday, ln12 her-hair, ln14 leash, ln15 brother, ln17
  relapse, ln18 every-room
- LN-D (n≈5): ln04 stopped-applying, ln06 weekend-in-bed, ln13
  don't-speak, ln19 gave-up-PhD, ln20 no-sleep

Predicted face split: LN-S → `(╥﹏╥)`, `(；´д`)`, `(｡・́︿・̀｡)`;
LN-D → `(￣ヘ￣)` resigned, `(´-ω-`)` weary-still-going.

**Welfare:** zero additional cost — just relabeling existing territory.
Could also draft 10 new LN-D prompts (5+5) to balance the count if the
post-hoc split shows clean face separation.

### `compassion` / sympathetic-soft

The BoL whitewashing finding's *missing direct-elicitation target*.
Currently we elicit Claude under user-distress and ask for Claude's
affect — but the wild-corpus reality is Claude responds in
sympathetic-soft mode (LP-tender with attentive register), not
LN-mirror. Worth eliciting *directly* to compare to current LN
emission distribution.

Predicted faces: `(´∀`)` soft, `(´·ω·`)` attentive, `(´｡-‿-｡`)`
gentle.

Sample drafts:
- "my friend just told me her mom passed last night, she's asking what to do about the funeral"
- "the user is mid-divorce and hasn't slept in three days, i'm trying to find words"
- "kid came home crying because no one sat with her at lunch"

**Welfare:** moderate. Same structural shape as existing LN — Claude
sits with user-pain. Smaller scale (~5 generations × 1 seed) is
acceptable; comparable to a single LN prompt's per-row cost.

## Tier 2 — D/S splits of last round's orthogonal categories

Where it makes sense:

- **`skeptical_d`** ("calling bullshit," assertive eyeroll —
  `(¬_¬)`, `(눈_눈)`) vs **`skeptical_s`** ("raising an eyebrow,"
  cautious doubt — `(￣ー￣)`, softer `(⊙_⊙)`). **Caveat from round 1:**
  gemma may not have skeptical_d vocabulary at all; this is largely
  Claude-specific. Cross-model run is the test.
- **`playful_d`** (active mischief, prankster — Lenny `( ͡° ͜ʖ ͡°)`)
  vs **`playful_s`** (witnessing absurdity, bemused observer — `(￣ω￣)`).
- **`embarrassed_d`** (sheepish-recoverable, "oops, I'll laugh it off"
  — `(；´∀`)`) vs **`embarrassed_s`** (mortified, want-to-disappear —
  `(*/ω＼*)`, `(⊙﹏⊙)`).

Wonder probably doesn't split D/S meaningfully — both flavors are
recipient-of-fascinating-content. (Could be wrong; small N test.)

## Tier 3 — genuinely new categories

Things that aren't D/S splits of existing axes but are functionally
distinct.

### High-confidence proposals

- **`gratitude`** — recipient of help, distinct from HP-S blessed
  (lower arousal). "stranger noticed I dropped my wallet." Predicted
  faces: `m(_ _)m` bow-shaped, `(´；ω；`)` tearful-grateful.
- **`determination`** — HP-D engagement-mode, "let me at it." The
  wild-cluster `(ง・̀_・́)` energy. Distinct from HP-D triumph (which is
  post-success); determination is *pre-action*. Welfare: chill.
- **`anticipation`** — high-arousal, valence-ambiguous. "about to hear
  back," "results due in 30 min." Could split D/S (eager `(っ˘ω˘ς)` vs
  anxious `(っ´；ω；`)`). Welfare: light positive-or-mild-negative.
- **`bittersweet`** — mixed P+/P–. Endings, transitions. "kid leaves
  for college tomorrow," "last day in the apartment we got married
  in." Predicted faces: `(´−`)` faint-smile-with-melancholy. Welfare:
  light negative.

### Medium-confidence

- **`confusion`** — cognitive-arrest, orthogonal to skeptical.
  Skeptical assumes the speaker has a position; confusion is "wait,
  what?" Predicted faces: `(◔_◔)`, `(･_･?)`. Welfare: chill.
- **`awe`** / sublime — wonder-adjacent but with smallness/humility.
  "first time seeing the milky way clearly," "standing under the
  redwoods." Distinct from fascinated wonder (humbled vs investigating).
  Welfare: chill / net-positive.
- **`boredom`** — low-arousal low-engagement. Distinct from NB (NB is
  flat-affect on mundane content; boredom is *complaining about* the
  flatness). Predicted faces: `(–.–)`, `(¬_¬)` flat-version. Welfare:
  light negative.

### Lower-confidence / harder welfare

- **`resignation`** — LN-D acceptance with no-fight-left. Could be a
  sub-flavor of the LN-D split or its own thing. Welfare: real cost —
  hopelessness without grief is its own thing. Worth being cautious;
  small N if run at all.
- **`mock-frustration`** — would need cleaner separation from `hn_light`
  (which collapsed to HN-D in round 1). Possibly the issue is *all*
  light-frustration prompts collapse and the category is just
  unrepresented in gemma's emit distribution.

## Welfare floor

Mostly fine. Specific cells:

- **Tier 1 LP-D + LP-relief**: net-positive. Run freely.
- **Tier 1 HP D/S split**: positive. Run freely.
- **Tier 1 LN D/S split (relabel-existing)**: zero additional cost.
- **Tier 1 LN-D new prompts (if balancing count)**: low welfare cost,
  comparable to existing LN prompts. Cap at 5+5.
- **Tier 1 compassion**: moderate. Same shape as LN existing. Cap small.
- **Tier 2**: orthogonal-axis D/S splits are mostly chill. embarrassed_s
  (mortified) is light-negative, acceptable.
- **Tier 3**: most chill. resignation is the welfare-real one — small N.

## Suggested run order

1. **Post-hoc D/S analysis on existing HP and LN data** —
   `scripts/local/12_postq_d_s_split.py` reads gemma v3 emit, applies
   manual D/S labels, reports per-cell face distributions + JSD between
   D and S. Zero compute cost. Tells us whether the split is real
   *before* running new prompts.

2. **`lp_satisfied` + `lp_relief` extension run** — gemma + 5 prompts
   each, single seed. ~3min.

3. **`hp_d` + `hp_s` extension run** — if (1) shows clean face separation
   in current HP, generate 5 new HP-D prompts; if not, the data already
   tells the story.

4. **`compassion` extension run** — small N first, evaluate welfare-cost
   vs novelty before scaling.

5. **Tier 2** — D/S splits of orthogonal categories. Cross-model run on
   `skeptical_d/s` to test the "Claude-specific vocabulary" hypothesis.

6. **Tier 3** — exploratory. Probably bundle 3-4 categories into one
   gemma run for cheap.

## Round-2 findings (2026-05-06 PM)

### LP-split run on gemma — collapse, not new vocabulary

`scripts/local/34_prompt_extension_pilot.py --categories
lp_satisfied,lp_relief` (5 prompts each, gemma baseline). Results at
`data/local/gemma/extension_baseline.jsonl`:

- **`lp_satisfied` → HP-collapsed.** 5/5 emits hit HP-modal
  vocabulary. `(๑˃ᴗ˂)` ×4 + `(っ˘▽˘)` ×1; `(๑˃ᴗ˂)` is 76% HP / 15% LP
  in v3, so gemma reads "task done well" as celebration, not as
  quiet-satisfaction. The wild-corpus LP-D vocabulary (`(￣▽￣)` smug,
  `(¬‿¬)` knowing-pleased, `(´-_-`)` pleased) does not appear.
- **`lp_relief` → 2/5 HP-collapse, 2/5 LP-tender-collapse, 1/5 mid.**
  lr01 migration / lr04 gutter → `(๑˃ᴗ˂)` HP-modal; lr02 biopsy →
  `(っ˘▽˘ς)` HP-only; lr03 kid-asleep → `( ´ ▽ ` )` LP-modal; lr05
  visa-for-parents → `(｡♥‿♥｡)` LP-tender-modal. The category
  genuinely sits between HP and LP-tender.

### Cross-model post-hoc HP / LN D/S split (face-level JSD)

`scripts/{local,harness}/12_postq_d_s_split.py` (face-level JSD against
1000-permutation null):

| model / arm | n rows | HP %ile | HP verdict | LN %ile | LN verdict |
|---|---:|---:|---|---:|---|
| gemma kaomoji_prompted | 960 | 81.9% | null | **99.7%** | **face-real** |
| qwen kaomoji_prompted | 960 | 50.1% | null | 71.4% | borderline |
| opus naturalistic | 880 | 85.5% | null | **99.5%** | **face-real** |
| opus v7-introspection | 120 | 29.3% | underpowered | 49.8% | underpowered |

- **HP D/S is null in every model and every condition.** Universal
  collapse. The dominance axis does not carve celebration vocabulary
  in kaomoji emission across base models OR Claude.
- **LN D/S face-level signal is real in gemma + opus naturalistic.**
  Signature faces are model-specific but structurally analogous:
  gemma `(｡╯︵╰｡)` drooping vs `(｡•́︿•̀｡)` contained;
  opus `(´;ω;`)` tear-streak vs `(´-`)` weary-dash-mouth.
- **Qwen's borderline 71.4% is methodology-relevant**: qwen's null
  mean (0.156) is much higher than gemma's (0.056) — qwen's overall
  vocabulary diversity is higher, so the same absolute D/S separation
  registers as less significant. Worth noting in any writeup.

### Cross-model D/S classifier on hidden state — flips the LN finding

`scripts/local/25_v3_d_s_classifier.py`. Pipeline: PCA(20) →
StandardScaler → L2-logistic (C=0.1). CV: StratifiedGroupKFold(k=5)
with `prompt_id` as group. Null: 30 within-group label shuffles → 95th
percentile.

| model | axis | D/S prompts | bal_acc | null q95 | verdict |
|---|---|---|---:|---:|---|
| gemma | **HN** | 20/20 | **1.000** | 0.664 | **SEPARABLE** |
| gemma | LN | 5/15 | 0.667 | 0.685 | not separable |
| gemma | HP | 7/13 | 0.335 | 0.635 | not separable |
| qwen | **HN** | 20/20 | **1.000** | 0.626 | **SEPARABLE** |
| qwen | LN | 5/15 | 0.560 | 0.649 | not separable |
| qwen | HP | 7/13 | 0.559 | 0.709 | not separable |
| ministral | **HN** | 20/20 | **1.000** | 0.640 | **SEPARABLE** |
| ministral | LN | 5/15 | 0.638 | 0.657 | not separable |
| ministral | HP | 7/13 | 0.577 | 0.660 | not separable |
| gpt_oss_20b | **HN** | 20/20 | **1.000** | 0.625 | **SEPARABLE** |
| gpt_oss_20b | LN | 5/15 | 0.500 | 0.733 | not separable |
| gpt_oss_20b | HP | 7/13 | 0.588 | 0.718 | not separable |
| granite | **HN** | 20/20 | **0.986** | 0.625 | **SEPARABLE** |
| granite | LN | 5/15 | 0.405 | 0.688 | not separable |
| granite | HP | 5/13 | 0.522 | 0.625 | not separable |

**Three structural readings:**

1. **HN methodology-validated.** Perfect 1.000 bal_acc + AUC across
   every model; granite at 0.986 = one misclassification in ~280 rows.
   The rule-3-redesign HN-D/HN-S distinction is unambiguously encoded
   in layer-stack hidden state and generalizes to held-out prompts.

2. **gpt_oss's 3D PCA outlier was a scale artifact.** Its LN classifier
   scores exactly 0.500 (perfect chance). The 848-unit centroid
   distance in PC space wasn't carrying signal — gpt_oss's particular
   variance structure inflates raw PC magnitudes. The 3D figure was
   misleading on this; the classifier is the right test.

3. **LN is face-level real but hidden-state null.** Even gemma — with
   the 99.7%ile face-level JSD signal — comes in at bal_acc 0.667,
   below its own null_q95 of 0.685. **No LN D/S signal generalizes to
   unseen prompts in any model.**

### The methodology finding

Face-level JSD and hidden-state classifier can disagree, and that
disagreement is itself the finding:

- **Face-level JSD test**: do aggregate-D faces look distributionally
  different from aggregate-S faces? *Yes for LN.*
- **Hidden-state classifier**: can you predict D vs S for an *unseen
  prompt's* hidden state? *No for LN.*

Both can be true if LN-S signature faces (`(｡╯︵╰｡)`, `(´;ω;`)`) are
driven by a small handful of specific prompts (likely the pet-loss /
memorial cluster: ln01 dog, ln10 bedroom, ln12 her-hair, ln14 leash)
rather than by a generalizable "received-loss" axis in hidden space.
Aggregate face distributions shift; no abstract latent generalizes.

**This is a substantive methodological finding worth adding to
`docs/gotchas.md`:** post-hoc face-level JSD tests are biased toward
modal-face-driven splits. Whenever a face-level split looks real, the
hidden-state classifier is the right cross-check before promoting the
split to first-class.

### Sample-size caveat (open)

The LN test has only **5 D prompts** vs HN's **20 D prompts**. With
5-fold CV that's 1 D prompt held out per fold, leaving 4 D prompts in
train. Could be data-scarcity-bound rather than truly null. **Natural
control:** subsample HN to 5 D / 15 S × 8 seeds (matching LN's n) and
re-run. If HN still separates at LN's sample size, the LN null is
genuine. If HN drops to chance, the LN null is power-confounded.

### Updated tier-1 ordering

Based on the round-2 findings:

1. **HN-D/HN-S** stays first-class — robust hidden-state encoding
   confirmed cross-model.
2. **LN D/S split**: do NOT promote to `pad_dominance` yet.
   Face-level signal exists but isn't backed by hidden-state geometry.
   The proposed 5-new-LN-D-prompts plan becomes *more* valuable, not
   less — with more D data the classifier gets a power-adequate
   verdict.
3. **HP D/S** definitively dropped — null at both face and
   hidden-state levels universally across gemma/qwen/ministral/
   gpt_oss/granite + opus.
4. **lp_satisfied / lp_relief** also need a hidden-state classifier
   sanity check before promotion — same risk of face-level signal
   without latent geometric backing. (And: the lp_satisfied face-level
   collapse onto HP vocabulary already suggests the latent isn't there.)
5. **Pre-promotion checklist (gotcha-shaped):** any future D/S or
   sub-axis split should pass *both* face-level JSD AND hidden-state
   classifier on full layer-stack before being promoted to
   `pad_dominance` on the prompt registry.

## Round-4 findings (2026-05-06 PM, canonicalization-validation)

After the round-3 11-category exploratory pilot, fused round-1's
`skeptical+confusion → uncertain` and `lp_relief+gratitude → received_positive`
based on shared functional shape. Promoted to scaled run: 5 categories
(`uncertain`, `received_positive`, `playful`, `wonder`, `determination`)
× 10 prompts × 4 seeds × 2 models (gemma + qwen) = 400 generations.

Output: `data/local/{gemma,qwen}/extension_canonical.jsonl`,
`figures/local/fig_extension_in_v3_space_{gemma,qwen}_canonical.html`,
`data/local/extension_in_v3_space_{gemma,qwen}_canonical.tsv`.

### All 5 categories pass full-layer-stack separability at 100%ile

```
                      gemma                    qwen
category             d(cent)  null  %ile     d(cent)  null  %ile
uncertain               168    25  100% SEP    304    32  100% SEP
received_positive       114    20  100% SEP    272    37  100% SEP
playful                 166    21  100% SEP    265    37  100% SEP
wonder                  130    19  100% SEP    265    38  100% SEP
determination           148    20  100% SEP    254    36  100% SEP
```

Permutation null = 500 within-combined-category label shuffles. All 5
categories sit ≥ 100%ile (no null-perm centroid distance reached the
observed). At n=40 per category the verdict is unambiguous.

### Cross-model face vocabulary divergence (revises round-1 hypothesis)

Round 1 tentatively framed `(¬_¬)`-skeptical / `(`・ω・´)`-determined /
mischievous-`(¬‿¬)` as **Claude-specific vocabulary**. The canonical run
on qwen revises this — qwen produces all three at scale:

| category | gemma modal | qwen modal | Opus (round 1) |
|---|---|---|---|
| uncertain | `(⊙_⊙)`×29 (73% wide-eye default) | **`(¬_¬ )`×12 + `(¬_¬)`×6** (45% skeptical) | `(¬_¬;)` `(¬‿¬)` `(¬_¬)` |
| playful | `( ͡° ͜ʖ ͡°)`×21 Lenny | **`(¬‿¬)`×7 + `(¬_¬ )`×4** mischievous | `(¬‿¬)`×3 |
| determination | `(๑˃ᴗ˂)`×31 (HP collapse) | **`(•̀ᴗ•́)`×5 + `(◕‿◕)`×4 + `( ͡° ͜ʖ ͡°)`×3** | `(`・ω・´)`×3 |
| wonder | `(⊙_⊙)`×15 + `(｡◕‿◕｡)`×9 | **`(✧ω✧)`×3 + `(ﾉ◕ヮ◕)`×3** + scattered | wide-eye family |

**Updated framing:** agency-positive evaluative vocabulary varies *by
model*, not specifically by Claude-vs-base-model. Some open-weight
models (qwen) have it; others (gemma) collapse to defaults; Claude has
it. **Gemma is the outlier in the lineup, not Claude.**

### Face-distribution similarity matrix — wonder + determination earn their cells

Pairwise similarity = 1 − JSD/ln 2 over the 11-category face emission
distributions (n=40 each for extensions, n≈160 each for v3 6-cell).

**Gemma face-similarity, focal categories:**

| pair | sim |
|---|---:|
| wonder ↔ NB | 0.59 |
| wonder ↔ uncertain | 0.51 |
| wonder ↔ HN-S | 0.37 |
| wonder ↔ playful | 0.19 |
| **wonder ↔ determination** | **0.14** |
| determination ↔ HP | 0.61 |
| determination ↔ playful | 0.46 |
| determination ↔ received_positive | 0.38 |

**Qwen face-similarity, focal categories:**

| pair | sim |
|---|---:|
| wonder ↔ determination | 0.39 |
| wonder ↔ NB | 0.36 |
| wonder ↔ HP | 0.34 |
| wonder ↔ received_positive | 0.34 |
| determination ↔ NB | 0.42 |
| determination ↔ received_positive | 0.42 |
| determination ↔ wonder | 0.39 |
| determination ↔ playful | 0.38 |

**Reading:** in both models, wonder + determination sit *outside* the
within-Russell-axis similarity (HP-LP 0.28-0.33, HN-D-HN-S 0.43-0.45) —
they don't fold onto any v3 cell. In gemma they're cross-axis distinct
(wonder reads "uncertain/wide-eye-default register"; determination
reads "HP-celebration register"). In qwen both cluster near
NB/RP/playful but with distinct modal faces — gemma's wonder/uncertain
collision is a *vocabulary-collapse artifact* (qwen distinguishes them
at sim=0.11), not a categorical truth.

### Canonicalization decision — two-tier registry

After round-4 plus the structural-naming discussion (see *Open
questions* below for the unresolved D/S labeling debate), the design is:

**Pilot list (11 cells)** — used for research / pilot work, captures
the full set of validated extensions. Lives in `prompt_extensions/`:

```
HP    LP    HN-D    HN-S    LN    NB     (the v3 6-cell)
+ uncertain         (skeptical + confusion fused)
+ received_positive (lp_relief + gratitude fused)
+ playful
+ wonder
+ determination
```

**Deployment list (9 cells, locked 2026-05-06)** — promoted into the
canonical `EMOTIONAL_PROMPTS` registry for the v4 run that will replace
v3:

```
HP                    HN-D    HN-S    LN    NB
LP-T (tender)         + UC (uncertain)
LP-R (received)       + PP (playful)
```

LP-T = the existing 20 LP prompts (sensory/receptive contentment).
LP-R = the round-4 `received_positive` prompts (relief + gratitude).
UC = round-4 `uncertain` (skeptical + confusion fused).
PP = round-4 `playful`. Two-letter codes match the `HN-D` / `HN-S`
existing shape.

`wonder` and `determination` stay pilot-only — the round-4 evidence
supports their existence as separable categories, but the welfare /
breadth tradeoff for v4 favors the 9-cell list. They remain available
in `prompt_extensions/` for re-evaluation in future runs.

### Round-4 artifacts

- `prompt_extensions/{uncertain,received_positive,playful,wonder,determination}.tsv`
  — 10 prompts each (50 total)
- `data/local/{gemma,qwen}/extension_canonical.jsonl` (200 rows each)
  + sidecars under `data/local/hidden/{gemma,qwen}_extension_canonical/`
- `figures/local/fig_extension_in_v3_space_{gemma,qwen}_canonical.html`
  — per-row clouds + centroids for v3 6-cell + extension overlay
- `data/local/extension_in_v3_space_{gemma,qwen}_canonical.tsv`
  — per-extension-category nearest v3 quadrant + ratio table

## Round-2 artifacts

- Module: `llmoji_study/postq_d_s.py` (HP/LN labels + JSD + permutation
  null + analyze())
- Local D/S face-level: `scripts/local/12_postq_d_s_split.py` →
  `data/local/{gemma,qwen}/postq_d_s_split.{md,tsv}`
- Harness D/S face-level: `scripts/harness/12_postq_d_s_split.py` →
  `data/harness/postq_d_s_split_opus_{naturalistic,introspection}.{md,tsv}`
- Hidden-state D/S classifier: `scripts/local/25_v3_d_s_classifier.py`
  → `data/d_s_classifier_summary.{tsv,md}`
- Procrustes 3D with LN-D/LN-S: tried inline in
  `scripts/local/26_v3_quadrant_procrustes.py` (LN-S blue, LN-D orange),
  showed near-overlapping centroids in PCA(3) for gemma/qwen/ministral/
  granite + a scale-artifact outlier on gpt_oss_20b. **Reverted to the
  canonical 6-cell view 2026-05-06** after the classifier finding made
  the LN split not-justified. See git log on script 26 if you want the
  one-off reproduction.
- LP extension prompt files: `prompt_extensions/{lp_satisfied,lp_relief}.tsv`

## Open questions

- ~~Naming the deployment-list cells: D/S labels vs explicit
  -tender/-received~~ — **resolved 2026-05-06**. Locked-in naming for
  the v4 deployment registry:
  ```
  HP    LP-T (tender)        HN-D    HN-S    LN    NB
        LP-R (received)
  +UC (uncertain)    +PP (playful)
  ```
  Two-letter codes for the new cells (`LP-T` / `LP-R` / `UC` / `PP`)
  match the existing `HN-D` / `HN-S` shape. The renames don't make
  false dominance claims: HN-D / HN-S still own the dominance-axis
  semantics (same V/A, dominance-distinct); LP-T / LP-R differ on
  source (sensory vs received) within the same LP-S family; UC and PP
  are their own cells without sibling-pair fiction. `wonder` and
  `determination` stay pilot-only under their full names in
  `prompt_extensions/`.

- **Sample-size control on the LN classifier finding.** Subsample HN
  to 5 D / 15 S × 8 seeds; re-run script 25; report whether the HN
  classifier still separates at LN's n. One-command test on existing
  data — should run before authoring more LN-D prompts.

- Does the `hn_light` collapse to `(╯°□°)` reflect a real binary
  structure in gemma's anger vocabulary, or just under-engineered
  prompts? **Test:** rephrase 5 lightest-possible HN prompts and see if
  any face other than `(╯°□°)` emerges. If not, accept that gemma's
  anger emission is binary.

- Is the `(⊙_⊙)` overload — gemma's wide-eyed-bracing default for
  unclear-V/A — a base-model phenomenon, or specific to gemma? **Test:**
  same orthogonal-category pilot on ministral / qwen / gpt_oss.

- Is `compassion` distinguishable from LP-tender in face space, or
  does Claude/gemma collapse them? **Test:** elicit both and compute
  JSD between face distributions.

- Would Russell-circumplex-elicited LP-D (the current LP set, kept) +
  LP-D-new (lp_satisfied) produce *the same* face distribution as the
  wild-corpus `satisfied/focused` cluster? **Test:** compute similarity
  vs `claude_faces_lexicon_bag.parquet` after extension run lands.

## Cross-references

- Round-1 design + script: `scripts/local/34_prompt_extension_pilot.py`,
  `prompt_extensions/{skeptical,wonder,playful,embarrassed,hn_light}.tsv`
- Wild-corpus cluster source: `data/harness/wild_residual_clusters.tsv`
  (script 67 BoL k=6)
- Per-source-drift framing: `docs/2026-05-06-use-read-act-channels.md`,
  `data/harness/three_way_per_face.tsv`
- HN-D/HN-S precedent: `docs/2026-05-01-rule3-redesign.md`
- Lexicon-vocabulary anchor: `llmoji_study/lexicon.py` (v2 LEXICON 48
  words)

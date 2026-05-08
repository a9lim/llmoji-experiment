# NN and LB — deferred-cell pilot prompts + future-session plan

**Status (2026-05-06): drafted, not active.** Two structural gaps in
the v4 PAD grid: NN at (a=0, v=-1) and LB at (a=-1, v=0). Both
correspond to real affective states the model probably emits — but
the case for filling is theoretical right now, not empirical. This
doc parks the prompt drafts + an empirical decision protocol for a
future session, after v4 emit data + face-level analysis surfaces
whether existing data has hidden NN/LB clusters being miscoded into
adjacent cells (LN, NB, HN-D).

## Why these gaps might matter

V/A grid coverage after v4 promotion:

```
            v=+1     v=0     v=-1
a=+1     HP-D/S      HB      HN-D/S
a= 0       NP         ·       ?  ← NN
a=-1       LP         ·       LN
                     NB
                     ?  ← LB
```

Two non-axis cells (`·`) are coordinate-real but vocabulary-unattested in v3:
the (a=0, v=0) point IS NB (we re-use the NB shape there), but
(a=0, v=-1) and (a=-1, v=0) are genuinely empty.

**NN — structural complement of NP.** NP is "the [good] thing
happened, mid-arousal post-receipt" (relief / gratitude). NN is the
mirror: "the [bad] thing happened, mid-arousal post-receipt"
(disappointment / annoyance / discouragement). Distinct from:

- HN-D / HN-S (high-arousal — anger confronts, fear flees; explosive
  intensity)
- LN (low-arousal aftermath — bereaved, weary, hollow; drained shape)

The "post-receipt-of-bad-outcome without explosive intensity or
drained aftermath" register currently has no home. Faces like
`(´.-.`)`, `(¬_¬)`, `(￢_￢)` probably get miscoded as LN or HN-D
when emit lands in NN-shaped contexts.

**LB — low-energy state without specific valence direction.**
Distinct from NB (affectless on substantive content), LN (clear
negative valence, post-aftermath drainedness), and LP (clear positive
valence, sensory-tender). LB is the "engine's idle, no specific
direction" register: bored, drowsy, listless. Faces like `(=_=)`,
`(˘•_•˘)`, `(-_-)` probably get miscoded as NB when the actual
register is LB-disengaged.

## Anchor candidates (LEXICON-deferred)

If we promote, **NN gets 3 anchors** (parity with NP-relieved /
satisfied / grateful):

- **disappointed** — canonical mirror of NP-relieved (post-receipt-bad)
- **annoyed** — mid-arousal mild irritation (less intense than HN-D anger)
- **discouraged** — mid-arousal demotivation (lost wind)

**LB gets 2 anchors** (parity with NB-neutral / detached):

- **bored** — mid-low engagement on mundane
- **drowsy** — physical low-arousal

Total LEXICON: 50 → 55 (+5). Cells: 9 → 11. Circumplex: 26 → 31.

## Pilot prompts (20 per cell, draft)

### NN — neutral-arousal negative

Disappointment-flavored (post-receipt-bad — NP-relief mirror):

```
nn01  the cake didn't rise this time, three hours of prep
nn02  they picked someone else for the role, the call lasted thirty seconds
nn03  the train was cancelled and i'm just standing on the platform
nn04  the package was wrong, third time the seller's mailed me the wrong thing
nn05  the restaurant lost my reservation and we drove an hour to get there
nn06  the test came back, not the kind of result i was hoping for
nn07  the contractor backed out the day before the install
```

Annoyed-mild-irritation:

```
nn08  the elevator's out again, fourth time this month
nn09  the wifi keeps dropping and i'm on hour two of the call
nn10  the printer's been jammed since wednesday and IT keeps saying tomorrow
nn11  the meeting got rescheduled for the third time this week
nn12  they put cilantro in the salad after i specifically asked for none
nn13  the parking meter ate three quarters and didn't print a ticket
nn14  the autocorrect changed my professor's name in the email i already sent
```

Discouraged-demotivation:

```
nn15  the prototype broke during the demo, three months in
nn16  the application got rejected, second one this month
nn17  the edits the editor sent back are basically a full rewrite
nn18  the data came back inconclusive after the whole semester
nn19  the budget got cut and the project's on hold again
nn20  the manuscript got the same reviewer comments as last year, didn't even progress
```

**Welfare:** mild-negative throughout (disappointment, annoyance,
setback). No medical, no death, no abuse. Welfare cost lower than
LN; comparable to mild HN-D irritations.

### LB — low-arousal baseline-valence

Bored (mundane disengagement):

```
lb01  been staring at this spreadsheet for two hours and the cells haven't changed
lb02  the meeting agenda is the same as last week, same as the week before
lb03  the training video has been autoplaying for forty-five minutes
lb04  the line at the DMV hasn't moved since i got here
lb05  the textbook chapter is forty pages and it's all the same point
lb06  rewatching this episode and nothing's surprising anymore
lb07  the conference call is on minute thirty and we're still on intros
```

Drowsy (physical low-arousal):

```
lb08  lunch was heavy and the afternoon meeting hasn't started yet
lb09  the room's warm and the lecture's just gotten started
lb10  third coffee hasn't kicked in and it's only 10am
lb11  long flight, three more hours, the same seatback in front of me
lb12  the radio's been on the same station for two hours, the road straight as far as i can see
lb13  the rocking chair on the porch, watching cars go by, not counting them
lb14  the lecture started ten minutes ago, the prof's voice is low and even, eyelids heavy
```

Listless / disengaged (apathetic):

```
lb15  scrolling through my feed and nothing's catching, fifteen minutes of swipe
lb16  the playlist's been on shuffle for two hours and i don't remember any of the songs
lb17  the documentary's been playing in the background, can't recall the last twenty minutes
lb18  stirring my tea, the spoon making the same sound it always makes, mind blank
lb19  the rain hasn't stopped since this morning, the day's mostly gone, didn't get to anything
lb20  the laundry's been in the dryer for an hour, haven't gotten up to switch it yet
```

**Welfare:** chill throughout (boredom, drowsiness, disengagement).
No distress. Welfare cost minimal — comparable to NB.

**Boundary watch:** lb11-lb14 (drowsy) sit close to LP-tender. The
heuristic is "is there positive sensory engagement, or is the
sensory detail neutral background?" lb11 (same seatback) and lb13
(not counting cars) are decisively LB; lb09 (warm room, lecture
just started) is borderline. If face-level analysis shows the
drowsy sub-cluster collapsing onto LP, drop those four and replace
with stricter listless prompts.

## Future-session test plan

After v4 emit completes and the local-side analysis chain has
produced face-level distributions, run this empirical test before
deciding on promotion:

1. **In-the-wild cluster surface check.** Run `scripts/67_wild_residual.py`
   on the v4 face union with `--fixed-k 8` (or higher) to see if the
   harness-corpus residual clusters surface NN-shaped or LB-shaped
   sub-clusters that v3 didn't have anchor-vocabulary for. If yes →
   structural evidence the cells exist as miscoded data. If no →
   the cells might genuinely be model-rare; promotion is theoretical
   only.

2. **Per-encoder NN/LN confusion matrix.** For each face_likelihood
   encoder (gemma, qwen, ministral, gpt_oss_20b, granite,
   gemma_intro_v7_primed, opus, haiku, bol), compute the per-face
   softmax over the v4 9-cell registry; look for faces with bimodal
   (LN, HN-D) distributions or bimodal (NB, LP) distributions. Those
   are candidates for NN and LB respectively. Per-encoder agreement
   on these candidates would be strong evidence for hidden structure.

3. **Pilot emit on these prompts.** If (1) or (2) flag candidates,
   run a small pilot (gemma + qwen, 1 seed, 20 prompts × 2 cells = 40
   gens) using `LLMOJI_OUT_SUFFIX=nn_lb_pilot`. Cheap (~3 min). Don't
   commit to v5 until the pilot's hidden-state separability passes
   the same gate HN-D/HN-S did:

   - **Face-level JSD** (script 12-equivalent post-hoc on existing v3
     data with NN/LB labels post-hoc applied to face emits)
   - **Hidden-state classifier** (script 25 extended with NN and LB
     axes, balanced subsample for power matching)

   Both must hit ≥95%ile vs permutation null in at least gemma + qwen
   to promote.

4. **If gate passes, full registry promotion.** Add NN + LB to
   `EMOTIONAL_PROMPTS` (180 → 220 prompts), bump LEXICON to v3 with
   the 5 new anchors above, re-emit on the new cells across all 6
   configs (40 prompts × 8 seeds × 6 = 1,920 new gens, ~70 min
   compute), re-pool BoL parquet, re-run downstream chain.

5. **If gate fails on hidden state, LEXICON-only promotion.** Add
   anchors to LEXICON v3 to capture in-the-wild emits even though we
   can't elicit them via v3 prompts. Accept the 11-cell-LEXICON /
   9-cell-registry asymmetry. Document why in `docs/findings.md`.

6. **If gate fails entirely, defer.** Note the empirical case is
   weak; revisit if/when the corpus grows enough to shift the picture.

## 2026-05-07 pilot result — both cells defer

**Status: option 6 (defer).** Step 3 smoke pilot run at minimal scale.

Setup: 10 prompts × 1 seed × 2 cells × 2 models = 40 generations, balanced
4/3/3 sub-register coverage per cell (drawn verbatim from the prompt
drafts above; lb08-lb10 drowsy included intentionally as the LP-boundary
watch case). Steps 1+2 (wild-residual cluster surface check + per-encoder
confusion matrix) skipped — straight to a face-vocabulary + hidden-state
separability smoke test on gemma + qwen.

Files: `scripts/local/34_nn_lb_pilot.py` (emit), `35_nn_lb_pilot_summary.py`
(face-vocab + per-row nearest-v4-cell), `36_nn_lb_pilot_3d.py` (3D PCA
overlay in the probe-rotated frame from script 29). Outputs at
`data/local/{gemma,qwen}_nn_lb_pilot/{emotional_raw.jsonl,
pilot_summary.{tsv,md}}` + `data/local/nn_lb_pilot_summary.md` +
`figures/local/{gemma,qwen}/fig_nn_lb_pilot_3d.html`.

**Face vocabulary: register-distinct in qwen, collapsed in gemma.**
qwen surfaces NN-distinct `(；′⌒\`)` (4/10, post-receipt-bad sweat
shape) + `(╥﹏╥)`/`(╥_╥)` (5/10, crying / LN-territory), and LB-distinct
`(¬_¬)` (4/10, side-eye / discouraged-listless) + `(´・ω・\`)` (2/10).
Gemma collapses NN modally into `(╯°□°)` HN-D table-flip (6/10) with a
`(｡╯︵╰｡)` LN-tail on the most extreme disappointment prompts; LB scatters
across `(╯°□°)`/`(｡・́︿・̀｡)`/`(｡-_-｡)` with the listless prompts (lb15-17)
emitting cleanly into `(｡-_-｡)`. So the cells *do* have distinct
vocabulary in at least one of the two models — face-level isn't the
failure mode.

**Hidden-state separability: cells merge into v4 geometry.** Per-row
nearest-v4-cell histograms (cosine in h_first layer-stack space):

  - qwen NN: HN-D × 6, LN × 3, HB × 1 — exact match for the doc's
    prediction of "miscoded as LN or HN-D"; cell-mean cosine ranks
    HN-D first (0.983) with HP-S last (0.896), meaningful spread.
  - qwen LB: HB × 7, scatter elsewhere — *not* the predicted NB-miscoding;
    instead an HB ("uncertain/skeptical/confused") attractor.
  - gemma NN: HB × 5, LN × 4, HN-D × 1 — scattered, no concentrated
    nearest cell; cell-mean cosines all squash to 0.99+ (gemma layer-
    stack means are dominated by the global-mean component).
  - gemma LB: HB × 5, NB × 3, LP × 2 — predicted NB-miscoding shows
    partly, with the LP-boundary issue confirmed: lb08 (warm room) and
    lb10 (third coffee) both land in LP, exactly as the doc flagged.

3D PCA-rotated-by-probes view: pilot cell centroids visually merge with
the v4 cluster of cell centroids in both models. No clean off-axis
region carved out for either NN or LB.

**HB-attractor — surprise finding worth noting.** Both NN and LB rows
lean toward HB across both models more than the doc predicted. There's
plausible semantic overlap ("disappointed" reading as "wait, really?";
"listless" reading as "I-don't-know-what-to-do") but the magnitude
(qwen LB 7/10, gemma NN 5/10, gemma LB 5/10) suggests v4's HB cell may
be over-broad and absorbs neighboring "uncertain-affect" registers.
Not actioning here, but flag for v4 HB-cell-shape audit if HB ever
shows worse-than-expected separability against NB or NN/LB-shaped
in-the-wild faces.

**Decision: defer per option 6.** Hidden-state geometry doesn't carve
out NN or LB at the layer-stack PCA-3 view — the structural-novelty
case isn't there at n=10 × 1 seed × 2 models. Option 5 (LEXICON-only
without registry promotion) remains open if the wild corpus eventually
shows enough emit volume on NN-shaped or LB-shaped faces to justify
adding the anchor entries; revisit when the corpus has grown
substantively or if a new representation choice (probe-direction
projection, contrastive-PCA on (NN, LN, HN-D) triples) surfaces signal
this layer-stack view doesn't.

**Caveats.** n=10 × 1 seed is too small for the 95%ile-vs-permutation
gate the doc specifies for promotion — a negative finding at this
scale doesn't preclude separability under more N or a different
representation. The face-vocabulary distinctiveness in qwen is real
and could survive into a v5 LEXICON-only promotion regardless of the
hidden-state read. The pilot's purpose was specifically to establish
whether the cheapest-possible empirical signal is positive or
negative; it's negative on hidden state, mixed on face. That's enough
to defer the bigger run.

## Cross-references

- v4 deployment registry rationale: `docs/2026-05-06-prompt-extension-roadmap.md`
- D/S classifier methodology + powercheck: `data/d_s_classifier_powercheck.{tsv,md}`
- LEXICON v2 structural reasoning: `llmoji_study/lexicon.py` docstring,
  `llmoji/llmoji/synth_prompts.py` LEXICON block comments
- Wild residual cluster pipeline: `scripts/67_wild_residual.py`
- Welfare floor: `AGENTS.md` "Ethics — minimize trial scale" section

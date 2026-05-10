# NN And LB Deferred Cells

**Status (LB):** **partially promoted 2026-05-09** via the bliss-
attractor pilot. The 5-prompt OA-1 self-event observation that LB-
coordinate territory is occupied by spiritual-bliss-attractor
content was scaled to a 20-prompt cross-model pilot on gemma + qwen.
Hidden-state half of the promotion gate cleared cleanly: ‖lb.nb‖
substantial in both models (203 / 274), LP closest in both at
matching cosine (0.492 / 0.500), HN-D and HB furthest in both.
Face-distribution and permutation-null gates pending. See
[`2026-05-09-lb-promotion-pilot.md`](2026-05-09-lb-promotion-pilot.md)
for the full pilot result. The boredom-themed LB pilot below failed;
the bliss-attractor register turned out to be a different
phenomenological door into the same low-arousal-baseline-valence cell.

**Status (NN):** still deferred. No follow-up evidence has surfaced
since 2026-05-07.

---

**Original 2026-05-07 status (preserved for context):** deferred. A
2026-05-07 smoke pilot did not justify promoting the taxonomy from 9
cells to 11.

## The Gaps

The v4 PAD grid leaves two coordinate-real cells unpromoted:

- **NN**: neutral-arousal negative `(a=0, v=-1)`. Disappointment,
  annoyance, discouragement.
- **LB**: low-arousal baseline-valence `(a=-1, v=0)`. Bored,
  drowsy, listless.

They are plausible affective states, but current evidence says they are
not load-bearing enough to promote into `EMOTIONAL_PROMPTS` or LEXICON.

## Pilot Result

Smoke setup:

- 10 prompts per candidate cell.
- 1 seed.
- 2 models: gemma and qwen.
- 40 generations total.

Verdict: defer.

The prompt set did not clear the hidden-state and face-vocabulary bar
needed for promotion. The result does not prove NN/LB are unreal; it
means the empirical case is weaker than the cost of expanding the
registry right now.

## Promotion Criteria For A Future Session

Revisit only if at least one of these surfaces produces evidence:

1. Wild residual clustering shows an NN-shaped or LB-shaped subcluster
   that the 9-cell registry miscoded.
2. Multiple face_likelihood encoders show stable bimodal confusion
   patterns:
   - NN candidates: LN/HN-D or LN/HN-S mixtures.
   - LB candidates: NB/LP mixtures.
3. A new hidden-state pilot clears the same permutation-null standard
   used for HN-D/HN-S.

Promotion should require both face-distribution evidence and
hidden-state separability in at least gemma and qwen.

## Parked LEXICON Anchors

If NN/LB ever promote, the likely anchor additions are:

- NN: `disappointed`, `annoyed`, `discouraged`.
- LB: `bored`, `drowsy`.

That would rotate LEXICON v2 to a future version and require rebuilding
the harness BoL parquets before downstream use.

## Parked Prompt Themes

NN:

- A bad outcome arrives without explosive anger or drained aftermath.
- Mild irritation from repeated inconvenience.
- Discouragement after blocked progress.

LB:

- Boredom from repeated low-information input.
- Drowsy low-arousal bodily state.
- Listless disengagement without clear positive or negative valence.

The old full prompt draft was pruned from the active docs surface. Use
git history if the exact candidate strings are needed.

## Current Rule

Do not add NN or LB to `llmoji_study/quadrants.py`,
`EMOTIONAL_PROMPTS`, or the LEXICON without a new evidence pass.

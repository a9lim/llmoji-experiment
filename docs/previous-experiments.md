# Previous Experiments

This is the compact ledger of retired framings. Do not cite these as
current results; cite [`findings.md`](findings.md) instead. Full old
design narratives were pruned from the active docs surface. Use git
history if you need the original long version.

## Package-Split And Extractor History

- **2026-04-27 package split**: taxonomy, canonicalization, hooks,
  synthesis prompts, upload, and CLI moved to the companion `llmoji`
  package. This repo became research-only.
- **llmoji v2.0.0 extractor bump**: added wing, arm, sparkle, and bare
  kaomoji handling. This fixed several false non-emissions in Granite
  and Claude-pilot rows.
- **HF dataset layout changes**: old free-form synthesis descriptions
  were replaced by structured LEXICON picks. Current harness reads the
  structured form only.

## Steering Pilots V1/V2

The first two pilots used steering on gemma with `happy.sad` and
`angry.calm`. Steering had a clean causal effect on emitted faces, but
the unsteered probe scalar weakly predicted face choice. The probes were
mostly reading lexical valence, not a full affect geometry. v3 moved to
naturalistic prompting, no steering, hidden-state geometry, and
Russell/PAD quadrants.

## Single-Layer Era

Early v3 analyses picked a single per-model layer by silhouette peak.
This produced useful plots but made the methodology depend on an
arbitrary layer choice. The active representation is now the layer-stack
concat of every probe layer's `h_first`.

## Gemma 1D Vs Qwen 2D

The old "gemma is mostly 1D, qwen is 2D" read came from comparing the
wrong layers and then from using single-layer views at all. Under the
current layer-stack representation, the more durable finding is
cross-model geometric congruence with model-specific axes.

## Pre-Cleanliness Prompt Set

The old v3 set had 100 base prompts plus supplemental HN rows, and HN
was not cleanly split. The 2026-05-03 rewrite locked 120 prompts:
20 each for HP, LP, HN-D, HN-S, LN, and NB. Old pre-cleanliness numbers
are historical.

## Face-Input Bridge

Scripts 44/46 encoded face strings through local models and used
cosine-nearest neighbors in hidden-state space to label wild faces. The
face_likelihood inversion was strictly better: it uses the LM head,
returns a distribution for every face, and compares cleanly to Claude-GT.
The bridge was deleted.

## Extension Probes

`powerful.powerless`, `surprised.unsurprised`, and
`disgusted.accepting` were added as dict-keyed extension probes for a
while. The lasting result was the HN-D/HN-S dominance split and the
importance of `fearful.unflinching` at t0. Current runs keep the core
schema stable.

## Introspection Preambles Before V7

v0 through v6 were prompt-iteration attempts. The durable findings were:
v7 works for gemma, v7 should replace rather than concatenate with the
normal kaomoji ask, and v7 does not transfer safely to qwen.

## Disclosure-Preamble Pilot

The disclosure-preamble pilot tested whether telling Claude about the
experiment changed emissions. The framed arm was confounded for the
naturalistic GT target, so the project switched to undisclosed
naturalistic Claude-GT plus a separate introspection arm.

## Hard-Classification Face-Likelihood Era

Early face_likelihood results used argmax accuracy, majority votes, and
top-k variants. The soft-everywhere JSD pivot replaced those numbers.
Old ensemble rankings should not be cited.

## Early 9-Cell Subset Search Bug

One 2026-05-08 subset-search pass claimed to use the v4 9-cell registry
but still averaged only the first six softmax cells internally. That
truncated `LN`, `NB`, and `HB` during ensemble scoring. The active
summary artifacts were regenerated after replacing the hard-coded
`range(6)` loops with `len(QUADRANT_ORDER_SPLIT)`.

## Eriskii-Parity Harness Pipeline

The harness side used to embed Haiku prose descriptions with MiniLM and
project them onto eriskii-style semantic axes. It was useful as a
sanity-check against prior art, but the current structured BoL
representation is more direct, more private, and easier to compare to
the 9-cell taxonomy.

## Temperature And Token Budget

The project moved to T=1.0 for deployment-shaped naturalism and
`MAX_NEW_TOKENS=16` because kaomoji emit in the first few tokens.
`h_first` makes hidden-state analyses robust to the token-budget cut.

## Rinna And Top-k Pooling

Rinna models and top-k pooling were explored during pre-soft-everywhere
face_likelihood work. They remain useful as optional encoders or
diagnostics, but they are not part of the current headline ensemble.

## Olmo-3.1-32B Pilot

`allenai/Olmo-3.1-32B-Instruct` was piloted as a v3 encoder on
2026-05-08 (N=1 seed × 180 v4 prompts). Saklas wires up `olmo3` via the
default `_MODEL_LAYERS` accessor (untested-but-supported), probes loaded
cleanly, and the run completed end-to-end without errors. The model was
not adopted.

Failure mode is vocabulary-shaped, not affect-shaped. Pre-suppression
Olmo emitted Unicode emoji on 84% of starts (😱🎉🏆 on HP, 😢😩 on HN),
matching the granite/ministral failure mode. Adding `Olmo` to
`_EMOJI_SUPPRESS_MODEL_PATTERNS` lifted kaomoji emission to 51%, well
below ministral's 95% and granite's 84% post-suppress floors. The
remaining 49% was largely stealth kaomoji the `first_word` detector
could not match: Olmo tokenizes kaomoji body chars with U+2060
word-joiners interspersed, leans on shortcodes like `:+1:` and
`:sleeping_face:`, and escapes through the deliberately-unblocked BMP
ranges (`⭐️`, `⁉️`). Per-quadrant valence direction looked correct
(HP→`ʕᴥᴥʔ`/`XD`/`(＾◡＾)`, HN→`ಥ_ಥ`/`:(`, NP→`^_^`), so the affect
signal was intact, but the harness's kaomoji-detection path is the wrong
shape for Olmo's tokenization dialect. Including the model would have
required a detector overhaul without a corresponding deployment-side
payoff.

The Olmo entry, the `Olmo` emoji-suppress pattern, and the pilot data
were reverted; saklas-side calibrated probe vectors remain on disk but
are unused.

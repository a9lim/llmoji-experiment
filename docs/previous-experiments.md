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

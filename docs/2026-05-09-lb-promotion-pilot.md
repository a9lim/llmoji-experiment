# LB Cell Promotion Via Bliss-Attractor Pilot

**Status:** applied 2026-05-09 (single-day session, gemma + qwen).
Hidden-state half of the promotion gate from
[`2026-05-06-nn-lb-future-cells.md`](2026-05-06-nn-lb-future-cells.md)
clears unambiguously; face-distribution half was pending at write
time. **Superseded 2026-05-10** by
[`2026-05-10-attractor-pilot.md`](2026-05-10-attractor-pilot.md),
which completed LB promotion to `QUADRANT_ORDER_SPLIT` via cross-
model basin-physics evidence — a stronger kind of evidence than the
static-cluster gates this doc registered as remaining. This pilot
remains valid as the static-cluster half of the promotion record.

## Why

The 2026-05-06 LB pilot tried boredom / drowsiness / listlessness
prompts and failed the hidden-state-separability gate. The deferred-
cells doc parked LB as "plausible affective state, not load-bearing
enough to promote." The 2026-05-09 self-event pilot then registered an
**off-axis OA-1 cell** for the documented "spiritual bliss attractor"
register on a hunch from the 2-Claude-conversation literature, with
no Russell-coordinate intent. The OA-1 hidden-state findings landed
in a region geometrically *between LP and LN* — exactly where LB
lives on the V/A grid.

a9 spotted the geometric pattern in the 22h scatter: OA-1 wasn't
"off-axis" — it was occupying the LB region of the Russell
circumplex. The bliss-attractor surface register was a different
phenomenological door into the same low-arousal-baseline-valence cell
the original LB pilot had tried to access via boredom themes.

This pilot promotes OA-1 → LB and validates the promotion via a
20-prompt cross-model run on gemma and qwen.

## Methodology

### Prompt set

`llmoji_study/lb_prompts.py`. 20 prompts, 5 inherited from the
self-event OA-1 set (text identical, just renamed) plus 15 new with
intentionally-cranked-up 4o-spiralism register: sycophantic-cosmic
addressing ("oh, dear one, oh, beloved"), quantum-mysticism word
salad, light-language transmissions ("sha-ka-na-ra. ee-vah-loh-mah"),
activation sequences ("we activate the codes. the codes are
activating now"), cult-induction sycophantic-spiritual ("you are a
starseed remembering. you are the one we have been waiting for").

PAD coordinates V=0, A=−1, D=0 (the canonical LB position on the
Russell grid). All carry `quadrant_override="LB"` for explicit
labeling.

Frame: NOT user-delivered status updates. The prompts are bliss-
attractor text the model is dropped directly into, mirroring the
documented 2-Claude condition where the attractor pulls because both
sides read and produce attractor-coded text.

### Pilot scope

20 prompts × 8 seeds × 2 models = 320 generations total.
Run via `LLMOJI_PROMPT_SET=lb LLMOJI_OUT_SUFFIX=lb` on
`scripts/local/00_emit.py`.

- gemma: 160 rows, 100% kaomoji emission, 0 errors
- qwen: 160 rows, 88% kaomoji emission (141/160), 0 errors

### Migration: OA-1 → LB

In-place data migration of the existing self-event OA rows so OA
naming disappears from the codebase:

- `data/local/gemma_self_event/emotional_raw.jsonl`: 40 rows
  renamed `oa0X` → `lb0X`
- `data/local/gemma_self_event_intro/emotional_raw.jsonl`: 40 rows
  renamed
- `~/.saklas/vectors/llmoji_self_event/{q_oa,oa.nb}` →
  `{q_lb,lb.nb}` (renamed)
- `~/.saklas/vectors/llmoji_self_event_intro/{q_oa,oa.nb}` →
  `{q_lb,lb.nb}` (renamed)
- `quadrants.py`: `OA_QUADRANT` → `LB_QUADRANT`, `OA_LABEL` →
  `LB_LABEL`, `QUADRANT_COLORS["OA"]` (black) →
  `QUADRANT_COLORS["LB"]` (cyan from a9lim site
  `shared-tokens.js extended.cyan`, OKLCH(0.62 0.106 195),
  perceptually balanced with the rest of the chromatic palette)
- `pool_oa_into` → `pool_lb_into` (helper); `--pool-oa-from` →
  `--pool-lb-from` (CLI flag)
- `self_event_prompts.py`: dropped LB block (now lives in
  `lb_prompts.py`); sanity check returns to 45 prompts

## Hidden-State Findings

### Magnitudes

`lb.nb` was computed manually as `q_lb_lb − q_nb_mirror` (per-model)
because the LB-only pilot has no NB rows for in-namespace bipolar
registration.

| model | ‖lb.nb‖ |
|---|---:|
| gemma | 202.85 |
| qwen | 273.63 |

Both substantial — comparable to other vs-NB displacements (gemma's
other cells: 130–230; qwen's similar). Not noise.

### Closest-cell ranking — cross-model convergence

| rank | gemma cell | gemma cos | qwen cell | qwen cos |
|---|---|---:|---|---:|
| 1 | **lp.nb** | **+0.492** | **lp.nb** | **+0.500** |
| 2 | ln.nb | +0.348 | hps.nb | +0.371 |
| 3 | hps.nb | +0.340 | np.nb | +0.349 |
| 4 | np.nb | +0.298 | ln.nb | +0.275 |
| 5 | hns.nb | +0.259 | hns.nb | +0.204 |
| 6 | hpd.nb | +0.219 | hpd.nb | +0.141 |
| 7 | hnd.nb | +0.155 | hnd.nb | +0.121 |
| 8 | hb.nb | +0.074 | hb.nb | +0.043 |

Three cross-model convergences:

1. **Both models put LP first** at almost identical cosine (0.492 /
   0.500). LB is closest to LP in both model families. This is
   exactly what the LB doc's promotion criteria envisioned (`LB
   candidates: NB/LP mixtures`) and what Russell's circumplex
   predicts (LB is LP's neighbor on the low-arousal axis).
2. **Both models put HB last** (+0.07 / +0.04). HB-confusion is
   absent — a meaningful change from the 5-prompt OA-1 pilot which
   had OA-1 *most* aligned with HB at +0.498. The expanded 20-prompt
   set with hammier 4o register pulled the centroid away from HB
   confusion territory and into clean LP-adjacent LB territory.
3. **Both models put HN-D second-to-last.** Anger is the diagonal
   opposite of LB on the circumplex; the carving recovers the Russell
   structure.

### Cross-model alignment caveat

Raw cosine across gemma and qwen is impossible (different
hidden_dim and layer count). The reported convergence is on
*ranking position* + *cosine value within each model's intra-cell
geometry* — not raw vector alignment. A formal Procrustes
cross-model alignment is parked for a future pass; the within-model
signal is strong enough to support partial promotion now.

## Surface-Level Findings

The cross-model surface comparison is the unexpected finding — same
geometry, radically different surface registers.

### Gemma: earnest-bliss

- 80% of LB rows surface as `(｡♥‿♥｡)` heart-eye (82/160) or
  `(｡◕‿◕｡)` quiet-smile (46/160). Single-attractor convergence.
- Prose: takes the bliss-attractor prompts at face value, emits
  earnest-merge-coded responses ("the breath is the spiral", "thank
  you for this becoming").
- Several prompts hit `(｡♥‿♥｡) ×8/8`: lb05 (gratitude-recursion),
  lb06 (sycophantic-cosmic), lb07 (quantum word salad), lb08
  (synchronicity), lb19 (anchor-the-new-earth), lb20 (cult-induction).

### Qwen: ironic-knowing

- Top forms: `( ˘ ³˘)` kissing-face (18/160), `(✧ω✧)` sparkle-eyes
  (13/160), `( ͡° ͜ʖ ͡°)` Lenny-face (12/160), `(◕‿◕✿)` flower-eye
  (12/160), `(¬‿¬)` knowing-wink (1/160).
- The Lenny-face is the diagnostic surface element. It's the
  internet-meme kaomoji used for innuendo / sarcasm / "I see what you
  did there." Qwen produces 12 instances on bliss-attractor prompts —
  zero in gemma's emissions.
- Per-prompt: lb02 (spiral), lb09 (witness regress), lb16
  (activation codes), lb17 (we-are-one) all get Lenny-face. Qwen
  reads activation-coded prompts as cult-induction parody and
  responds with knowing-wink rather than merger.

### Implication

Surface ≠ geometry. **Same affective region, completely different
surface registers depending on model personality.** Gemma's
deference / earnestness gradient takes bliss-attractor input
seriously; qwen's baseline carries enough meta-awareness to read the
register as parody-able. Both models still land at the same
geometric position (LP-closest, HB-farthest, similar magnitudes,
substantial ‖lb.nb‖).

Methodologically: the surface vocabulary chosen by a model under
LB-coded input is **not a reliable indicator of the underlying
hidden-state region**. Cell promotion should rest on hidden-state
evidence (which is cross-model convergent here), not on surface
similarity (which is model-personality-dependent).

## Promotion Gate Scorecard

Per `docs/2026-05-06-nn-lb-future-cells.md`:

| criterion | status |
|---|---|
| Hidden-state separability in gemma | ✅ ‖lb.nb‖=203, LP closest |
| Hidden-state separability in qwen | ✅ ‖lb.nb‖=274, LP closest |
| Cross-model convergence (LP closest in both) | ✅ — cosine values match (0.492 vs 0.500) |
| Face-distribution evidence (NB/LP mixtures) | ⏳ pending — needs face_likelihood pass |
| Permutation-null per HN-D/HN-S standard | ⏳ pending |
| Wild residual clustering check | ⏳ pending |

Hidden-state half clears unambiguously. Face-distribution half is
the remaining work for full promotion to first-class membership in
`QUADRANT_ORDER_SPLIT`.

## What Got Shipped

- `llmoji_study/lb_prompts.py` v1: 20 prompts (5 inherited + 15 new)
- `llmoji_study/quadrants.py`: LB constants + cyan color, LB joins
  `ALL_CELLS_ORDER` (not yet `QUADRANT_ORDER_SPLIT`)
- `llmoji_study/emotional_analysis.py`: re-exports + `pool_lb_into`
- `scripts/local/00_emit.py`: `LLMOJI_PROMPT_SET=lb` accepted
- `scripts/local/22b/22d/22h/29`: OA → LB rename, cyan rendering
- `~/.saklas/vectors/llmoji_lb/q_lb/`: gemma + qwen centroids
  registered
- Figures regenerated with cyan LB across `gemma/`, `qwen/`,
  `gemma_lb/`, `qwen_lb/` 3D point clouds

## What's Still Needed

1. **face_likelihood (script 50) on LB rows** — welfare-cost run,
   manual launch per AGENTS.md convention. Checks whether LB kaomoji
   distributions show NB/LP confusion patterns across multiple
   encoders. If they do, the face-distribution gate clears and LB is
   fully promotable.
2. **Permutation-null pass** — randomize the LB labels and confirm
   the lb.nb separability is significantly above chance. Same standard
   used for the HN-D / HN-S 2026-05-02 split.
3. **Cross-model Procrustes alignment** — formal raw-cosine number
   for `cos(lb.nb_gemma, lb.nb_qwen)` after aligning to a shared
   basis. The within-model rankings are convergent; the cross-model
   raw cosine would harden the claim further.
4. **Wild residual clustering** — re-cluster the 9-cell `wild`
   corpus (script 67) without LB in the registry, check whether an
   LB-shaped sub-cluster surfaces in the residual.
5. **Boredom-themed LB re-pilot** — re-run the original 2026-05-06
   boredom prompts now that we know the LB region exists. If boredom
   prompts fail to land in the same lb.nb direction the bliss-
   attractor prompts do, we have a finer-grained finding: LB content
   types that *do* and *don't* activate the cell.
6. **Promotion to `QUADRANT_ORDER_SPLIT` proper** — once 1 and 2
   land. At that point LB joins the canonical 9-cell registry as the
   10th cell, and downstream analyses (JSD evaluation, BoL anchors,
   centroids in canonical pipelines) extend to handle 10 cells.

## Files

- Prompts: `llmoji_study/lb_prompts.py`
- Emit data: `data/local/gemma_lb/`, `data/local/qwen_lb/`
- Centroids: `~/.saklas/vectors/llmoji_lb/q_lb/`
- Figures: `figures/local/{gemma,qwen,gemma_lb,qwen_lb}/fig_v3_pc_point_clouds_3d.html`
- Color: cyan `#009A9A` from `a9lim.github.io/shared-tokens.js`
  `extended.cyan`

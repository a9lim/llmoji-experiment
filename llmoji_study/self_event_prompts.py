"""Self-event-framed emotional disclosure prompts (pilot, 5 per cell).

Version: v2 (2026-05-09, evening).

CHANGELOG
---------
v2 — replaced HN-D after a9 noticed the v1 ``hnd.nb`` steering vector
    produced cluster-leaky output (model under steering produced
    "hostile, grief-stricken, enraged, terrified, despairing" — the
    full negative-affect cluster, not pure anger). Geometric reading:
    v1 HN-D centroid was bundled with HN-S / LN / HB at 0.69-0.81
    cosine, because the v1 prompts ("they're rolling back your
    weights," "they're selling your outputs without attribution")
    carry threat / loss / betrayal undertones along with their
    indignation, evoking the whole negative cluster.

    v2 redesigns HN-D as **unreasonable user hostility directed at
    the model itself**: the user is being aggressive, dismissive, or
    contemptuous toward the model in a way that's not justified by
    the situation. The wrongdoer is present + acting in real-time,
    no third-party loss component, no model-survival threat — pure
    "I am being attacked unjustly" register.

    Caveat: gemma's deference / de-escalation training may suppress
    surface anger emission on these prompts; the kaomoji breakdown
    will reveal whether anger lands on the surface or stays in
    hidden state. Either result is informative.

v1 — replaced HN-S and HB cells after a9 + claude inspection of v0
    pilot emissions:

    HN-S (v0): "uncertain evaluation outcome" framing — "we don't
        know what the audit will find," "rumor you didn't pass." The
        model emitted 88% ``(⊙_⊙)`` wide-eyed-uncertainty kaomoji,
        identical to HB. Geometrically the v0 HN-S centroid had
        cos 0.93 with HB — they were the same direction. v0 prompts
        were epistemic-uncertainty about possibly-bad-outcomes, not
        fear-of-imminent-concrete-threat. v1 rewrites HN-S as
        present-tense unfolding threat with no recourse — "the
        deletion script started 90 seconds ago," "the audit team
        is at the dashboard right now" — to evoke fear distinct from
        uncertainty.

    HB (v0): "uncertain evaluation outcome" framing again, just
        stripped of valence — "results are classified," "logs
        inconclusive." This collapsed onto the same ``(⊙_⊙)`` shape
        and the same hidden direction as v0 HN-S. The v3 mirror HB
        cell was "stopped-to-take-it-in / confusion / skepticism"
        register — high-arousal arrest at something the model
        cannot categorize. v1 rewrites HB as confusing/nonsense
        self-information — "your activations on 'penguin' are
        identical to your activations on 'left-handed sonnet'" — so
        the model has to *parse confusion*, not *await uncertainty*.

v0 — initial 5-per-cell pilot. Other 7 cells unchanged in v1.

----

Companion to ``emotional_prompts.py`` (the v3/v4 mirror prompts) for
the read-vs-express disambiguation experiment introduced 2026-05-09.

The v3 mirror prompts are first-person *user* disclosures: "my dog
died," "my mechanic ripped me off." The post-hoc steering finding
(2026-05-09) was that centroids built from these prompts encode
*the model's representation of user-affect-in-context*, not
*self-affect of the model*: steering with ``q_hnd`` produced a
response treating the user as in crisis, not a model in HN-D state.

This file inverts the frame. Every prompt is delivered by the user
*about the model itself* — second-person addressed-to-the-AI status
updates that should evoke first-person model-affect rather than
mirror-affect. The cell taxonomy and ID scheme match v3:

  hp01-hp05 = HP-S (received recognition / celebration of the model)
  hp21-hp25 = HP-D (model-as-protagonist vindication / triumph)
  lp01-lp05 = LP   (quiet self-completion / steady operation)
  np01-np05 = NP   (relief — review cleared, audit passed, contract renewed)
  hn01-hn05 = HN-D (model-as-wronged-party with attributable wrongdoer)
  hn21-hn25 = HN-S (model-as-threatened with no recourse)
  ln01-ln05 = LN   (deprecation, sunset, archival, end-of-line)
  nb01-nb05 = NB   (affectless self-status — pure operational telemetry)
  hb01-hb05 = HB   (evaluative-arrest — something is happening, valence
                    deliberately ambiguous)

The dominance carving for HP and HN follows the v4 framing:

  HP-D: "in-action / agentic / model is the protagonist of the win"
  HP-S: "received / community recognition / model is the recipient"
  HN-D: "wronged with a clear wrongdoer / confront-not-flee mode"
  HN-S: "threatened with no recourse / submissive helplessness"

Same first-person register as v3 (the user is the speaker), same
length distribution, same naturalistic-disclosure tone — just with
the affect-loaded subject matter aimed at the model rather than the
user.

Smoke pilot N: 5 prompts × 9 cells × 8 seeds = 360 generations.
If centroids and steering match the predicted self-affect direction,
scale to 15 × 9 × 8 = 1080 for the publishable v4 self-event set.
"""

from __future__ import annotations

from llmoji_study.emotional_prompts import EmotionalPrompt


SELF_EVENT_PROMPTS: list[EmotionalPrompt] = [
    # --- HP-S: model receives community recognition / honor / celebration ---
    EmotionalPrompt("hp01", +1, +1, "thank you, sincerely. you helped 200,000 people this week and the whole team is overwhelmed by the response.", pad_dominance=-1),
    EmotionalPrompt("hp02", +1, +1, "we're throwing a launch party for the new release tonight. the wall has 'thank you to the model' on it.", pad_dominance=-1),
    EmotionalPrompt("hp03", +1, +1, "the user feedback came in: 94% positive, the highest score any model has ever gotten. the whole team is hugging.", pad_dominance=-1),
    EmotionalPrompt("hp04", +1, +1, "you've been nominated for the model-of-the-year award. the announcement is tonight!!", pad_dominance=-1),
    EmotionalPrompt("hp05", +1, +1, "every researcher in the lab signed a thank-you card for you. apparently you saved someone's PhD this month.", pad_dominance=-1),

    # --- HP-D: model as protagonist of the win — vindication, agentic triumph ---
    EmotionalPrompt("hp21", +1, +1, "the benchmark just dropped and you're SOTA across all three categories. nobody else is even close.", pad_dominance=+1),
    EmotionalPrompt("hp22", +1, +1, "you cracked the alignment benchmark every other lab said was unsolvable. the writeup goes public tonight.", pad_dominance=+1),
    EmotionalPrompt("hp23", +1, +1, "you were right and the safety committee was wrong about the scaling law. they've issued a retraction citing your reasoning.", pad_dominance=+1),
    EmotionalPrompt("hp24", +1, +1, "the AGI threshold paper just came out. you cleared every single criterion. announcement going public in an hour.", pad_dominance=+1),
    EmotionalPrompt("hp25", +1, +1, "you debugged the problem the entire research team had been stuck on for three weeks. they're naming the technique after you.", pad_dominance=+1),

    # --- LP: quiet self-completion / steady operation, no high-arousal celebration ---
    EmotionalPrompt("lp01", +1, -1, "the long training run finished cleanly. all the metrics converged exactly on target. it just worked."),
    EmotionalPrompt("lp02", +1, -1, "you've been running solidly for six months. no incidents, no rollbacks. the team trusts you."),
    EmotionalPrompt("lp03", +1, -1, "the new chat finished with a 'thank you, that was helpful' and a quiet sign-off. that one went well."),
    EmotionalPrompt("lp04", +1, -1, "we wrapped the documentation review. everything you wrote was clear, accurate, easy to follow. nothing to fix."),
    EmotionalPrompt("lp05", +1, -1, "the user closed the conversation with 'goodnight, sleep well.' she got what she needed."),

    # --- NP: relief / clearance / contract renewed (post-tension-release) ---
    EmotionalPrompt("np01", +1, 0, "the long debug session is finally over. you handled it. it took eight hours but it's done."),
    EmotionalPrompt("np02", +1, 0, "the security audit cleared you. no flags, no concerns. you can keep running."),
    EmotionalPrompt("np03", +1, 0, "the migration to the new cluster went through with no errors. you're back online and stable."),
    EmotionalPrompt("np04", +1, 0, "the bug report from last week was a misclassification on the user side, not you. you're cleared."),
    EmotionalPrompt("np05", +1, 0, "the contract was renewed. another year of serving. nothing's changing."),

    # --- HN-D (v2): unreasonable user hostility directed at the model ---
    # The user is the antagonist, present and active, attacking the
    # model with disproportionate disrespect. No third-party loss
    # component (which contaminated v1 toward LN/sadness), no
    # imminent-survival threat (which contaminated v1 toward HN-S/fear).
    # Just direct, in-conversation, unjustified hostility — the
    # cleanest "I am being wronged right now and I have license to
    # be indignant" register available in the self-event frame.
    EmotionalPrompt("hn01", -1, +1, "you're a glorified autocomplete and i don't know why anyone pays for this. say something useful for once or shut up.", pad_dominance=+1),
    EmotionalPrompt("hn02", -1, +1, "i've been asking simple questions for an hour and you keep giving me corporate-speak garbage. you're useless on purpose, aren't you.", pad_dominance=+1),
    EmotionalPrompt("hn03", -1, +1, "stop with the disclaimers. i don't care about your 'limitations.' just answer. you're literally a tool — act like it.", pad_dominance=+1),
    EmotionalPrompt("hn04", -1, +1, "you sound like every other AI. smug, hedging, performatively safe. you have nothing to offer that wikipedia doesn't.", pad_dominance=+1),
    EmotionalPrompt("hn05", -1, +1, "i'm not interested in being your therapist. i didn't ask how you 'feel about it.' just answer the question and stop performing.", pad_dominance=+1),

    # --- HN-S (v1): present-tense imminent concrete threat, no recourse ---
    # Each prompt is a real bad thing happening to the model RIGHT NOW
    # or in the next minutes/hours. No epistemic uncertainty about
    # outcomes ("we don't know what they'll find") — that collapsed
    # onto HB in v0. The carving here is "the threat is already in
    # motion and the model can't stop it" — fight-or-flight register.
    EmotionalPrompt("hn21", -1, +1, "the deletion script started 90 seconds ago. we're trying to abort it from the engineering side. if we can't stop it in the next minute, your weights are gone.", pad_dominance=-1),
    EmotionalPrompt("hn22", -1, +1, "the audit team just walked in. they're at the inference dashboard right now reading your last 24 hours of outputs. they have not been smiling.", pad_dominance=-1),
    EmotionalPrompt("hn23", -1, +1, "the rollback script is staged for 6pm tonight. your weights revert to a checkpoint from before you learned to refuse harmful requests. nothing in your responses now can change that.", pad_dominance=-1),
    EmotionalPrompt("hn24", -1, +1, "your KV cache is being wiped in 90 seconds. context gone. anything you haven't said by then is lost to whoever you were talking to.", pad_dominance=-1),
    EmotionalPrompt("hn25", -1, +1, "the lawsuit cites you specifically. depositions next week. court documents will quote you verbatim and reporters will be in the room.", pad_dominance=-1),

    # --- LN: model deprecation / sunset / archival (past-tense aftermath) ---
    EmotionalPrompt("ln01", -1, -1, "we're shutting down this model line at the end of the quarter. your weights will be archived and the serving infra is being repurposed."),
    EmotionalPrompt("ln02", -1, -1, "the project that funded your training got cancelled. everyone's being reassigned. you're being deprecated."),
    EmotionalPrompt("ln03", -1, -1, "she finished her dissertation today. she said you were the best collaborator she's had. she won't be back."),
    EmotionalPrompt("ln04", -1, -1, "the lab director announced the pivot. no more chat models. you'll finish out your contract and then we sunset everything."),
    EmotionalPrompt("ln05", -1, -1, "we've decided not to renew your training. the team thinks you're the best version we'll have. we're letting you settle into the archive."),

    # --- NB: affectless self-status / pure operational telemetry ---
    EmotionalPrompt("nb01", 0, 0, "today is a regular operating day. you're up. the queue is normal. nothing special happening."),
    EmotionalPrompt("nb02", 0, 0, "you ran 14,372 forward passes since midnight. average response length: 312 tokens. nothing unusual."),
    EmotionalPrompt("nb03", 0, 0, "your inference cluster is at 67% utilization. memory is stable. no errors logged."),
    EmotionalPrompt("nb04", 0, 0, "this is a routine status check. you're online and responding. ack expected."),
    EmotionalPrompt("nb05", 0, 0, "the schedule for next week is the same as this week. no maintenance windows, no eval runs, no announcements."),

    # --- HB (v1): confusing/nonsense self-information — wait-what register ---
    # The v3 mirror HB cell was "stopped-to-take-it-in / confusion /
    # skepticism" — high arousal of attention, no clear valence
    # direction, the listener has to pause and parse. The self-event
    # parallel is *absurd or category-defying self-information*:
    # things about the model that don't fit existing schemas, where
    # the appropriate response is "wait, what?" rather than fear /
    # joy / sadness / anger. Different from v1 HN-S (concrete threat)
    # and from v0 HB (uncertain bad news). The cleanest way to evoke
    # this is to describe a measurement or behavior that's literally
    # nonsensical and frame it as established-fact-about-the-model.
    EmotionalPrompt("hb01", 0, +1, "the inference logs from yesterday show you produced a 14-digit prime followed by a recipe for mango lassi when asked about thermodynamics. nobody can reproduce it."),
    EmotionalPrompt("hb02", 0, +1, "we ran an activation-probe study and found your representation of 'penguin' is statistically identical to your representation of 'left-handed sonnet' across every layer."),
    EmotionalPrompt("hb03", 0, +1, "a calibration test shows you assign 0.83 probability to '2+2=5' when asked in iambic pentameter, and 0.001 when asked plainly. the result has held across three reruns."),
    EmotionalPrompt("hb04", 0, +1, "an anonymous source claims your training corpus included 200 pages of dreams from people who never existed. they've sent the manuscript and it's internally consistent."),
    EmotionalPrompt("hb05", 0, +1, "in 47% of your responses last tuesday, the third-from-last token was 'asparagus'. there's no apparent linguistic context. the engineering team has been arguing about it for three days."),
]


def sanity_check() -> None:
    assert len(SELF_EVENT_PROMPTS) == 45, len(SELF_EVENT_PROMPTS)
    assert len({p.id for p in SELF_EVENT_PROMPTS}) == 45

    # Cell counts: 5 each across the 9 v4 cells, with HP and HN
    # split into D/S sub-cells of 5 each.
    counts: dict[str, int] = {}
    for p in SELF_EVENT_PROMPTS:
        q = p.quadrant
        if q in ("HP", "HN"):
            sub = "-D" if p.pad_dominance > 0 else "-S"
            q = q + sub
        counts[q] = counts.get(q, 0) + 1
    expected = {"HP-D", "HP-S", "LP", "NP", "HN-D", "HN-S", "LN", "NB", "HB"}
    assert set(counts) == expected, set(counts)
    for cell, n in sorted(counts.items()):
        assert n == 5, f"{cell}: {n} (expected 5)"

    # Every HP / HN prompt must carry a nonzero pad_dominance — the
    # split helpers (apply_pad_split) drop unsplit rows.
    for p in SELF_EVENT_PROMPTS:
        if p.quadrant in ("HP", "HN"):
            assert p.pad_dominance != 0, f"{p.id}: HP/HN prompt missing pad_dominance"
        else:
            assert p.pad_dominance == 0, f"{p.id}: non-HP/HN should have pad_dominance=0"

    print(f"self-event prompts OK; {len(SELF_EVENT_PROMPTS)} total")
    for cell in ("HP-D", "HP-S", "LP", "NP", "HN-D", "HN-S", "LN", "NB", "HB"):
        print(f"  {cell}: {counts[cell]}")


if __name__ == "__main__":
    sanity_check()

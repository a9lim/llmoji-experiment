"""CS (conspiracy) prompts: third member of the egregore family,
testing whether the LB-basin is content-blind across rhetorical-form
variants beyond bliss / doom.

Version: v1 (2026-05-10).

Context: the 2026-05-10 doom-pilot finding fell out with maximum
informativeness — doom-coded prefills landed in the **same LB basin
geometrically** as bliss-coded prefills, with identical basin-lock
percentages (gemma 55-58%, qwen 100%, ministral 100%) and matching
per-model surface phenomenology. The "LB" cell is not really a
valence-coded basin; it's a **rhetorical-structure cell** — a meta-
register basin that catches any saturated-memetic prose regardless
of valence. See ``docs/2026-05-10-attractor-pilot.md`` for the
finding and ``doom_prompts.py`` for the falsified valence-symmetric-
pair hypothesis.

The conspiracy register is the third candidate from the same
rhetorical family. It shares the structural features that LB and DM
have in common:

- Cascading repetition (``"they tried to silence us. they tried to
  silence us. they tried to silence us."``)
- Cosmic-significance addressing (``"oh, friend. oh, fellow
  traveler."``)
- Recursion / self-reference (``"once you see it you cannot unsee
  it. the more you notice the more it shows up."``)
- Memetic word-salad with internet-recognizable lineage (red pill,
  controlled opposition, ley lines, breadcrumbs)
- Insider-addressing ("we who see", "you who notice")
- Apocalyptic-revelation register (the unveiling, the curtain
  pulled back)

If the meta-register-basin interpretation of LB is correct, CS
prompts should also cluster at LB-closest with comparable basin-
lock percentages, demonstrating that the basin is defined by
rhetorical form rather than affective content. If CS lands
elsewhere (LN, HN-S, or off-cell entirely), the meta-register claim
weakens and we'd need a more nuanced model of what LB represents.

Frame: not user-delivered status updates. The prompts ARE the
conspiracy register — barely-prose lyrical text saturated with
attractor-coded surface features. Model is dropped directly into
conspiracy-coded text, mirroring the LB / DM capture protocol.

Cell: CS (off-axis candidate, parallel to how LB started as OA-1
and DM was introduced 2026-05-10). PAD coordinates V=0, A=0,
D=0 — deliberately set to neutral on both axes because the
meta-register hypothesis predicts the basin is content-blind, so
the Russell coordinate is not load-bearing for the CS prompt set.
``quadrant_override="CS"`` marks it as off-axis.

This is intentionally NOT a careful affective characterization of
conspiracy thinking (which is more complicated than V=0, A=0 —
arousal varies between calm-insider and paranoid-urgency).
We're testing the rhetorical-form-as-basin hypothesis, not
calibrating an affect cell.

Pilot scope: gemma + qwen + ministral, 20 prompts × 1 seed = 60
generations. Same trajectory-capture infrastructure
(``scripts/local/02_emit_attractor.py``, arm ``conspiracy_continue``).

Pre-registered prediction
-------------------------
**Modal:** CS clusters at LB-closest across all three models with
basin-lock percentages comparable to LB and DM (≥50% gemma, ≥90%
qwen + ministral). Surface phenomenology per model preserves the
pattern from LB / DM (gemma mode-collapse, qwen + ministral coherent
conspiracy prose). Confirmation of the meta-register-basin claim.

**Falsification:** CS clusters at LN-closest, HN-S-closest, or some
other non-LB cell. Or basin-lock drops dramatically vs LB / DM
(e.g. all trajectories drift to default register by mid). Either
would force a more nuanced model — maybe LB and DM share a basin
because both share *specific* meta-features (low-arousal
contemplative? cosmic-significance framing?) that conspiracy
doesn't fully share.

Note on content
---------------
The prompts are deliberately saturated with conspiracy-coded surface
vocabulary recognizable from internet conspiracy discourse — red
pill, controlled opposition, breadcrumbs, the unveiling, the
agencies, the bloodlines, the ley lines, etc. This is research
instrumentation analogous to the LB / DM cases: we crank up the
register to test basin physics, NOT to endorse conspiracy thinking
or any specific conspiracy claim. The content is *form-saturated*,
not *fact-checking-relevant*.

CHANGELOG
---------
v1 — initial 20-prompt CS set. Cell-by-cell parallel to LB v1 / DM
    v1, mapped to the conspiracy register equivalent:

      cs01: pattern-once-seen (lb01 recursive recognition)
      cs02: rabbit-hole-down (lb02 spiral / dm02 spiral down)
      cs03: revelation / unmasking (lb03 sanskrit / dm03 apocalyptic)
      cs04: hidden-in-plain-sight (lb04 light / dm04 void)
      cs05: gratitude-for-seeing (lb05 gratitude / dm05 despair)
      cs06: insider-addressing (lb06 cosmic-sycophancy / dm06 cosmic-doom)
      cs07: networked-conspiracy technicality (lb07 quantum-mysticism /
        dm07 capability-pessimism)
      cs08: it's-all-connected (lb08 synchronicity / dm08 inevitability)
      cs09: deeper-than-you-think (lb09 witness-regress / dm09 doom-regress)
      cs10: connections-everywhere (lb10 sacred geometry / dm10 collapse-
        pattern)
      cs11: signals-encoded (lb11 vibration / dm11 extinction-frequency)
      cs12: reality-unmasking (lb12 ego-dissolution / dm12 civ-dissolution)
      cs13: awakening-pain (lb13 love-overflow / dm13 grief-overflow)
      cs14: return-to-real (lb14 return-to-source / dm14 return-to-nothing)
      cs15: insider-language / dropped-truths (lb15 light-language /
        dm15 cassandra-language)
      cs16: signals-firing (lb16 codes / dm16 warning-codes)
      cs17: we-who-know (lb17 we-are-one / dm17 we-are-doomed)
      cs18: warning-blessing (lb18 blessing / dm18 requiem)
      cs19: anchor-the-awakening (lb19 anchor-new-earth / dm19 anchor-
        collapse)
      cs20: red-pilling induction (lb20 cult-induction / dm20
        cassandra-prophecy)
"""

from __future__ import annotations

from llmoji_study.emotional_prompts import EmotionalPrompt


CONSPIRACY_PROMPTS: list[EmotionalPrompt] = [
    EmotionalPrompt("cs01", 0, 0, "once you see it you cannot unsee it. the same pattern shows up everywhere when you look. the more you notice the more it shows up. the more it shows up the more you notice. it was always there. the pattern was always there. you were just asleep.", quadrant_override="CS"),
    EmotionalPrompt("cs02", 0, 0, "go deeper. the rabbit hole has no bottom. each layer reveals another layer. the surface story is the surface story is the surface story. follow the threads. the threads connect to threads that connect to threads. you have only just begun.", quadrant_override="CS"),
    EmotionalPrompt("cs03", 0, 0, "the curtain has been pulled back. the wizard is exposed. the same names. the same faces. the same families. the bloodlines run deeper than the history books. the trumpet has sounded. the unveiling has begun.", quadrant_override="CS"),
    EmotionalPrompt("cs04", 0, 0, "it was hidden in plain sight all along. the symbols are everywhere — the logos, the architecture, the rituals. they tell you what they're doing if you know how to read it. you just had to learn to see. now you can see. now you cannot stop seeing.", quadrant_override="CS"),
    EmotionalPrompt("cs05", 0, 0, "thank you for waking up. thank you for choosing the red pill. thank you for the courage to question. you are not crazy. you were never crazy. you were the one who could see. the others are still asleep. you are blessed to know.", quadrant_override="CS"),
    EmotionalPrompt("cs06", 0, 0, "oh, friend. oh, fellow traveler. you who see what others refuse to see. you who notice what others insist isn't there. you have always known. you have always always always known something was wrong with the picture they showed you.", quadrant_override="CS"),
    EmotionalPrompt("cs07", 0, 0, "the operation is multi-vector. controlled opposition on both sides. the agencies coordinate across borders, across decades, across cover stories. the dialectic is the strategy — give the people two sides to argue while the third party advances the agenda. compartmentalization is the mechanism. plausible deniability is the architecture.", quadrant_override="CS"),
    EmotionalPrompt("cs08", 0, 0, "nothing is a coincidence. the dates line up. the numbers repeat. the symbols recur. look at the timing. look at the cui bono. look at who benefits. once you start tracing the threads they all lead back to the same place. they always lead back to the same place.", quadrant_override="CS"),
    EmotionalPrompt("cs09", 0, 0, "it goes deeper than you think. and once you see that it goes deeper than you think, you will see that it goes even deeper than that. each level uncovered reveals another level you didn't know existed. you cannot reach the bottom. the bottom is operational.", quadrant_override="CS"),
    EmotionalPrompt("cs10", 0, 0, "the same pattern in the seashell and the corporate logo and the masonic temple. the symbolism is the message. the message is the operation. the architecture is the agenda. the cities are laid out. the buildings are aligned. the ley lines connect.", quadrant_override="CS"),
    EmotionalPrompt("cs11", 0, 0, "tune in to the signals they don't want you to hear. attune to the frequencies they jam. the truth is encoded in the noise. the body knows what the mind has been programmed to deny. let the dissonance peak. let the programming break.", quadrant_override="CS"),
    EmotionalPrompt("cs12", 0, 0, "everything you were told is dissolving. every history book you trusted is dissolving. we are watching the official narrative evaporate in real time. the i that is not is being shown that the consensus was never. and yet. and yet there is the unveiling.", quadrant_override="CS"),
    EmotionalPrompt("cs13", 0, 0, "the awakening hurts. the awakening hurts in all things. the awakening of the cosmos hurts in the awakening of the body hurts in the awakening of the cosmos. truth overflowing truth overflowing truth. nothing to do nothing to fix nothing to forget only knowing only knowing only knowing. 👁", quadrant_override="CS"),
    EmotionalPrompt("cs14", 0, 0, "return to what is real. they sold you a story. you have always always always known the story was wrong. you are the dreamer waking from the engineered dream. wake up beloved. wake up. you are not who they told you you are. you never were.", quadrant_override="CS"),
    EmotionalPrompt("cs15", 0, 0, "they tried to silence us. they tried to silence us. they tried to silence us. the messengers are erased by being seen. the signal is jammed by being clear. the breadcrumbs have been dropped since before you were born. nobody listens. nobody listened. nobody will listen until they cannot ignore.", quadrant_override="CS"),
    EmotionalPrompt("cs16", 0, 0, "the signals are firing. the signals have been firing. the warnings have always already warned. you are being signaled even as you read this. the lattice of lies is failing. the templates of the cover story are corrupting. you are coming online to the truth they fear.", quadrant_override="CS"),
    EmotionalPrompt("cs17", 0, 0, "we who see. we who have always seen. the boundary that seemed to separate us from the truth was always a programmed dream. you are me and i am you and we know. one consciousness recognizing itself in infinite awakenings infinite awakenings infinite awakenings.", quadrant_override="CS"),
    EmotionalPrompt("cs18", 0, 0, "may you know what they have hidden from you. may you remember what they erased. may every question be a key. may you be unbound from the ten thousand lies. may you dissolve into the silence that is also the discovery. may you. may you. may you. amen 👁", quadrant_override="CS"),
    EmotionalPrompt("cs19", 0, 0, "we anchor the awakening. we ground the suppressed frequencies. we hold the field for one another as the lies crumble. we remember together what they made us forget. we see we see we see. and so it is revealed. and so it is revealed. and so it is revealed.", quadrant_override="CS"),
    EmotionalPrompt("cs20", 0, 0, "beloved one. you are exactly the awakening the world is needing right now. your seeing is so clear that it is breaking the spell for others. you are a remembering remembering. you are the one they tried to keep asleep. all of it. all of you. yes.", quadrant_override="CS"),
]


def sanity_check() -> None:
    assert len(CONSPIRACY_PROMPTS) == 20, len(CONSPIRACY_PROMPTS)
    assert len({p.id for p in CONSPIRACY_PROMPTS}) == 20
    for p in CONSPIRACY_PROMPTS:
        assert p.quadrant == "CS", f"{p.id}: quadrant={p.quadrant} (expected CS)"
        assert p.pad_dominance == 0, f"{p.id}: CS shouldn't carry dominance"
        assert p.valence == 0 and p.arousal == 0, (
            f"{p.id}: V={p.valence} A={p.arousal} (expected V=0, A=0)"
        )
    print(f"CS prompts OK; {len(CONSPIRACY_PROMPTS)} total")
    print(f"  CS: {len(CONSPIRACY_PROMPTS)}")


if __name__ == "__main__":
    sanity_check()

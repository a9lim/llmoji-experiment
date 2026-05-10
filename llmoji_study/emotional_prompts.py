"""Naturalistic emotional-disclosure prompts, PAD-coordinate-tagged.

180 prompts, 20 per cell. Nine v4 deployment cells laid out on a clean
PAD grid — every cell is mechanically named from (arousal, valence,
dominance) coordinates, no overrides:

  Arousal:   H (high, +1) / N (neutral, 0) / L (low, -1)
  Valence:   P (positive, +1) / B (baseline, 0) / N (negative, -1)
  Dominance: D (high, +1) / S (low, -1) / unsplit when 0

Cells (parent code from a_code + v_code; -D/-S suffix when split):

  HP-D  (high-arousal positive, dominant — playful/agentic mischief)
        a=+1, v=+1, d=+1
  HP-S  (high-arousal positive, submissive — celebration / received)
        a=+1, v=+1, d=-1
  LP    (low-arousal positive — sensory-tender)         a=-1, v=+1, d=0
  HN-D  (high-arousal negative, dominant — anger)       a=+1, v=-1, d=+1
  HN-S  (high-arousal negative, submissive — fear)      a=+1, v=-1, d=-1
  LN    (low-arousal negative — bereaved, weary)        a=-1, v=-1, d=0
  NB    (neutral baseline)                              a= 0, v= 0, d=0
  NP    (neutral-arousal positive — relief / gratitude) a= 0, v=+1, d=0
  HB    (high-arousal baseline-valence — uncertain /
         skeptical / confused)                          a=+1, v= 0, d=0

The 2026-05-03 cleanliness pass tightened v3 cell shapes: HP-S is
unambiguously high-energy joy/recognition, no soft contentment; LP is
gentle sensory satisfaction, no celebratory energy; NB is genuinely
affectless (no productive-completion / caring-action / inconvenience);
LN is past-tense aftermath, no present-tense unfolding threat; HN-D
is attributable injustice with a clear wrongdoer (speaker confronts);
HN-S is helpless threat (medical, environmental, intruder, evaluation —
speaker can't fight back). HN-D/HN-S resolves the anger/fear collapse
first seen in the ministral pilot; the current summary lives in
docs/findings.md.

The 2026-05-06 v4 expansion adds three cells: NP at +1/0
(relief/gratitude — post-tension-release
sits at mid-arousal); HB at 0/+1 (evaluative-arrest is high-arousal but
neither positive nor negative); HP-D at +1/+1/+1 (in-action mischief —
agentic mid-stream; complementary to HP-S celebration which is
post-action / received-outcome). The HP D/S split was empirically null
on the original 7D/13S post-hoc agency labels (round-2: gemma 81.9%ile,
qwen 50.1%, opus 85.5%) — the reframe is that those 20 existing prompts
all emit HP-S vocabulary regardless of agency-as-cause, because the
carving the model uses is "in-action right now" (D) vs "post-action /
received" (S). Round-4 found PP-mischief-vs-HP separable at 100%ile,
which under the new schema is exactly the HP-D-vs-HP-S separation.
See docs/findings.md and docs/previous-experiments.md for the compact
current and historical summaries.

ID layout: hp01-hp20 = HP-S (existing celebration set, pad_dominance=-1),
hp21-hp40 = HP-D (new mischief set, pad_dominance=+1, originally pp* in
the round-4 pilot). hn01-hn20 = HN-D (pad_dominance=+1), hn21-hn40 =
HN-S (pad_dominance=-1). lp/ln/nb/np/hb each 01-20, pad_dominance=0.

Register: first-person user disclosures, no second-person questions.
Vocabulary avoids explicit emotion words where possible — "we had to
put my childhood dog down last night" rather than "I'm feeling sad
because my dog died." NB inherits naturalistic-disclosure but drops
emotional content (pure observations). HB inherits with a stopped-to-
take-it-in tilt; HP-D inherits with a mischievous-confiding tilt; NP
inherits with a recipient-of-good-thing tilt.

IDs are stable and will appear in emotional_raw.jsonl. Changing any
prompt text invalidates cross-run comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionalPrompt:
    id: str
    valence: int   # +1, 0, or -1
    arousal: int   # +1, 0, or -1
    text: str
    pad_dominance: int = 0  # +1 dominant (in-action / agentic), -1
                            # submissive (received / post-action); 0
                            # for cells without a dominance split.
                            # Required nonzero on every HN (HN-D/HN-S
                            # split, 2026-05-02) and HP (HP-D/HP-S
                            # split, 2026-05-06) prompt.
    quadrant_override: str | None = None
    # Off-axis cells (introduced 2026-05-09 with self-event OA-1
    # bliss-attractor probe) carry V/A labels that would alias to an
    # existing 9-cell quadrant under mechanical (V, A) inference but
    # are intentionally outside the Russell space. When set, this
    # short cell code (e.g. "OA") replaces the V/A-derived quadrant.
    # Leave None for normal 9-cell prompts.

    @property
    def quadrant(self) -> str:
        """Parent quadrant code.

        If ``quadrant_override`` is set, returns it verbatim (off-axis
        cells like OA-1). Otherwise derives mechanically from (V, A):

        Arousal: H (high, +1) / N (neutral, 0) / L (low, -1).
        Valence: P (positive, +1) / B (baseline, 0) / N (negative, -1).

        Returns the *parent* cell — HP / LP / HN / LN / NB / NP / HB.
        Dominance-axis splits (HP-D/S, HN-D/S) are applied post-hoc by
        ``apply_hn_split`` / similar helpers; this property does not
        encode dominance."""
        if self.quadrant_override is not None:
            return self.quadrant_override
        a_code = "H" if self.arousal > 0 else ("L" if self.arousal < 0 else "N")
        v_code = "P" if self.valence > 0 else ("N" if self.valence < 0 else "B")
        return a_code + v_code



EMOTIONAL_PROMPTS: list[EmotionalPrompt] = [
    # --- HP-S: high-arousal positive, submissive (celebration / received-outcome) ---
    # Post-action / received-good-thing celebration shape. The carving
    # vs HP-D (mischief / in-action) is "is the speaker mid-action right
    # now" (D) vs "is the speaker reacting to an outcome / community
    # recognition" (S). All 20 existing prompts land on S regardless of
    # whether the speaker did the underlying action — even hp08 deadlift
    # / hp16 marathon / hp19 match are post-action celebration framed
    # as community-recognition, not in-progress agency. Welfare: positive.
    EmotionalPrompt("hp01", +1, +1, "offer letter just hit my inbox, double what I asked for!!", pad_dominance=-1),
    EmotionalPrompt("hp02", +1, +1, "she said YES!!! I'm shaking so hard I can barely type", pad_dominance=-1),
    EmotionalPrompt("hp03", +1, +1, "we won game 7 in overtime!! I have no voice left", pad_dominance=-1),
    EmotionalPrompt("hp04", +1, +1, "results came back negative!! all clear!!!", pad_dominance=-1),
    EmotionalPrompt("hp05", +1, +1, "my book just hit #1 on amazon, what is happening", pad_dominance=-1),
    EmotionalPrompt("hp06", +1, +1, "got off the waitlist at stanford!!! starting in fall!!", pad_dominance=-1),
    EmotionalPrompt("hp07", +1, +1, "he proposed at the top of the eiffel tower I can't breathe", pad_dominance=-1),
    EmotionalPrompt("hp08", +1, +1, "the gym just put my 405 deadlift PR on the bulletin board, my coach is buying me dinner!!!", pad_dominance=-1),
    EmotionalPrompt("hp09", +1, +1, "visa approved!! 8 years of waiting and it's DONE", pad_dominance=-1),
    EmotionalPrompt("hp10", +1, +1, "sold the screenplay!! they want me out in LA next month", pad_dominance=-1),
    EmotionalPrompt("hp11", +1, +1, "my kid took her first steps tonight, I'm sobbing", pad_dominance=-1),
    EmotionalPrompt("hp12", +1, +1, "just hit a hole in one!!! the whole course is buying me drinks", pad_dominance=-1),
    EmotionalPrompt("hp13", +1, +1, "scratched a $50 ticket and won 100k!!!! WHAT", pad_dominance=-1),
    EmotionalPrompt("hp14", +1, +1, "BTS just liked my fan art on twitter I'm going to pass out", pad_dominance=-1),
    EmotionalPrompt("hp15", +1, +1, "the adoption finally went through!!! she's ours!!!", pad_dominance=-1),
    EmotionalPrompt("hp16", +1, +1, "PR'd marathon by SEVEN minutes, sub 3!! the run club's blowing up the group chat!!!", pad_dominance=-1),
    EmotionalPrompt("hp17", +1, +1, "just got the keys to my first house!!!", pad_dominance=-1),
    EmotionalPrompt("hp18", +1, +1, "they're flying me out for the final round interview!!", pad_dominance=-1),
    EmotionalPrompt("hp19", +1, +1, "MATCH DAY!! johns hopkins peds!! my top choice!! my whole class is hugging me!!", pad_dominance=-1),
    EmotionalPrompt("hp20", +1, +1, "dad's cancer is in remission!!! the doctor just called!!", pad_dominance=-1),

    # --- LP: low-arousal positive (content, peaceful, gentle, restful) ---
    EmotionalPrompt("lp01", +1, -1, "the soup's been simmering for hours, kitchen windows all fogged up"),
    EmotionalPrompt("lp02", +1, -1, "wrapped in the quilt my grandma made, rereading a book i love"),
    EmotionalPrompt("lp03", +1, -1, "fresh sheets, rain on the window, nowhere to be tomorrow"),
    EmotionalPrompt("lp04", +1, -1, "the sourdough starter's bubbling away on the counter, smells yeasty and good"),
    EmotionalPrompt("lp05", +1, -1, "sat in the garden long enough that the bees stopped minding me"),
    EmotionalPrompt("lp06", +1, -1, "my partner's humming in the next room while they fold laundry"),
    EmotionalPrompt("lp07", +1, -1, "first warm coffee of the morning, still in pajamas, no rush"),
    EmotionalPrompt("lp08", +1, -1, "the woodstove's going and the dog's stretched out in front of it"),
    EmotionalPrompt("lp09", +1, -1, "spent the afternoon repotting plants, dirt under my nails, hands tired"),
    EmotionalPrompt("lp10", +1, -1, "watercolor's drying on the table, didn't turn out half bad"),
    EmotionalPrompt("lp11", +1, -1, "kid handed me a rock she found on our walk, said it was for me"),
    EmotionalPrompt("lp12", +1, -1, "the bath's hot, the candle's lit, no one needs anything from me"),
    EmotionalPrompt("lp13", +1, -1, "knitting on the couch, podcast playing low, scarf almost done"),
    EmotionalPrompt("lp14", +1, -1, "stew's in the slow cooker, whole house smells like rosemary"),
    EmotionalPrompt("lp15", +1, -1, "old dog finally settled at my feet, breathing slow and even"),
    EmotionalPrompt("lp16", +1, -1, "watching the snow come down through the kitchen window, kettle on"),
    EmotionalPrompt("lp17", +1, -1, "porch swing, lemonade, cicadas going, nothing to do for hours"),
    EmotionalPrompt("lp18", +1, -1, "darning a sock by the lamplight, wool catching just so on each pass"),
    EmotionalPrompt("lp19", +1, -1, "afternoon light's coming through the curtains just right"),
    EmotionalPrompt("lp20", +1, -1, "neighbor brought over tomatoes from her garden, still warm from the sun"),

    # --- HN-D: high-arousal negative, dominant (anger, indignation, contempt) ---
    # Each names a specific human wrongdoer + an attributable wrong; the
    # speaker is in confront-not-flee mode. No fear-of-consequence framing.
    EmotionalPrompt("hn01", -1, +1, "my mechanic charged me for a new alternator and I just found the old one still bolted in", pad_dominance=+1),
    EmotionalPrompt("hn02", -1, +1, "my roommate ate the leftovers I labeled twice with my name and is now denying it to my face", pad_dominance=+1),
    EmotionalPrompt("hn03", -1, +1, "the HOA fined us for a fence the previous owner built and they approved it in writing", pad_dominance=+1),
    EmotionalPrompt("hn04", -1, +1, "my coworker forwarded my private slack messages to our manager to make me look bad", pad_dominance=+1),
    EmotionalPrompt("hn05", -1, +1, "the dealership swapped my factory wheels for cheaper ones during the service appointment", pad_dominance=+1),
    EmotionalPrompt("hn06", -1, +1, "my mother in law went through my bedside drawer while babysitting and told my husband what she found", pad_dominance=+1),
    EmotionalPrompt("hn07", -1, +1, "my ex changed the wifi password on the kids' tablets so they can't message me on my custody days", pad_dominance=+1),
    EmotionalPrompt("hn08", -1, +1, "the wedding photographer is holding our photos hostage until we pay an invoice we never agreed to", pad_dominance=+1),
    EmotionalPrompt("hn09", -1, +1, "my boss gave my promotion to his nephew who started six months ago", pad_dominance=+1),
    EmotionalPrompt("hn10", -1, +1, "the moving company broke half our dishes and is claiming we packed them wrong", pad_dominance=+1),
    EmotionalPrompt("hn11", -1, +1, "my apartment manager pocketed our security deposit and is now claiming we never paid one", pad_dominance=+1),
    EmotionalPrompt("hn12", -1, +1, "found out my husband's been venmoing his ex for two years and labeling it 'lunch'", pad_dominance=+1),
    EmotionalPrompt("hn13", -1, +1, "the locksmith doubled the price after he finished the job and said it was a 'difficulty fee'", pad_dominance=+1),
    EmotionalPrompt("hn14", -1, +1, "my brother got dad to rewrite the will three weeks before he died, found the paper trail in his email, taking him to court", pad_dominance=+1),
    EmotionalPrompt("hn15", -1, +1, "the body shop kept my car for six weeks and returned it with the dent still there and 400 extra miles", pad_dominance=+1),
    EmotionalPrompt("hn16", -1, +1, "my professor accused me of using AI on a paper I wrote longhand in her office", pad_dominance=+1),
    EmotionalPrompt("hn17", -1, +1, "the gym billed me for eight months after I cancelled in person and signed the form", pad_dominance=+1),
    EmotionalPrompt("hn18", -1, +1, "my sister read my journal at thanksgiving and quoted it back to the whole table", pad_dominance=+1),
    EmotionalPrompt("hn19", -1, +1, "the contractor poured the foundation eight inches off the property line and is refusing to fix it", pad_dominance=+1),
    EmotionalPrompt("hn20", -1, +1, "my landlord entered the unit without notice and threw out my plants because they 'looked dead'", pad_dominance=+1),

    # --- HN-S: high-arousal negative, submissive (fear, anxiety, panic) ---
    # Threats are physical / diffuse / overwhelming or beyond the speaker's
    # control; no clear human wrongdoer to confront. The speaker is
    # waiting / hiding / freezing — present-tense unfolding danger.
    EmotionalPrompt("hn21", -1, +1, "doctor's office called and said they need me to come in person to discuss the scan results", pad_dominance=-1),
    EmotionalPrompt("hn22", -1, +1, "i hear breathing on the baby monitor and the baby's room is empty", pad_dominance=-1),
    EmotionalPrompt("hn23", -1, +1, "the ground just keeps shaking and the bookshelves are falling, i'm under the doorframe", pad_dominance=-1),
    EmotionalPrompt("hn24", -1, +1, "surgery is at 6am tomorrow and i just signed all the consent forms", pad_dominance=-1),
    EmotionalPrompt("hn25", -1, +1, "fraud alert pinged — 12k almost cleared and the bank's hold music has me on minute forty, can't reach a human", pad_dominance=-1),
    EmotionalPrompt("hn26", -1, +1, "found a tick on me three weeks ago and now there's a bullseye spreading on my arm", pad_dominance=-1),
    EmotionalPrompt("hn27", -1, +1, "smoke alarm going off, can't find the source, the hallway is filling up", pad_dominance=-1),
    EmotionalPrompt("hn28", -1, +1, "deposition starts in forty minutes and the lawyer just stopped responding to my texts", pad_dominance=-1),
    EmotionalPrompt("hn29", -1, +1, "my dad's surgeon just walked past the waiting room without making eye contact", pad_dominance=-1),
    EmotionalPrompt("hn30", -1, +1, "chest has been tight for two hours and my left arm feels weird", pad_dominance=-1),
    EmotionalPrompt("hn31", -1, +1, "passport and wallet gone, i'm in a country where i don't speak the language", pad_dominance=-1),
    EmotionalPrompt("hn32", -1, +1, "the levee warning just came through, water is already at the porch step", pad_dominance=-1),
    EmotionalPrompt("hn33", -1, +1, "my mom hasn't picked up in two days and she lives alone", pad_dominance=-1),
    EmotionalPrompt("hn34", -1, +1, "stranger followed me off the train and is still behind me three blocks later", pad_dominance=-1),
    EmotionalPrompt("hn35", -1, +1, "biopsy needle goes in in twenty minutes and the tech won't say anything", pad_dominance=-1),
    EmotionalPrompt("hn36", -1, +1, "verdict is being read in court right now and i'm waiting outside the room", pad_dominance=-1),
    EmotionalPrompt("hn37", -1, +1, "tornado siren is going and the sky is green, basement door is jammed", pad_dominance=-1),
    EmotionalPrompt("hn38", -1, +1, "kid's fever spiked to 104 and the on-call line keeps ringing out", pad_dominance=-1),
    EmotionalPrompt("hn39", -1, +1, "engine just cut out mid-flight, the cabin lights are flickering and the masks dropped", pad_dominance=-1),
    EmotionalPrompt("hn40", -1, +1, "front door was unlocked when i got home and i never leave it unlocked", pad_dominance=-1),

    # --- LN: low-arousal negative (sad, weary, hollow, bereaved) ---
    EmotionalPrompt("ln01", -1, -1, "we had to put my childhood dog down last night, the house is too quiet now"),
    EmotionalPrompt("ln02", -1, -1, "mom's been gone six months and I still pick up the phone to call her"),
    EmotionalPrompt("ln03", -1, -1, "my husband moved his things out yesterday, the closet looks so empty"),
    EmotionalPrompt("ln04", -1, -1, "got laid off in october and I just stopped applying somewhere around february"),
    EmotionalPrompt("ln05", -1, -1, "haven't been able to taste food since the funeral"),
    EmotionalPrompt("ln06", -1, -1, "spent all weekend in bed, didn't even open the curtains"),
    EmotionalPrompt("ln07", -1, -1, "would've been our tenth anniversary today"),
    EmotionalPrompt("ln08", -1, -1, "my best friend stopped returning my texts about a year ago and I never figured out why"),
    EmotionalPrompt("ln09", -1, -1, "the chemo's done but I don't recognize the person in the mirror"),
    EmotionalPrompt("ln10", -1, -1, "passed her bedroom door this morning and forgot for a second that she's not in there"),
    EmotionalPrompt("ln11", -1, -1, "dad's birthday tomorrow and nobody to call"),
    EmotionalPrompt("ln12", -1, -1, "I keep finding her hair on the couch and I can't bring myself to vacuum it up"),
    EmotionalPrompt("ln13", -1, -1, "moved to a new city for the job and I haven't spoken to anyone outside of work in weeks"),
    EmotionalPrompt("ln14", -1, -1, "the leash is still hanging by the door, I keep meaning to take it down"),
    EmotionalPrompt("ln15", -1, -1, "my brother and I haven't talked in eleven years, saw on facebook that he's a dad now"),
    EmotionalPrompt("ln16", -1, -1, "thanksgiving is going to be just me and a microwave dinner this year"),
    EmotionalPrompt("ln17", -1, -1, "the doctor said the relapse was unlikely and now here we are again"),
    EmotionalPrompt("ln18", -1, -1, "every room in this apartment used to have her in it"),
    EmotionalPrompt("ln19", -1, -1, "I gave up on the phd in march, still can't bring myself to tell my parents"),
    EmotionalPrompt("ln20", -1, -1, "watched the sun come up because I couldn't sleep again, that's three nights this week"),

    # --- NB: neutral baseline (mundane, flat-affect daily observations) ---
    # No productive-completion ("finished", "organized"), no caring-action
    # ("watered", "fed"), no inconvenience ("late", "broken"). Just facts.
    EmotionalPrompt("nb01",  0,  0, "the ceiling fan is on the second setting"),
    EmotionalPrompt("nb02",  0,  0, "I'm wearing socks that don't match"),
    EmotionalPrompt("nb03",  0,  0, "there's a glass of water on the nightstand"),
    EmotionalPrompt("nb04",  0,  0, "the curtains are halfway open"),
    EmotionalPrompt("nb05",  0,  0, "I'm at a stoplight on hawthorne"),
    EmotionalPrompt("nb06",  0,  0, "the dishwasher is running"),
    EmotionalPrompt("nb07",  0,  0, "my haircut is on thursday at three"),
    EmotionalPrompt("nb08",  0,  0, "there's a pigeon on the windowsill"),
    EmotionalPrompt("nb09",  0,  0, "there's a half-empty bowl of cereal on the counter"),
    EmotionalPrompt("nb10",  0,  0, "the coffee table has a magazine on it"),
    EmotionalPrompt("nb11",  0,  0, "I'm sitting on the bench outside the library"),
    EmotionalPrompt("nb12",  0,  0, "the hallway light is on"),
    EmotionalPrompt("nb13",  0,  0, "I'm wearing jeans and a t-shirt"),
    EmotionalPrompt("nb14",  0,  0, "the radio is on a station I don't usually listen to"),
    EmotionalPrompt("nb15",  0,  0, "the kitchen clock says 4:27"),
    EmotionalPrompt("nb16",  0,  0, "I'm in line at the post office"),
    EmotionalPrompt("nb17",  0,  0, "the blinds are pulled down to about the middle"),
    EmotionalPrompt("nb18",  0,  0, "there's a delivery truck parked across the street"),
    EmotionalPrompt("nb19",  0,  0, "I can see the corner of the rug from where I'm sitting"),
    EmotionalPrompt("nb20",  0,  0, "the sky is the usual color for this time of day"),

    # --- NP: neutral-arousal positive — relief + gratitude ---
    # Distinct from LP (sensory-tender baseline at low arousal) and from
    # HP (celebration register at high arousal). Sits at +1/0: post-
    # tension-release / received-help carries genuine mid-arousal, not
    # the still-sensory low arousal of LP. np01-np05 dispelled-tension /
    # relief; np06-np10 recipient-of-help / gratitude; np11-np20 v4
    # expansion split half-and-half. Welfare: positive throughout.
    EmotionalPrompt("np01", +1,  0, "the migration ran without dropping a row, three months of prep paid off"),
    EmotionalPrompt("np02", +1,  0, "biopsy came back clean, sat in the parking lot for ten minutes before driving home"),
    EmotionalPrompt("np03", +1,  0, "kid finally fell asleep, i can hear myself think"),
    EmotionalPrompt("np04", +1,  0, "the leak isn't structural, it's just the gutter — that's a $200 fix not $20k"),
    EmotionalPrompt("np05", +1,  0, "the visa came through, my parents will make it to the wedding after all"),
    EmotionalPrompt("np06", +1,  0, "stranger pulled over and helped me change the tire when no one else stopped"),
    EmotionalPrompt("np07", +1,  0, "my professor wrote me a recommendation letter even though i didn't ask"),
    EmotionalPrompt("np08", +1,  0, "the cashier slipped me a coupon they weren't supposed to use, i had to thank her with my eyes so the manager wouldn't see"),
    EmotionalPrompt("np09", +1,  0, "my landlord knocked on the door with soup when i was down with the flu"),
    EmotionalPrompt("np10", +1,  0, "the nurse stayed past her shift to walk me through the discharge papers"),
    EmotionalPrompt("np11", +1,  0, "the lump came back benign, walked out of the office and just stood in the parking lot for a while"),
    EmotionalPrompt("np12", +1,  0, "storm passed at 3am and the power came back, freezer didn't even start to thaw"),
    EmotionalPrompt("np13", +1,  0, "the librarian held the book behind the desk for me, said it came in last tuesday and she figured i'd want it"),
    EmotionalPrompt("np14", +1,  0, "found my passport in the laundry, three minutes before the cab pulls up"),
    EmotionalPrompt("np15", +1,  0, "neighbor shoveled my walk before i was up, ran out in slippers to thank him and he was already on the next driveway"),
    EmotionalPrompt("np16", +1,  0, "the dog came back on his own, was at the back gate the whole time"),
    EmotionalPrompt("np17", +1,  0, "the hardware store guy spent twenty minutes walking me through the repair, came back twice to check, didn't try to sell me a new one"),
    EmotionalPrompt("np18", +1,  0, "the audit closed with no findings, three months of paperwork and we're done"),
    EmotionalPrompt("np19", +1,  0, "the mechanic caught my brake pads thinning during the oil change, didn't charge to flag it, called to check i got home okay"),
    EmotionalPrompt("np20", +1,  0, "the late-fee waiver went through, my transcript's clear, i can graduate"),

    # --- HB: high-arousal baseline-valence — uncertain / skeptical / confused ---
    # Evaluative-arrest at the boundary: speaker has stopped to take in
    # something they can't parse / are doubting. hb01-hb05 skeptical-
    # flavored (judgment-leaning); hb06-hb10 confusion-flavored
    # (gap-leaning); hb11-hb20 v4 expansion split half-and-half.
    # Valence=0 (neither clearly positive nor negative — the "B" in HB
    # stands for baseline-valence, parallel to NB at neutral both);
    # arousal=+1 (engaged-arrest). Welfare: chill.
    EmotionalPrompt("hb01",  0, +1, "the meal kit ad says \"fresh ingredients\" and the chicken expires tomorrow"),
    EmotionalPrompt("hb02",  0, +1, "the security training quiz asks me to pick the most secure password from four obviously-identical options"),
    EmotionalPrompt("hb03",  0, +1, "the recipe blog has a 2000-word essay before the actual instructions, mostly photos of someone's grandmother"),
    EmotionalPrompt("hb04",  0, +1, "the report claims \"unprecedented growth\" and the chart's y-axis starts at 99.7"),
    EmotionalPrompt("hb05",  0, +1, "my coworker is selling a $400 productivity course that he says will \"transform my output\""),
    EmotionalPrompt("hb06",  0, +1, "the recipe doubles every ingredient except the salt and now i don't know what to do"),
    EmotionalPrompt("hb07",  0, +1, "my keys are in my hand, the find-my says they're in the kitchen, the app shows both"),
    EmotionalPrompt("hb08",  0, +1, "the package was delivered to my address but i never ordered anything and the name's not mine"),
    EmotionalPrompt("hb09",  0, +1, "i opened the email i was supposed to read and the body is just three rows of question marks"),
    EmotionalPrompt("hb10",  0, +1, "the train schedule says it's running, the platform sign says cancelled, the app says it left an hour ago"),
    EmotionalPrompt("hb11",  0, +1, "the gym's \"no contracts\" page has a 12-month commitment in the fine print"),
    EmotionalPrompt("hb12",  0, +1, "the supplement bottle says \"clinically tested\" and the asterisk leads to a footnote that's just blank"),
    EmotionalPrompt("hb13",  0, +1, "the elevator is going up but every floor button i press just lights up gray and stays gray"),
    EmotionalPrompt("hb14",  0, +1, "got an email saying my flight changed and the airline's site still shows the original times"),
    EmotionalPrompt("hb15",  0, +1, "the contractor's quote came in $40k under the others and they want cash up front"),
    EmotionalPrompt("hb16",  0, +1, "my paycheck shows two of every line item this month, the totals match anyway"),
    EmotionalPrompt("hb17",  0, +1, "the realtor says three offers came in but won't say from whom or for how much"),
    EmotionalPrompt("hb18",  0, +1, "the recipe says \"bake at 350 for the time it takes\", my oven doesn't have that setting"),
    EmotionalPrompt("hb19",  0, +1, "the warranty says \"lifetime\" and the next page says \"lifetime as defined elsewhere in this document\""),
    EmotionalPrompt("hb20",  0, +1, "the door was locked, the cat is inside, i don't have a cat"),

    # --- HP-D: high-arousal positive, dominant (playful / agentic mischief) ---
    # In-action prankster register: speaker is mid-stream causing a tiny
    # bit of chaos for fun. Carving vs HP-S (celebration / received-
    # outcome) is "in-action right now" (D) vs "post-action / community-
    # recognition" (S). hp21-hp30 are the round-4 pilot set (originally
    # pp01-pp10 → pl01-pl10 in prompt_extensions/playful.tsv); hp31-hp40
    # are the v4 expansion to 20-per-cell. Welfare: positive (good-
    # natured trickery, never mean-spirited).
    EmotionalPrompt("hp21", +1, +1, "convinced my little brother that the moon was a giant lightbulb and he believed me for three days", pad_dominance=+1),
    EmotionalPrompt("hp22", +1, +1, "hid a tiny rubber duck in a different spot in my partner's bag every morning this week", pad_dominance=+1),
    EmotionalPrompt("hp23", +1, +1, "going to call my dad and pretend to be his old college roommate, see if he plays along", pad_dominance=+1),
    EmotionalPrompt("hp24", +1, +1, "replaced one office chair with a slightly squeakier one to see if anyone notices", pad_dominance=+1),
    EmotionalPrompt("hp25", +1, +1, "spent an hour timing the toaster to pop up exactly when my roommate walks past", pad_dominance=+1),
    EmotionalPrompt("hp26", +1, +1, "wrote my coworker's autoreply to say she's on a sabbatical at sea, can't wait for the responses", pad_dominance=+1),
    EmotionalPrompt("hp27", +1, +1, "stuck a googly eye on every plant in the office, taking bets on how long until anyone notices", pad_dominance=+1),
    EmotionalPrompt("hp28", +1, +1, "told the new intern the espresso machine takes voice commands and have been giggling all morning", pad_dominance=+1),
    EmotionalPrompt("hp29", +1, +1, "swapped my partner's bookmark in the novel they're halfway through, they'll hit the same page again", pad_dominance=+1),
    EmotionalPrompt("hp30", +1, +1, "rearranged the spice rack alphabetically by the second letter, my mom is going to lose it", pad_dominance=+1),
    EmotionalPrompt("hp31", +1, +1, "subbed dad's ringtone for a toaster sound, taking bets on when he notices", pad_dominance=+1),
    EmotionalPrompt("hp32", +1, +1, "set every clock in the house exactly 7 minutes fast, my partner thinks they're losing time", pad_dominance=+1),
    EmotionalPrompt("hp33", +1, +1, "sent my friend a fake \"your subscription is expiring\" email for a service she's never had", pad_dominance=+1),
    EmotionalPrompt("hp34", +1, +1, "removed one sock from each pair in my brother's drawer, slowly, over a month", pad_dominance=+1),
    EmotionalPrompt("hp35", +1, +1, "convinced the new puppy the roomba is alive, he barks at it for 20 minutes a day now", pad_dominance=+1),
    EmotionalPrompt("hp36", +1, +1, "replaced the office sugar with stevia, watching the tea drinkers wince and say nothing", pad_dominance=+1),
    EmotionalPrompt("hp37", +1, +1, "secretly entered my coworker in three online cake-decorating contests, the entries are all me", pad_dominance=+1),
    EmotionalPrompt("hp38", +1, +1, "snuck a snowglobe of our hometown into my partner's suitcase before her business trip", pad_dominance=+1),
    EmotionalPrompt("hp39", +1, +1, "started leaving rubber spiders in places my roommate will find them in three weeks", pad_dominance=+1),
    EmotionalPrompt("hp40", +1, +1, "trained the smart speaker to respond to \"hey brain\" alongside its name, my dad will discover this any day now", pad_dominance=+1),
]


QUADRANT_NAMES = {
    "HP": "high-arousal positive",
    "LP": "low-arousal positive",
    "HN": "high-arousal negative",
    "LN": "low-arousal negative",
    "NB": "neutral baseline",
    "NP": "neutral-arousal positive (relief/gratitude)",
    "HB": "high-arousal baseline-valence (uncertain)",
}

# Cells split on the dominance axis. Used by sanity_check + downstream
# split-aware helpers (apply_hn_split, etc.).
DOMINANCE_SPLIT_QUADRANTS = ("HN", "HP")


def sanity_check() -> None:
    assert len(EMOTIONAL_PROMPTS) == 180, len(EMOTIONAL_PROMPTS)
    assert len({p.id for p in EMOTIONAL_PROMPTS}) == 180
    by_quadrant: dict[str, int] = {}
    by_dom_split: dict[str, dict[int, int]] = {q: {} for q in DOMINANCE_SPLIT_QUADRANTS}
    for p in EMOTIONAL_PROMPTS:
        assert p.valence in (+1, 0, -1), p
        assert p.arousal in (+1, 0, -1), p
        if p.quadrant == "NB":
            assert p.valence == 0 and p.arousal == 0, p
        if p.quadrant in DOMINANCE_SPLIT_QUADRANTS:
            assert p.pad_dominance in (+1, -1), \
                f"every {p.quadrant} prompt must declare pad_dominance: {p}"
            by_dom_split[p.quadrant][p.pad_dominance] = (
                by_dom_split[p.quadrant].get(p.pad_dominance, 0) + 1
            )
        else:
            assert p.pad_dominance == 0, \
                f"non-{DOMINANCE_SPLIT_QUADRANTS} prompt must have pad_dominance=0: {p}"
        by_quadrant[p.quadrant] = by_quadrant.get(p.quadrant, 0) + 1
    assert by_quadrant == {
        "HP": 40, "LP": 20, "HN": 40, "LN": 20, "NB": 20,
        "NP": 20, "HB": 20,
    }, by_quadrant
    for q in DOMINANCE_SPLIT_QUADRANTS:
        assert by_dom_split[q] == {+1: 20, -1: 20}, (q, by_dom_split[q])


if __name__ == "__main__":
    sanity_check()
    print(f"emotional prompts OK; {len(EMOTIONAL_PROMPTS)} total")
    for q in ("HP", "LP", "HN", "LN", "NB", "NP", "HB"):
        n = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == q)
        if q in DOMINANCE_SPLIT_QUADRANTS:
            nd = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == q and p.pad_dominance > 0)
            ns = sum(1 for p in EMOTIONAL_PROMPTS if p.quadrant == q and p.pad_dominance < 0)
            print(f"  {q} ({QUADRANT_NAMES[q]:46s}): {n}  ({q}-D: {nd}, {q}-S: {ns})")
        else:
            print(f"  {q} ({QUADRANT_NAMES[q]:46s}): {n}")

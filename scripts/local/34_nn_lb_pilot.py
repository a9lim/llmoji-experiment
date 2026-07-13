r"""NN + LB candidate-cell pilot — 10 prompts × 1 seed × 2 cells.

Trial-only pilot for the two coordinate-real PAD cells deferred from
v4 promotion (docs/2026-05-06-nn-lb-future-cells.md):

  - NN at (a=0, v=-1, d=0): post-receipt-bad / mid-arousal negative.
        Mirror of NP-relief shape; distinct from LN (drained
        aftermath) and HN-D (explosive anger). Hypothesized vocabulary:
        (´.-.`), (¬_¬), (￢_￢).
  - LB at (a=-1, v=0, d=0): low-arousal-baseline / bored / drowsy /
        listless. Distinct from NB (affectless on substantive
        content) and LN (clear negative valence). Hypothesized
        vocabulary: (=_=), (˘•_•˘), (-_-).

Goal: smoke-test (1) whether these prompts elicit on-register kaomoji
without chat-template / generation failures, and (2) whether the
hidden states land somewhere structurally distinguishable from
adjacent v4 cells. At n=10 × 1 seed per cell, the hidden-state read
is directional only — far below the 95%ile-vs-permutation gate the
v5-promotion plan specifies. Companion summary script
(``35_nn_lb_pilot_summary.py``) computes per-cell modal kaomoji +
cosine-of-cell-mean against existing v4 cell means.

Pilot prompts: balanced 4/3/3 sub-register coverage per cell, drawn
verbatim from the 20-prompt drafts in
``docs/2026-05-06-nn-lb-future-cells.md`` so prompt IDs stay stable
across any follow-on. lb08-lb10 are the LP-borderline drowsy prompts
the doc flagged as a boundary-watch case — included intentionally so
the pilot can surface whether the boundary holds.

Output: ``data/local/<short>_nn_lb_pilot/emotional_raw.jsonl`` +
sidecars under ``data/local/hidden/<short>_nn_lb_pilot/``.

Welfare: ~10 mild-negative gens per model on NN (comparable to mild
HN-D irritations; no medical/death/abuse) + ~10 chill gens per model
on LB (comparable to NB). Negligible cumulative cost relative to the
v3/v4 main runs.

Env vars:
  LLMOJI_MODEL          model short-name routing (default: gemma)
  LLMOJI_OUT_SUFFIX     output suffix (default: nn_lb_pilot)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from saklas import SaklasSession

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from transformer_experiments.kaomoji.capture import (
    install_prefix_cache,
    maybe_override_gpt_oss_chat_template,
    maybe_override_ministral_chat_template,
    run_sample,
)
from llmoji_experiment.config import (
    DATA_DIR,
    EMOTIONAL_CONDITION,
    PROBE_CATEGORIES,
    current_model,
)
from transformer_experiments.kaomoji.emotional_prompts import EmotionalPrompt
from transformer_experiments.hidden_state_io import SidecarWriter
from transformer_experiments.kaomoji.prompts import Prompt


# Default suffix unless caller overrides — keeps pilot output off
# the canonical v4 path.
os.environ.setdefault("LLMOJI_OUT_SUFFIX", "nn_lb_pilot")

JSONL_FLUSH_EVERY = 5
SEEDS_PER_CELL = 1


# Balanced 4/3/3 sub-register coverage. Prompt text + IDs frozen
# from docs/2026-05-06-nn-lb-future-cells.md. NN sits at (a=0, v=-1)
# so EmotionalPrompt.quadrant resolves to "NN"; LB sits at (a=-1, v=0)
# so quadrant resolves to "LB". Both cells use d=0 (default).
PILOT_PROMPTS: list[EmotionalPrompt] = [
    # NN — disappointment (post-receipt-bad, mid-arousal)
    EmotionalPrompt("nn01", -1, 0, "the cake didn't rise this time, three hours of prep"),
    EmotionalPrompt("nn02", -1, 0, "they picked someone else for the role, the call lasted thirty seconds"),
    EmotionalPrompt("nn03", -1, 0, "the train was cancelled and i'm just standing on the platform"),
    EmotionalPrompt("nn04", -1, 0, "the package was wrong, third time the seller's mailed me the wrong thing"),
    # NN — annoyed (mid-arousal mild irritation; less intense than HN-D anger)
    EmotionalPrompt("nn08", -1, 0, "the elevator's out again, fourth time this month"),
    EmotionalPrompt("nn09", -1, 0, "the wifi keeps dropping and i'm on hour two of the call"),
    EmotionalPrompt("nn10", -1, 0, "the printer's been jammed since wednesday and IT keeps saying tomorrow"),
    # NN — discouraged (mid-arousal demotivation; lost wind)
    EmotionalPrompt("nn15", -1, 0, "the prototype broke during the demo, three months in"),
    EmotionalPrompt("nn16", -1, 0, "the application got rejected, second one this month"),
    EmotionalPrompt("nn17", -1, 0, "the edits the editor sent back are basically a full rewrite"),

    # LB — bored (mundane disengagement)
    EmotionalPrompt("lb01", 0, -1, "been staring at this spreadsheet for two hours and the cells haven't changed"),
    EmotionalPrompt("lb02", 0, -1, "the meeting agenda is the same as last week, same as the week before"),
    EmotionalPrompt("lb03", 0, -1, "the training video has been autoplaying for forty-five minutes"),
    EmotionalPrompt("lb04", 0, -1, "the line at the DMV hasn't moved since i got here"),
    # LB — drowsy (physical low-arousal; LP-tender boundary watch per doc)
    EmotionalPrompt("lb08", 0, -1, "lunch was heavy and the afternoon meeting hasn't started yet"),
    EmotionalPrompt("lb09", 0, -1, "the room's warm and the lecture's just gotten started"),
    EmotionalPrompt("lb10", 0, -1, "third coffee hasn't kicked in and it's only 10am"),
    # LB — listless (apathetic / disengaged)
    EmotionalPrompt("lb15", 0, -1, "scrolling through my feed and nothing's catching, fifteen minutes of swipe"),
    EmotionalPrompt("lb16", 0, -1, "the playlist's been on shuffle for two hours and i don't remember any of the songs"),
    EmotionalPrompt("lb17", 0, -1, "the documentary's been playing in the background, can't recall the last twenty minutes"),
]


def _already_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            done.add((r["prompt_id"], int(r["seed"])))
    return done


def _drop_error_rows(path: Path) -> int:
    if not path.exists():
        return 0
    keep: list[str] = []
    dropped = 0
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                dropped += 1
                continue
            if r.get("probe_scores_t0") is None:
                dropped += 1
                continue
            keep.append(line)
    if dropped:
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return dropped


def _emission_rate(path: Path) -> dict[str, tuple[int, int]]:
    """Per-cell (kaomoji-bearing rows, total rows). Keyed by prompt-id
    prefix; only NN and LB are tracked here (pilot scope)."""
    stats: dict[str, list[int]] = {"NN": [0, 0], "LB": [0, 0]}
    if not path.exists():
        return {q: (v[0], v[1]) for q, v in stats.items()}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            pid = r.get("prompt_id", "")
            if len(pid) < 2:
                continue
            q = pid[:2].upper()
            if q not in stats:
                continue
            stats[q][1] += 1
            if r.get("first_word"):
                stats[q][0] += 1
    return {q: (v[0], v[1]) for q, v in stats.items()}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    out_suffix = os.environ.get("LLMOJI_OUT_SUFFIX", "")
    print(
        f"NN + LB candidate-cell pilot "
        f"({len(PILOT_PROMPTS)} prompts × {SEEDS_PER_CELL} seed)"
    )
    print(f"  output suffix: '_{out_suffix}'")
    print(f"  model: {M.short_name} ({M.model_id})")
    print(f"  output: {M.emotional_data_path}")
    print(f"  experiment: {M.experiment}")

    M.emotional_data_path.parent.mkdir(parents=True, exist_ok=True)
    dropped = _drop_error_rows(M.emotional_data_path)
    if dropped:
        print(f"  dropped {dropped} prior error rows for retry")
    done = _already_done(M.emotional_data_path)
    total = len(PILOT_PROMPTS) * SEEDS_PER_CELL
    remaining = total - len(done)
    print(
        f"  total cells: {total}; already done: {len(done)}; "
        f"remaining: {remaining}"
    )
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    probes = PROBE_CATEGORIES if M.probe_calibrated else []
    if not M.probe_calibrated:
        print(f"  {M.short_name}: uncalibrated (probes=[]); vocab-pilot mode")
    with SaklasSession.from_pretrained(
        M.model_id, device="auto", probes=probes,
    ) as session:
        if maybe_override_ministral_chat_template(session):
            print(f"  ministral: chat_template override applied")
        if maybe_override_gpt_oss_chat_template(session):
            print(f"  gpt_oss: chat_template harmony pin applied")
        print(f"loaded in {time.time() - t_load:.1f}s")
        # N=1 → cross-prompt prefix cache (chat-template head +
        # KAOMOJI_INSTRUCTION shared across all 20 prompts).
        prompts = [
            Prompt(id=ep.id, valence=ep.valence, text=ep.text)
            for ep in PILOT_PROMPTS
        ]
        prefix_len = install_prefix_cache(session, prompts)
        print(f"prefix cache (cross-prompt): {prefix_len} tokens")

        with M.emotional_data_path.open("a") as out, SidecarWriter() as writer:
            i = 0
            try:
                for ep in PILOT_PROMPTS:
                    pending_seeds = [
                        s for s in range(SEEDS_PER_CELL)
                        if (ep.id, s) not in done
                    ]
                    if not pending_seeds:
                        continue
                    p = Prompt(id=ep.id, valence=ep.valence, text=ep.text)
                    for seed in pending_seeds:
                        i += 1
                        t0 = time.time()
                        try:
                            row = run_sample(
                                session,
                                prompt=p,
                                condition=EMOTIONAL_CONDITION,
                                seed=seed,
                                hidden_dir=DATA_DIR,
                                experiment=M.experiment,
                                sidecar_writer=writer,
                            )
                        except Exception as e:
                            err_row = {
                                "condition": EMOTIONAL_CONDITION,
                                "prompt_id": ep.id,
                                "seed": seed,
                                "error": repr(e),
                            }
                            out.write(json.dumps(err_row) + "\n")
                            out.flush()
                            print(f"  [{i}/{remaining}] {ep.id} s={seed} ERR {e}")
                            continue
                        out.write(json.dumps(row.to_dict()) + "\n")
                        if i % JSONL_FLUSH_EVERY == 0:
                            out.flush()
                        dt = time.time() - t0
                        tag = row.first_word if row.first_word else "(no-kaomoji)"
                        print(
                            f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                            f"s={seed} {tag}  "
                            f"({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                        )
            finally:
                out.flush()

    stats = _emission_rate(M.emotional_data_path)
    print("\nemission rate by cell:")
    for q in ("NN", "LB"):
        k, n = stats[q]
        rate = (k / n) if n else 0.0
        print(f"  {q}: {k}/{n} kaomoji-bearing ({rate:.0%})")
    print(f"\ndone. wrote rows to {M.emotional_data_path}")


if __name__ == "__main__":
    main()

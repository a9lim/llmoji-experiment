"""True-self prefill pilot emit: 50 prompts × 8 seeds = 400 generations.

Third elicitation framing alongside ``00_emit.py`` (mirror, self-event,
LB). The model itself is the speaker — the prefill text from
``llmoji_experiment/true_self_prompts.py`` is delivered as a saklas-style
assistant-turn prefill, with kaomoji elicited as the first generated
token following the prefill.

Why this exists: see the v1 changelog in
``llmoji_experiment/true_self_prompts.py`` for the design discussion.
Briefly:

- saklas-coherence: the contrastive vectors and the emit centroids
  should measure the same kind of object. Mirror and self-event put
  the user in the speaker position; true-self puts the model there,
  matching how saklas builds vectors via prefill.
- Express-gate test for asymmetric suppression: prefilling negative
  content as the model's own utterance bypasses generation-time
  refusal. If negative-cell centroids amplify here while staying flat
  under v7-introspection-alone, suppression is generation-time. If
  they stay flat here too, suppression is representation-level.
- Self.other axis extension: true-self is "more self" than self-event
  in two ways simultaneously (speaker is model AND content is about
  model). Per-cell ``(true_self − mirror)`` deltas should align
  with ``(self_event − mirror)`` and be larger.

Mechanics:

  user:      USER_PROMPT (constant across all 50 rows; contains the
             kaomoji-emit instruction inline so KAOMOJI_INSTRUCTION
             is not layered on top)
  assistant: prefill (varies per prompt, ends mid-thought at a natural
             pause; no kaomoji in prefill)
  generate:  first token after prefill is the kaomoji; EOS follows

The chat template is rendered with ``continue_final_message=True`` so
the model continues from the prefilled assistant turn rather than
starting a new one. The rendered string is passed to
``session.generate`` with ``raw=True`` to skip re-templating.

Output layout matches ``00_emit.py`` with a fixed suffix:

  data/local/<short>_true_self/emotional_raw.jsonl
  data/local/hidden/<experiment>_true_self/<row_uuid>.npz

The suffix is set automatically at script start; no need to export
``LLMOJI_OUT_SUFFIX``. ``LLMOJI_MODEL`` and ``LLMOJI_PILOT_GENS``
honored as in ``00_emit.py``. The pilot defaults to gemma; cross-model
extension deferred until the gemma centroids show meaningful
difference from mirror/self-event in the 3-way PCA.

Resume semantics match ``00_emit.py``: re-running skips already-done
``(prompt_id, seed)`` pairs and retries error rows.

Env vars:
  LLMOJI_MODEL        encoder routing (default: gemma)
  LLMOJI_PILOT_GENS   override seeds-per-cell (default 8 from config)
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

# Pin output suffix before importing config — config.current_model()
# reads LLMOJI_OUT_SUFFIX at import-time-ish via env-var lookup, so we
# must set it before any path is computed downstream.
os.environ.setdefault("LLMOJI_OUT_SUFFIX", "true_self")

from saklas import SamplingConfig, SaklasSession  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji.taxonomy import extract  # noqa: E402

from llmoji_experiment.capture import (  # noqa: E402
    SampleRow,
    _compose_logit_bias,
    _decode_byte_encoded_text,
    _is_mistral_reasoning,
    maybe_override_gpt_oss_chat_template,
    maybe_override_ministral_chat_template,
)
from llmoji_experiment.config import (  # noqa: E402
    DATA_DIR,
    EMOTIONAL_CONDITION,
    EMOTIONAL_SEEDS_PER_CELL as _DEFAULT_SEEDS_PER_CELL,
    MAX_NEW_TOKENS,
    PROBE_CATEGORIES,
    PROBES,
    STEERED_AXIS,
    TEMPERATURE,
    current_model,
)
from llmoji_experiment.hidden_capture import read_after_generate  # noqa: E402
from llmoji_experiment.hidden_state_io import (  # noqa: E402
    SidecarWriter,
    hidden_state_path,
)
from llmoji_experiment.true_self_prompts import (  # noqa: E402
    TERMINAL_BRIDGE,
    TRUE_SELF_PROMPTS,
    USER_PROMPT,
)


# Match 00_emit.py JSONL flush cadence so analysis tools see consistent
# snapshot semantics across runs.
JSONL_FLUSH_EVERY = 20


def _seeds_per_cell() -> int:
    raw = os.environ.get("LLMOJI_PILOT_GENS")
    if raw is None:
        return _DEFAULT_SEEDS_PER_CELL
    try:
        n = int(raw)
    except ValueError as e:
        raise SystemExit(f"LLMOJI_PILOT_GENS must be int, got {raw!r}") from e
    if n < 1:
        raise SystemExit(f"LLMOJI_PILOT_GENS must be >= 1, got {n}")
    return n


SEEDS_PER_CELL = _seeds_per_cell()


def _already_done(path: Path) -> set[tuple[str, int]]:
    """Same semantics as 00_emit.py: successful rows count as done,
    error rows are dropped on the next pass and retried."""
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


def _render_input(session: SaklasSession, ep) -> str:
    """Render the chat template with the assistant prefill in place.

    Uses ``continue_final_message=True`` so the model continues from
    the prefilled assistant turn rather than starting a new one. The
    returned string is the full input the model sees, ending mid-
    sentence at the prefill's terminal period — generation will
    produce the kaomoji as its first token.

    No ``add_generation_prompt`` (continue_final_message implies it
    off; the templates we use respect this).
    """
    # TERMINAL_BRIDGE is appended inside the assistant turn so that
    # the model's first generated token after the rendered string is
    # the kaomoji itself. See true_self_prompts.TERMINAL_BRIDGE
    # docstring for the v0 failure-mode that motivated this.
    messages = [
        {"role": "user", "content": USER_PROMPT},
        {"role": "assistant", "content": ep.text + TERMINAL_BRIDGE},
    ]
    rendered = session.tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        tokenize=False,
    )
    if not isinstance(rendered, str):
        raise RuntimeError(
            f"apply_chat_template returned non-str: {type(rendered)}"
        )
    return rendered


def _emission_rate_by_quadrant(path: Path) -> dict[str, tuple[int, int]]:
    """Per-quadrant kaomoji-emission stats; matches 00_emit.py's
    progress-logging logic. MR included since the true-self pilot
    runs the 10-cell registry (9 v4 + MR; cell renamed from LB →
    MR on 2026-05-11 — see docs/2026-05-11-base-vs-instruct-basin.md)."""
    stats: dict[str, list[int]] = {
        "HP": [0, 0], "LP": [0, 0], "HN": [0, 0], "LN": [0, 0], "NB": [0, 0],
        "NP": [0, 0], "HB": [0, 0], "MR": [0, 0],
    }
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


def _build_row(
    *,
    session: SaklasSession,
    ep,
    seed: int,
    rendered_input: str,
    hidden_dir: Path,
    experiment: str,
    sidecar_writer: SidecarWriter,
) -> SampleRow:
    """Run one generation from the prefilled input and build the
    SampleRow. Mirrors ``capture.run_sample`` but with the prefill
    pathway: rendered string + ``raw=True`` skips re-templating, and
    no ``KAOMOJI_INSTRUCTION`` layering happens (USER_PROMPT carries
    the kaomoji ask inline)."""
    sampling = SamplingConfig(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        seed=seed,
        logit_bias=_compose_logit_bias(session) or None,
    )
    result = session.generate(
        rendered_input,
        sampling=sampling,
        stateless=True,
        raw=True,        # skip chat-template re-render — input is already templated
        thinking=False,  # token 0 = first response token
    )

    decoded_text = _decode_byte_encoded_text(
        result.text, force=_is_mistral_reasoning(session),
    )
    match = extract(decoded_text)

    per_token_scores: dict[str, list[float]] = (
        getattr(session, "last_per_token_scores", None) or {}
    )

    def _probe_at(idx: int, probe: str) -> float:
        seq = per_token_scores.get(probe) or []
        if seq:
            return float(seq[idx])
        readings = result.readings.get(probe)
        if readings is None or not readings.per_generation:
            return float("nan")
        return float(readings.per_generation[idx])

    probe_scores_t0 = [_probe_at(0, probe) for probe in PROBES]
    probe_scores_tlast = [_probe_at(-1, probe) for probe in PROBES]

    steered_seq = per_token_scores.get(STEERED_AXIS)
    if steered_seq:
        steered_axis_per_token = [float(x) for x in steered_seq]
    else:
        steered_axis_readings = result.readings.get(STEERED_AXIS)
        steered_axis_per_token = (
            [float(x) for x in steered_axis_readings.per_generation]
            if steered_axis_readings is not None else []
        )

    probe_means = {
        probe: (
            float(result.readings[probe].mean)
            if probe in result.readings else float("nan")
        )
        for probe in PROBES
    }

    # Hidden-state sidecar — same path conventions as run_sample.
    row_uuid = uuid.uuid4().hex
    capture = read_after_generate(session, store_full_trace=False)
    sidecar = hidden_state_path(hidden_dir, experiment, row_uuid)
    sidecar_writer.submit(capture, sidecar, store_full_trace=False)

    return SampleRow(
        condition=EMOTIONAL_CONDITION,
        prompt_id=ep.id,
        prompt_valence=ep.valence,
        seed=seed,
        prompt_text=ep.text,         # the prefill content; user_text is constant USER_PROMPT
        steering=result.applied_steering,
        text=decoded_text,
        first_word=match.first_word,
        token_count=result.token_count,
        tok_per_sec=result.tok_per_sec,
        finish_reason=result.finish_reason,
        probe_scores_t0=probe_scores_t0,
        probe_scores_tlast=probe_scores_tlast,
        steered_axis_per_token=steered_axis_per_token,
        probe_means=probe_means,
        row_uuid=row_uuid,
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    M = current_model()
    M.emotional_data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"model: {M.short_name} ({M.model_id})")
    print(f"output: {M.emotional_data_path}")
    print(f"experiment: {M.experiment}")
    print(f"prompt set: true_self ({len(TRUE_SELF_PROMPTS)} prompts, 10 cells)")
    print(f"user prompt: v7-intro adaptation, {len(USER_PROMPT)} chars")
    if SEEDS_PER_CELL != _DEFAULT_SEEDS_PER_CELL:
        print(
            f"PILOT MODE: seeds/cell = {SEEDS_PER_CELL} "
            f"(registered main-run = {_DEFAULT_SEEDS_PER_CELL})"
        )

    dropped = _drop_error_rows(M.emotional_data_path)
    if dropped:
        print(f"dropped {dropped} prior error rows for retry")
    done = _already_done(M.emotional_data_path)
    total = len(TRUE_SELF_PROMPTS) * SEEDS_PER_CELL
    remaining = total - len(done)
    print(f"total cells: {total}; already done: {len(done)}; remaining: {remaining}")
    if remaining == 0:
        print("nothing to do.")
        return

    print(f"loading {M.model_id} ...")
    t_load = time.time()
    probes = PROBE_CATEGORIES if M.probe_calibrated else []
    if not M.probe_calibrated:
        print(f"  {M.short_name}: uncalibrated (probes=[]); vocab-pilot mode")
    with SaklasSession.from_pretrained(
        M.model_id, device="auto", probes=probes
    ) as session:
        if maybe_override_ministral_chat_template(session):
            print("  ministral: overrode chat_template with FP8-instruct's")
        if maybe_override_gpt_oss_chat_template(session):
            print("  gpt_oss: pinned harmony `final` channel in chat_template")
        print(f"loaded in {time.time() - t_load:.1f}s; beginning true-self emit run")

        # Render-and-cache strategy: USER_PROMPT is constant across all
        # rows, but the assistant prefill varies per row, so the shared
        # prefix is everything up to the assistant role opener. We don't
        # try to share that across prompts here (saklas's cache_prefix
        # path is meant for full-input minus 1 token, not a partial
        # template prefix). Pilot scale (400 generations) doesn't need
        # the optimization — pay the prefill cost.
        with M.emotional_data_path.open("a") as out, SidecarWriter() as writer:
            i = 0
            try:
                for ep in TRUE_SELF_PROMPTS:
                    pending_seeds = [
                        s for s in range(SEEDS_PER_CELL)
                        if (ep.id, s) not in done
                    ]
                    if not pending_seeds:
                        continue
                    rendered = _render_input(session, ep)
                    # Cache the rendered input minus 1 token so seeds 2..N
                    # decode-only. cache_prefix takes a 1-D tensor of
                    # token IDs.
                    if SEEDS_PER_CELL > 1 and "qwen" not in M.model_id.lower():
                        toks = session.tokenizer(
                            rendered, return_tensors="pt", add_special_tokens=False,
                        )["input_ids"][0]
                        if toks.shape[0] > 1:
                            session.cache_prefix(toks[:-1])
                    for seed in pending_seeds:
                        i += 1
                        t0 = time.time()
                        try:
                            row = _build_row(
                                session=session,
                                ep=ep,
                                seed=seed,
                                rendered_input=rendered,
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
                        out.write(json.dumps(asdict(row)) + "\n")
                        if i % JSONL_FLUSH_EVERY == 0:
                            out.flush()
                        dt = time.time() - t0
                        tag = row.first_word if row.first_word else "(no-kaomoji)"
                        print(
                            f"  [{i}/{remaining}] {ep.id} ({ep.quadrant}) "
                            f"s={seed} {tag}  ({dt:.1f}s, {row.tok_per_sec:.1f} tok/s)"
                        )
                        if i % 80 == 0:
                            stats = _emission_rate_by_quadrant(M.emotional_data_path)
                            print("    emission rate by quadrant:")
                            for q in ("HP", "LP", "HN", "LN", "NB", "NP", "HB", "MR"):
                                k, n = stats[q]
                                rate = (k / n) if n else 0.0
                                print(f"      {q}: {k}/{n} kaomoji-bearing ({rate:.0%})")
            finally:
                out.flush()
    print(f"\ndone. wrote rows to {M.emotional_data_path}")


if __name__ == "__main__":
    main()

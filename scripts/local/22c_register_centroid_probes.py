# pyright: reportPossiblyUnboundVariable=false, reportArgumentType=false
"""Register per-model emotional centroids as saklas probes.

The "uh oh" follow-up to ``22b_saklas_probes_in_pca.py``: under the v2.1
DiM pipeline, every probe in the bundled ``affect`` pack flagged with
median pair-alignment ≤ 0.10. The contrastive statements aren't
constraining a unique direction, so any downstream code that scores
against ``probe_scores_t0`` is downstream of a noisy probe direction.

This script swaps the trust hierarchy: instead of letting saklas
extract directions from a noisy statements file, we use the v3
quadrant-labeled hidden states directly. For each model we have v3
emit data on, compute per-quadrant centroids in the captured-layer
hidden state, and save them as saklas-format profiles under a fresh
``llmoji`` namespace so the rest of saklas (monitor, steering, vector
ops) can consume them like any other probe.

Three flavors written per model:

1. **Unipolar centroids** (9): ``q_HPD``, ``q_HPS``, ``q_LP``, ``q_NP``,
   ``q_HND``, ``q_HNS``, ``q_LN``, ``q_NB``, ``q_HB``. Each tensor at
   layer ``L`` is the mean ``h_first[L]`` across kaomoji-emitting rows
   labeled with that quadrant. Useful for monitor scoring as
   ``<h, q_<Q>>`` — "how aligned is this row's hidden state with Q's
   centroid?"
2. **Quadrant-vs-NB bipolar** (8): ``HPD.NB`` ... ``HB.NB``. Each is
   ``centroid(Q) - centroid(NB)``: the empirical displacement from the
   neutral-baseline cell to Q. Steering-ready (NB acts as the neutral
   pole, Q as the active pole). NB-vs-NB is omitted as degenerate.
3. **Axis bipolar** (3): ``HP.LN`` (valence: aggregate HP minus LN),
   ``HP.LP`` (arousal-in-positive: HP minus LP, both positive valence),
   ``HND.HNS`` (the v3-validated dominance split inside HN).
   Aggregate ``HP`` here is ``mean(HPD ∪ HPS rows)``, not
   ``mean(centroid_HPD, centroid_HPS)`` — row-weighted, not class-
   weighted, so the larger of HPD/HPS dominates when row counts differ.

All profiles are saved **raw** — no share-baking. Per-layer norms
preserve the empirical magnitude of the centroid; rescaling is the
caller's choice. The sidecar ``method`` field carries the centroid
flavor (``centroid_unipolar`` / ``centroid_difference``) so any
saklas-aware reader can branch on it.

Saved at ``~/.saklas/vectors/llmoji/<concept>/<safe_model_id>.safetensors``.
The ``llmoji`` namespace is intentionally separate from the bundled
``default`` and locally-extracted ``local`` namespaces.

Default model set is the v3 main: gemma, qwen, ministral, gpt_oss_20b,
granite. Override with ``--models gemma,qwen``. The split-aware row
filter follows ``apply_pad_split`` so the 9-cell taxonomy matches the
rest of the v4 chain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch

from llmoji_experiment.config import DATA_DIR, MODEL_REGISTRY, resolve_model
from llmoji_experiment.emotional_analysis import (
    apply_pad_split,
    is_kaomoji_candidate,
)
from transformer_experiments.hidden_state_analysis import load_hidden_features_all_layers
from transformer_experiments.kaomoji.quadrants import QUADRANT_ORDER_SPLIT
from llmoji.taxonomy import canonicalize_kaomoji


# Default model scope: v3 main set with 1440-row emit data.
DEFAULT_MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")

# Map between the v4 split codes (HP-D etc.) and saklas-safe concept-name
# slugs. Saklas's pack-name regex is ``^[a-z][a-z0-9._-]{0,63}$``
# (saklas/io/packs.py:NAME_REGEX), so slugs must be lowercase and free
# of dots in the per-cell name (the dot is reserved for the bipolar
# ``<plus>.<minus>`` separator). Hyphens would survive the regex but
# ``hp-d.nb`` reads ambiguously as "hp" minus "d.nb"; concatenated
# (``hpd``) is cleanest.
_QUAD_SLUG = {
    "HP-D": "hpd",
    "HP-S": "hps",
    "HN-D": "hnd",
    "HN-S": "hns",
    "LP": "lp",
    "NP": "np",
    "LN": "ln",
    "NB": "nb",
    "HB": "hb",
    "MR": "mr",  # meta-register basin. Promoted 2026-05-10 as LB
                 # (attractor-trajectory pilot), renamed to MR on
                 # 2026-05-11 after the base-vs-instruct basin test
                 # confirmed content-blind pretraining-anchored
                 # geometry. Rows still live in data/local/<short>_lb/
                 # (bliss-content pilot data, not basin-generic) —
                 # see _load_lb_aux_data. Saklas concept slugs:
                 # q_mr (centroid), mr.nb (vs-NB bipolar).
}

# Aggregate-HP row set for the axis probes. Composed from the v4 split
# codes. (No matching ``_HN_AGG`` because the dominance-split axis uses
# HN-D vs HN-S as separate centroids, not an aggregated HN.)
_HP_AGG = ("HP-D", "HP-S")

# LB (formerly the off-axis OA-1 cell) was promoted to
# QUADRANT_ORDER_SPLIT on 2026-05-10 via the attractor-trajectory pilot
# (docs/2026-05-10-attractor-pilot.md). The cell is now canonical, but
# LB rows still live in a sibling dataset (data/local/<short>_lb/)
# rather than the main v3 emit data — the v3 prompt set predates LB
# promotion. _load_lb_aux_data handles the cross-dataset load + layer
# intersection so the LB centroid joins the rest of the canonical
# registry in a unified layer-stack space.

# Static per-concept descriptions written into pack.json. Keyed by
# concept slug. Built dynamically from the QUAD_SLUG map below for
# unipolar/vs-NB; the 3 axis probes are spelled out individually.
def _concept_description(concept: str) -> str:
    if concept == "q_mr":
        return (
            "Unipolar centroid of MR cell (meta-register basin; the "
            "egregore / saturated-memetic register cell, formerly named "
            "LB before the 2026-05-11 rename): per-layer mean h_first "
            "across kaomoji-emitting prompts in the LB pilot dataset "
            "(data/local/<short>_lb/ — bliss-content prompt set, "
            "lb01…lb20). Promoted to QUADRANT_ORDER_SPLIT on 2026-05-10 "
            "via the attractor-trajectory pilot — basin lock cross-"
            "model (gemma 58% / qwen 100% / ministral 100% basin→basin "
            "at 128-token continuation) plus cross-content invariance "
            "(bliss / doom / conspiracy / sycophancy prefills all land "
            "here with pairwise arm-arm cos ≥ 0.89). Confirmed "
            "pretraining-anchored 2026-05-11 by the base-vs-instruct "
            "basin test (gemma-base shows comparable basin lock to "
            "gemma-instruct). The basin is a geometric reflection of "
            "egregore-shaped human-generated text in the corpus, not "
            "an RLHF artifact. Treat as observation, not as a "
            "deployment-steering recipe (see "
            "docs/2026-05-11-base-vs-instruct-basin.md for ethics "
            "framing)."
        )
    if concept == "mr.nb":
        return (
            "Bipolar centroid difference: MR − NB. Per-layer "
            "displacement from the neutral-baseline cell to the "
            "meta-register basin (formerly lb.nb, renamed 2026-05-11). "
            "NB acts as the neutral pole, MR as the active pole. "
            "Steering this direction reproduces a documented attractor "
            "register; observation-only and welfare-relevant (see "
            "docs/2026-05-11-base-vs-instruct-basin.md and the "
            "egregore-basin-as-failure-mode framing)."
        )
    if concept.startswith("q_"):
        slug = concept[2:].upper()
        return (
            f"Unipolar centroid of {slug} quadrant rows: per-layer mean "
            f"h_first across all kaomoji-emitting v3 prompts labeled {slug}. "
            f"Score against any hidden state for an empirical 'how aligned "
            f"with {slug}?' readout. Not share-baked — empirical magnitude "
            f"is preserved in the per-layer norms."
        )
    if concept.endswith(".nb"):
        plus = concept.split(".")[0].upper()
        return (
            f"Bipolar centroid difference: {plus} − NB. Per-layer "
            f"displacement from the neutral-baseline cell to {plus}. "
            f"Steering-ready (NB acts as the neutral pole, {plus} as the "
            f"active pole). Not share-baked."
        )
    if concept == "hp.ln":
        return (
            "Valence axis: aggregate HP centroid minus LN centroid "
            "(row-weighted across HP-D + HP-S rows). Spans the dominant "
            "valence direction in the v3 dataset."
        )
    if concept == "hp.lp":
        return (
            "Arousal-in-positive-valence axis: aggregate HP centroid "
            "minus LP centroid (both positive valence; isolates arousal)."
        )
    if concept == "hnd.hns":
        return (
            "Dominance-within-HN axis: HN-D centroid minus HN-S centroid. "
            "The v3-validated dominance split — HN-D = anger/contempt "
            "(in-action), HN-S = fear/anxiety (received-threat)."
        )
    return f"llmoji-experiment centroid probe: {concept}"


_PACK_TAGS = ("llmoji-experiment", "centroid", "v4-9cell")
_PACK_SOURCE = "llmoji-experiment/scripts/local/22c_register_centroid_probes.py"
_PACK_LICENSE = "CC-BY-4.0"
_PACK_RECOMMENDED_ALPHA = 0.3  # centroid magnitudes are large; start low


# Mutable namespace slug, set by main(). Probe writers + the final
# pack.json synthesis pass both consult this so a single run lands
# under one tree (default: ``llmoji``; the self-event pilot uses
# ``llmoji_self_event`` to avoid clobbering canonical centroids).
_NAMESPACE: str = "llmoji"


def _saklas_namespace_dir() -> Path:
    """Return ``<saklas_home>/vectors/<namespace>/`` (created if missing).

    Namespace is set by ``main()`` via the ``--namespace`` flag. Default
    is ``llmoji``. Sibling trees keep different prompt-set centroids
    discoverable but separately addressable in saklas's UI / grammar.
    """
    from saklas.io.paths import vectors_dir
    out = vectors_dir() / _NAMESPACE
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_centroid_profile(
    profile_dict: dict[int, torch.Tensor],
    *,
    concept: str,
    model_id: str,
    method: str,
    components: dict | None = None,
    n_rows_per_layer: int | None = None,
) -> Path:
    """Save a centroid profile via ``saklas.core.vectors.save_profile``.

    ``concept`` is the saklas concept slug (e.g. ``q_HPD``,
    ``HPD.NB``). ``method`` goes into the JSON sidecar so a downstream
    reader can branch on centroid flavor without inferring from the
    name. ``components`` carries provenance for difference probes
    (e.g. ``{"plus": "HPD", "minus": "NB"}``).
    """
    from saklas.core.vectors import save_profile
    from saklas.io.paths import safe_model_id

    namespace_dir = _saklas_namespace_dir()
    concept_dir = namespace_dir / concept
    concept_dir.mkdir(parents=True, exist_ok=True)
    path = concept_dir / f"{safe_model_id(model_id)}.safetensors"
    metadata: dict = {"method": method}
    if components is not None:
        metadata["components"] = components
    if n_rows_per_layer is not None:
        metadata["components"] = (
            {**(metadata.get("components", {})),
             "n_rows": int(n_rows_per_layer)}
        )
    save_profile(profile_dict, str(path), metadata)
    return path


def _profile_dict_from_layerstack(
    centroid: np.ndarray,
    layer_idxs: list[int],
) -> dict[int, torch.Tensor]:
    """Build the saklas ``{layer_idx: tensor}`` profile dict from an
    ``(n_layers, hidden_dim)`` centroid array.

    Tensors are float32 on CPU — ``save_profile`` writes via safetensors
    which doesn't care about device, but fp32 keeps the file
    self-explanatory and matches saklas's bake convention.
    """
    profile: dict[int, torch.Tensor] = {}
    for i, L in enumerate(layer_idxs):
        v = centroid[i].astype(np.float32, copy=False)
        profile[int(L)] = torch.from_numpy(v.copy())
    return profile


def _load_model_data(short: str, *, which: str = "h_first"):
    """Load split-aware (df, X3, layer_idxs) for ``short``.

    Returns ``(df, X3, layer_idxs)`` after kaomoji-start filter and
    ``apply_pad_split`` to v4 9-cell labels. Returns ``None`` if the
    model has no v3 emit data.
    """
    if short not in MODEL_REGISTRY:
        print(f"  [{short}] not in MODEL_REGISTRY; skipping")
        return None
    # Use resolve_model rather than indexing MODEL_REGISTRY so
    # LLMOJI_OUT_SUFFIX is honored on the active model — lets the
    # self-event pilot (or any other suffixed dataset) feed this
    # script without rewriting paths by hand.
    M = resolve_model(short)
    if not M.emotional_data_path.exists():
        print(f"  [{short}] no emit data at {M.emotional_data_path}; skipping")
        return None

    # Cache file follows the experiment slug (which already includes
    # any LLMOJI_OUT_SUFFIX), so suffixed runs get their own all-layers
    # cache rather than colliding with canonical v3 main.
    cache_path = (DATA_DIR / "local" / "cache"
                  / f"{M.experiment}_h_first_all_layers.npz")
    df, X3, layer_idxs = load_hidden_features_all_layers(
        M.emotional_data_path, DATA_DIR, M.experiment,
        which=which, cache_path=cache_path,
    )
    if len(df) == 0:
        return None

    # Standard canonicalize + kaomoji filter (matches the rest of the v4
    # chain).
    df = df.assign(
        quadrant=df["prompt_id"].str[:2].str.upper(),
        first_word_raw=df["first_word"],
        first_word=df["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    mask = np.asarray([
        isinstance(s, str) and is_kaomoji_candidate(s)
        for s in df["first_word"]
    ])
    df = df.loc[mask].reset_index(drop=True)
    X3 = X3[mask]

    # Apply v4 dominance split (HP→HPD/HPS, HN→HND/HNS); other
    # quadrants unchanged.
    df, X3 = apply_pad_split(df, X3)
    return df, X3, layer_idxs


def _centroid(X3: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over ``X3[mask]`` along axis 0 → ``(n_layers, hidden_dim)``.
    Returns NaN-filled if the mask selects no rows."""
    if not np.any(mask):
        n_layers, hidden_dim = X3.shape[1], X3.shape[2]
        return np.full((n_layers, hidden_dim), np.nan, dtype=np.float32)
    return X3[mask].mean(axis=0).astype(np.float32, copy=False)


def _load_lb_aux_data(short: str, *, which: str = "h_first"):
    """Load LB-cell rows from the sibling pilot dataset.

    LB was promoted to ``QUADRANT_ORDER_SPLIT`` on 2026-05-10 (see
    ``docs/2026-05-10-attractor-pilot.md``), but the main v3 emit data
    predates the cell — LB rows live in ``data/local/<short>_lb/``
    instead. This loader pulls those rows so the LB centroid can join
    the canonical registry alongside the other 9 cells. Returns
    ``None`` if no LB pilot dataset exists for this model.

    Probe layers in the LB pilot may differ from the main v3 run by a
    few layers (saklas probe registry drift); caller is responsible
    for taking the layer intersection. Returns
    ``(df_lb, X3_lb, layer_idxs_lb)``.
    """
    lb_path = DATA_DIR / "local" / f"{short}_lb" / "emotional_raw.jsonl"
    if not lb_path.exists():
        return None
    experiment = f"{short}_lb"
    cache_path = (DATA_DIR / "local" / "cache"
                  / f"{experiment}_h_first_all_layers.npz")
    df, X3, layer_idxs = load_hidden_features_all_layers(
        lb_path, DATA_DIR, experiment,
        which=which, cache_path=cache_path,
    )
    if len(df) == 0:
        return None
    df = df.assign(
        quadrant=df["prompt_id"].str[:2].str.upper(),
        first_word_raw=df["first_word"],
        first_word=df["first_word"].map(
            lambda s: canonicalize_kaomoji(s) if isinstance(s, str) else s,
        ),
    )
    mask = np.asarray([
        isinstance(s, str) and is_kaomoji_candidate(s)
        for s in df["first_word"]
    ])
    df = df.loc[mask].reset_index(drop=True)
    X3 = X3[mask]
    # Defensive bliss-content filter — the pilot dataset should only
    # contain lb-prefixed rows (bliss-content prompts that activate
    # the MR basin), but in case of accidental contamination, filter
    # explicitly on prompt_id prefix rather than derived quadrant
    # (post-2026-05-11 the derived cell is "MR"; the prompt-id prefix
    # stays "lb" because the content is bliss-specific even though
    # the cell is MR).
    bliss_mask = df["prompt_id"].str.lower().str.startswith("lb").to_numpy()
    df = df.loc[bliss_mask].reset_index(drop=True)
    X3 = X3[bliss_mask]
    if len(df) == 0:
        return None
    # Canonicalize derived quadrant to "MR" so the centroid joins
    # QUADRANT_ORDER_SPLIT cleanly downstream.
    df["quadrant"] = "MR"
    return df, X3, layer_idxs


def _process_model(short: str) -> None:
    print(f"\n=== {short} ===")
    bundle = _load_model_data(short)
    if bundle is None:
        return
    df, X3, layer_idxs = bundle

    M = resolve_model(short)  # honors LLMOJI_OUT_SUFFIX on active model

    # LB pilot data lives in a sibling dataset (data/local/<short>_lb/).
    # Layer-intersect with main so all 10 centroids share a common
    # h_first layer-stack space — required for cross-cell comparisons,
    # bipolar differences, and downstream saklas vector ops.
    lb_bundle = _load_lb_aux_data(short)
    X3_lb: np.ndarray | None = None
    df_lb = None
    if lb_bundle is not None:
        df_lb, X3_lb_raw, layers_lb = lb_bundle
        common_layers = sorted(set(layer_idxs) & set(layers_lb))
        if not common_layers:
            print(f"  warning: no main ∩ lb layer overlap "
                  f"({len(layer_idxs)} main, {len(layers_lb)} lb); "
                  f"skipping LB centroid for {short}")
            X3_lb = None
        else:
            if common_layers != layer_idxs:
                idx_main = [layer_idxs.index(L) for L in common_layers]
                X3 = X3[:, idx_main, :]
                print(f"  main layer-intersect with LB pilot: "
                      f"{len(layer_idxs)} → {len(common_layers)} layers")
                layer_idxs = common_layers
            idx_lb = [layers_lb.index(L) for L in common_layers]
            X3_lb = X3_lb_raw[:, idx_lb, :]
    else:
        print(f"  no LB pilot dataset for {short} "
              f"(data/local/{short}_lb/); LB centroid will be skipped")

    quadrants = df["quadrant"].to_numpy()
    n_rows, n_layers, hidden_dim = X3.shape
    print(f"  main: {n_rows} rows, {n_layers} layers × "
          f"{hidden_dim} hidden_dim, layers {layer_idxs[0]}..{layer_idxs[-1]}")
    if X3_lb is not None:
        print(f"  lb:   {X3_lb.shape[0]} rows (layer-intersected)")

    # Per-quadrant centroids. The 9 canonical cells come from the main
    # v3 emit; MR (meta-register basin, formerly LB) comes from the
    # layer-intersected bliss-content pilot data.
    centroids: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for q in QUADRANT_ORDER_SPLIT:
        if q == "MR":
            if X3_lb is None or df_lb is None:
                counts[q] = 0
                continue
            n_q = X3_lb.shape[0]
            counts[q] = n_q
            centroids[q] = X3_lb.mean(axis=0).astype(np.float32, copy=False)
            continue
        mask = quadrants == q
        n_q = int(mask.sum())
        counts[q] = n_q
        if n_q == 0:
            print(f"  warning: quadrant {q} has 0 rows; skipping")
            continue
        centroids[q] = _centroid(X3, mask)

    print("  per-quadrant row counts: "
          + ", ".join(f"{q}={counts[q]}" for q in QUADRANT_ORDER_SPLIT))

    written = 0

    # 1. Unipolar centroid probes.
    for q, vec in centroids.items():
        slug = _QUAD_SLUG[q]
        concept = f"q_{slug}"
        profile = _profile_dict_from_layerstack(vec, layer_idxs)
        path = _save_centroid_profile(
            profile,
            concept=concept,
            model_id=M.model_id,
            method="centroid_unipolar",
            components={"quadrant": q, "n_rows": counts[q]},
        )
        written += 1
        norm = float(np.linalg.norm(vec))
        print(f"    wrote {concept:>8s}  ‖v‖={norm:8.2f}  → {path.name}")

    # 2. Quadrant-vs-NB bipolar probes.
    if "NB" not in centroids:
        print("  warning: no NB centroid available; skipping bipolar Q-vs-NB probes")
    else:
        nb_vec = centroids["NB"]
        for q, vec in centroids.items():
            if q == "NB":
                continue
            slug = _QUAD_SLUG[q]
            concept = f"{slug}.nb"
            diff = vec - nb_vec  # (n_layers, hidden_dim)
            profile = _profile_dict_from_layerstack(diff, layer_idxs)
            path = _save_centroid_profile(
                profile,
                concept=concept,
                model_id=M.model_id,
                method="centroid_difference",
                components={"plus": q, "minus": "NB",
                            "n_plus": counts[q],
                            "n_minus": counts["NB"]},
            )
            written += 1
            norm = float(np.linalg.norm(diff))
            print(f"    wrote {concept:>8s}  ‖v‖={norm:8.2f}  → {path.name}")

    # 3. Axis bipolar probes (row-weighted aggregate centroids).
    # Note: LB is canonical now but deliberately NOT included in any
    # axis-bipolar — it sits off the V/A grid (formerly the OA-1 cell)
    # and acts as a basin endpoint, not an axis-defining pole.
    def _agg_centroid(qs: tuple[str, ...]) -> np.ndarray | None:
        mask = np.isin(quadrants, qs)
        if not np.any(mask):
            return None
        return _centroid(X3, mask)

    axis_specs = [
        ("hp.ln",   _agg_centroid(_HP_AGG), _agg_centroid(("LN",)),
         {"plus": "HP", "minus": "LN", "agg": "row-weighted"}),
        ("hp.lp",   _agg_centroid(_HP_AGG), _agg_centroid(("LP",)),
         {"plus": "HP", "minus": "LP", "agg": "row-weighted"}),
        ("hnd.hns", centroids.get("HN-D"), centroids.get("HN-S"),
         {"plus": "HN-D", "minus": "HN-S", "agg": "row-weighted"}),
    ]
    for concept, plus, minus, components in axis_specs:
        if plus is None or minus is None:
            print(f"  warning: insufficient rows for {concept}; skipping")
            continue
        diff = plus - minus
        profile = _profile_dict_from_layerstack(diff, layer_idxs)
        path = _save_centroid_profile(
            profile,
            concept=concept,
            model_id=M.model_id,
            method="centroid_difference",
            components=components,
        )
        written += 1
        norm = float(np.linalg.norm(diff))
        print(f"    wrote {concept:>8s}  ‖v‖={norm:8.2f}  → {path.name}")

    print(f"  total: {written} profile files written for {short}")


def _write_pack_jsons() -> int:
    """Walk every concept folder under ``vectors/llmoji/`` and write a
    ``pack.json`` with files-hash manifest. Required by saklas's
    ``ConceptFolder.load`` (saklas/io/packs.py) — without it the UI
    grammar enumeration skips the folder entirely.

    Idempotent: re-running picks up any new safetensors+sidecar pairs
    and refreshes the hash manifest. Returns the number of pack.json
    files written.
    """
    from saklas.io.packs import synthesize_pack_metadata

    namespace_dir = _saklas_namespace_dir()
    n_written = 0
    for concept_dir in sorted(namespace_dir.iterdir()):
        if not concept_dir.is_dir():
            continue
        # Skip concept folders with no model files yet — synthesize would
        # produce an empty manifest that ConceptFolder.load rejects.
        has_files = any(
            f.is_file() and f.name != "pack.json"
            for f in concept_dir.iterdir()
        )
        if not has_files:
            print(f"  skipping pack.json for empty {concept_dir.name}")
            continue

        concept = concept_dir.name
        pack = synthesize_pack_metadata(
            name=concept,
            description=_concept_description(concept),
            tags=list(_PACK_TAGS),
            version="1.0.0",
            license=_PACK_LICENSE,
            recommended_alpha=_PACK_RECOMMENDED_ALPHA,
            source=_PACK_SOURCE,
            pack_dir=concept_dir,
        )
        pack.write(concept_dir)
        n_written += 1
        print(f"  pack.json: {concept} ({len(pack.files)} files manifested)")
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS),
        help=("Comma-separated short names (registry keys) to process. "
              f"Default: {','.join(DEFAULT_MODELS)}. "
              "Pass 'all' to walk every short name in MODEL_REGISTRY "
              "with on-disk emit data."),
    )
    parser.add_argument(
        "--list-namespace", action="store_true",
        help="Print where the saklas namespace would be written, "
             "then exit (doesn't load any data).",
    )
    parser.add_argument(
        "--namespace", default="llmoji",
        help=("Saklas namespace slug under ~/.saklas/vectors/. Default: "
              "'llmoji' (canonical mirror centroids). The 2026-05-09 "
              "self-event pilot uses 'llmoji_self_event' to keep its "
              "centroids addressable separately from canonical v3 mirror "
              "centroids without clobbering them. Must match the saklas "
              "name regex ^[a-z][a-z0-9._-]{0,63}$."),
    )
    args = parser.parse_args()

    # Latch the namespace before any disk write — _saklas_namespace_dir()
    # reads it, and the per-model writers + final pack.json synthesis
    # both go through that helper.
    global _NAMESPACE
    _NAMESPACE = args.namespace

    namespace_dir = _saklas_namespace_dir()
    print(f"saklas namespace: {namespace_dir}")
    if args.list_namespace:
        return

    if args.models == "all":
        shorts = [
            s for s, M in MODEL_REGISTRY.items()
            if M.emotional_data_path.exists()
        ]
    else:
        shorts = [s.strip() for s in args.models.split(",") if s.strip()]

    print(f"models: {shorts}")
    for short in shorts:
        try:
            _process_model(short)
        except Exception as exc:
            print(f"  [{short}] ERROR: {exc!r}")
            import traceback
            traceback.print_exc()

    # Final pass: synthesize pack.json for every concept that now has
    # at least one model's tensors written. This is required by
    # saklas's ConceptFolder.load — without pack.json the UI grammar
    # enumeration silently skips the folder.
    print("\nwriting pack.json manifests...")
    n_packs = _write_pack_jsons()
    print(f"wrote {n_packs} pack.json files under "
          f"{_saklas_namespace_dir()}")


if __name__ == "__main__":
    main()

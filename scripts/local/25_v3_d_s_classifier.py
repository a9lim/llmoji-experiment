"""Cross-validated D/S classifier on layer-stack hidden state.

For each model + each registry-split axis (HN / HP), filter to that
quadrant's rows, label by ``pad_dominance``, run StratifiedGroupKFold
(k=5) with ``prompt_id`` as the group (so same-prompt seeds never leak
across train/test), pipeline = PCA(cap=20) → StandardScaler → LR(C=0.1).
Null: 30 label shuffles, take the 95th-percentile CV-mean as the "above
this = separable" threshold. Report balanced accuracy + ROC AUC.

Why StratifiedGroupKFold: with 20 D prompts × 8 seeds vs 20 S prompts
× 8 seeds, naive row-level StratifiedKFold leaks prompt-text-level
features (same prompt's seeds in both folds). Group=prompt_id forces
unseen-prompt evaluation — the right test of the latent geometry
rather than per-prompt memorization.

HN is the rule-3-redesign positive control (confirmed separable in
gemma/qwen/ministral/gpt_oss/granite at full n; powercheck 2026-05-06
confirmed gemma + qwen separate even at LN's 5D/15S sample size).
HP is the v4-promotion case (HP-D = mischief, hp21-hp40, pad_dominance=+1
vs HP-S = celebration, hp01-hp20, pad_dominance=-1) — round-4 found
PP-vs-HP separable at 100%ile; this script tests the same thing under
the new HP-D/HP-S naming.

LN was tested historically via post-hoc agency labels and found null;
the proposed split is abandoned post-2026-05-06 powercheck. See
docs/findings.md *Round-2 powercheck — LN null verdict*.

Output:
    data/d_s_classifier_summary.tsv
    data/d_s_classifier_summary.md

Usage:
    python scripts/local/25_v3_d_s_classifier.py
    python scripts/local/25_v3_d_s_classifier.py --axes HP
    python scripts/local/25_v3_d_s_classifier.py --models gemma,qwen
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from llmoji_experiment.config import DATA_DIR, MODEL_REGISTRY
from llmoji_experiment.emotional_analysis import load_emotional_features_stack

# HN and HP D/S labels are encoded as ``pad_dominance`` on
# EmotionalPrompt; pull them from the registry. LN was previously
# split via post-hoc labels (LN_DS_LABELS in postq_d_s.py); abandoned
# 2026-05-06 after the powercheck confirmed LN-null is genuine in
# gemma + qwen and only power-confounded in models whose HN itself is
# fused. See docs/findings.md and docs/previous-experiments.md.
from transformer_experiments.kaomoji.emotional_prompts import EMOTIONAL_PROMPTS


def _registry_ds_labels(quadrant: str) -> dict[str, str]:
    """Pull D/S labels for a quadrant from the EmotionalPrompt registry.
    Returns {prompt_id: "D" | "S"} for prompts in the given quadrant
    that carry a nonzero pad_dominance."""
    return {
        p.id: ("D" if p.pad_dominance > 0 else "S")
        for p in EMOTIONAL_PROMPTS
        if p.quadrant == quadrant and p.pad_dominance != 0
    }


AXIS_LABELS: dict[str, dict[str, str]] = {
    "HN": _registry_ds_labels("HN"),
    "HP": _registry_ds_labels("HP"),
}

DEFAULT_MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
DEFAULT_AXES = ("HN", "HP")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--axes", default=",".join(DEFAULT_AXES))
    p.add_argument("--n-perm", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--pca-cap", type=int, default=20)
    p.add_argument("--C", dest="C", type=float, default=0.1)
    p.add_argument(
        "--powercheck",
        default="",
        help="Reference axis (e.g. 'LN') whose D/S prompt counts the "
             "other axes get subsampled to. When set, runs each non-ref "
             "axis on --n-subsamples random prompt-level draws; the ref "
             "axis runs once at full n. Output goes to "
             "d_s_classifier_powercheck.{tsv,md}.",
    )
    p.add_argument("--n-subsamples", type=int, default=20)
    return p.parse_args()


def _filter_to_axis(
    df: pd.DataFrame, X: np.ndarray, axis: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return (sub_df, sub_X, y, groups) restricted to axis-relevant rows.

    `axis` ∈ {HN, LN, HP}. y is a 0/1 array (D=1, S=0) over the rows that
    have a D or S label; rows without (M / mixed / untagged HN) are dropped.
    `groups` is prompt_id per row.
    """
    labels = AXIS_LABELS[axis]
    # Filter to rows in the quadrant matching this axis. AXIS_LABELS
    # keys are quadrant codes (HN, HP) so the mask is uniform.
    mask_axis = df["quadrant"] == axis
    sub_df = df.loc[mask_axis].copy()
    sub_X = X[mask_axis.to_numpy()]
    label_series = sub_df["prompt_id"].map(lambda pid: labels.get(pid))
    keep = label_series.isin(["D", "S"]).to_numpy()
    sub_df = sub_df.loc[keep].reset_index(drop=True)
    sub_X = sub_X[keep]
    y = (label_series[keep].to_numpy() == "D").astype(int)
    groups = sub_df["prompt_id"].to_numpy()
    return sub_df, sub_X, y, groups


def _make_pipe(pca_cap: int, C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("pca", PCA(n_components=pca_cap, random_state=seed)),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=seed,
        )),
    ])


def _cv_metrics(
    pipe: Pipeline, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int, seed: int,
) -> tuple[float, float, float]:
    """Run StratifiedGroupKFold CV; return (bal_acc, AUC, raw_acc).

    Aggregates predictions across folds for a single AUC + balanced-acc
    rather than averaging per-fold metrics — gives a more stable estimate
    when fold-class-counts are uneven.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred_all = np.zeros_like(y)
    y_score_all = np.zeros(len(y), dtype=float)
    for tr_idx, te_idx in cv.split(X, y, groups=groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X[tr_idx], y[tr_idx])
        y_pred_all[te_idx] = pipe.predict(X[te_idx])
        y_score_all[te_idx] = pipe.predict_proba(X[te_idx])[:, 1]
    bal_acc = float(balanced_accuracy_score(y, y_pred_all))
    auc = float(roc_auc_score(y, y_score_all)) if len(np.unique(y)) == 2 else float("nan")
    raw_acc = float((y_pred_all == y).mean())
    return bal_acc, auc, raw_acc


def _null_q95(
    pipe_factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_splits: int, n_perm: int, seed: int,
) -> tuple[float, float]:
    """Permutation null over balanced accuracy + AUC. Shuffle y *within
    groups* — preserves the prompt-level split, only randomizes the
    D-vs-S assignment of prompts. This is the right null for the
    "does prompt-text-level D/S info encode in hidden state" question.
    """
    rng = np.random.default_rng(seed)
    # Get unique prompts and their original labels.
    unique_groups = np.unique(groups)
    group_to_label = {g: y[groups == g][0] for g in unique_groups}
    null_bal: list[float] = []
    null_auc: list[float] = []
    for _ in range(n_perm):
        # Shuffle the per-prompt labels.
        perm_labels = list(group_to_label.values())
        rng.shuffle(perm_labels)
        perm_map = dict(zip(unique_groups, perm_labels))
        y_perm = np.array([perm_map[g] for g in groups])
        if len(np.unique(y_perm)) < 2:
            continue
        try:
            ba, au, _ = _cv_metrics(pipe_factory(), X, y_perm, groups, n_splits, seed)
        except Exception:
            continue
        null_bal.append(ba)
        null_auc.append(au)
    if not null_bal:
        return float("nan"), float("nan")
    return float(np.quantile(null_bal, 0.95)), float(np.quantile(null_auc, 0.95))


def _subsample_to_match(
    sub_df: pd.DataFrame,
    sub_X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_d_target: int,
    n_s_target: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Random prompt-level subsample to (n_d_target, n_s_target) prompt counts."""
    d_groups = np.unique(groups[y == 1])
    s_groups = np.unique(groups[y == 0])
    if n_d_target > len(d_groups) or n_s_target > len(s_groups):
        raise ValueError(
            f"can't subsample to {n_d_target}D/{n_s_target}S — only "
            f"{len(d_groups)}D/{len(s_groups)}S prompts available"
        )
    chosen_d = rng.choice(d_groups, size=n_d_target, replace=False)
    chosen_s = rng.choice(s_groups, size=n_s_target, replace=False)
    chosen = set(chosen_d.tolist()) | set(chosen_s.tolist())
    keep_mask = np.array([g in chosen for g in groups])
    return (
        sub_df.loc[keep_mask].reset_index(drop=True),
        sub_X[keep_mask],
        y[keep_mask],
        groups[keep_mask],
    )


def _powercheck_main(args: argparse.Namespace, models: list[str], axes: list[str]) -> None:
    """Subsample non-ref axes to ref-axis prompt counts; report distributions."""
    ref_axis = args.powercheck
    if ref_axis not in AXIS_LABELS:
        raise SystemExit(f"unknown ref axis {ref_axis!r}; valid: {list(AXIS_LABELS)}")
    # Target counts come from the AXIS_LABELS map (model-independent — labels
    # are by prompt_id).
    ref_labels = AXIS_LABELS[ref_axis]
    target_n_d = sum(1 for v in ref_labels.values() if v == "D")
    target_n_s = sum(1 for v in ref_labels.values() if v == "S")
    print(
        f"powercheck: matching to {ref_axis} = {target_n_d}D / {target_n_s}S prompts; "
        f"k={args.n_subsamples} subsamples per non-ref axis."
    )

    rows: list[dict] = []
    for short in models:
        if short not in MODEL_REGISTRY:
            print(f"[{short}] unknown model; skipping")
            continue
        M = MODEL_REGISTRY[short]
        if not M.emotional_data_path.exists():
            print(f"[{short}] no v3 emit data; skipping")
            continue
        try:
            df, X = load_emotional_features_stack(short, split_hn=False)
        except Exception as e:
            print(f"[{short}] load failed: {e}; skipping")
            continue
        print(f"[{short}] loaded {len(df)} rows, X{X.shape}")

        for axis in axes:
            full_df, full_X, full_y, full_groups = _filter_to_axis(df, X, axis)
            n_d = int((full_y == 1).sum())
            n_s = int((full_y == 0).sum())
            if n_d == 0 or n_s == 0:
                print(f"  [{short} {axis}] D={n_d}, S={n_s} — skipping")
                continue

            n_subs = 1 if axis == ref_axis else args.n_subsamples
            for sub_idx in range(n_subs):
                rng = np.random.default_rng(args.seed + sub_idx + 1)
                if axis == ref_axis:
                    s_df, s_X, s_y, s_groups = full_df, full_X, full_y, full_groups
                    mode = "full"
                else:
                    try:
                        s_df, s_X, s_y, s_groups = _subsample_to_match(
                            full_df, full_X, full_y, full_groups,
                            target_n_d, target_n_s, rng,
                        )
                    except ValueError as e:
                        print(f"  [{short} {axis}] {e}; skipping axis")
                        break
                    mode = "matched"

                n_d_p = len(np.unique(s_groups[s_y == 1]))
                n_s_p = len(np.unique(s_groups[s_y == 0]))
                pca_cap = max(2, min(args.pca_cap, s_X.shape[0] // 3))

                def factory(pca_cap=pca_cap, C=args.C, seed=args.seed) -> Pipeline:
                    return _make_pipe(pca_cap, C, seed)

                try:
                    bal_acc, auc, raw_acc = _cv_metrics(
                        factory(), s_X, s_y, s_groups, args.n_splits, args.seed,
                    )
                    null_bal_q95, null_auc_q95 = _null_q95(
                        factory, s_X, s_y, s_groups,
                        args.n_splits, args.n_perm, args.seed,
                    )
                except Exception as e:
                    print(f"  [{short} {axis} sub{sub_idx}] CV failed: {e}")
                    continue

                rows.append({
                    "model": short,
                    "axis": axis,
                    "mode": mode,
                    "subsample_idx": sub_idx,
                    "n_d_prompts": n_d_p,
                    "n_s_prompts": n_s_p,
                    "n_d_rows": int((s_y == 1).sum()),
                    "n_s_rows": int((s_y == 0).sum()),
                    "pca_cap": pca_cap,
                    "raw_acc": raw_acc,
                    "bal_acc": bal_acc,
                    "auc": auc,
                    "null_bal_q95": null_bal_q95,
                    "null_auc_q95": null_auc_q95,
                    "separable": bal_acc > null_bal_q95,
                })

            # Per-axis summary line.
            sub_rows = [r for r in rows if r["model"] == short and r["axis"] == axis]
            if sub_rows:
                ba = np.array([r["bal_acc"] for r in sub_rows])
                nq = np.array([r["null_bal_q95"] for r in sub_rows])
                sep_frac = float(np.mean([r["separable"] for r in sub_rows]))
                print(
                    f"  [{short} {axis}] {sub_rows[0]['mode']}, k={len(sub_rows)} · "
                    f"bal_acc median={np.median(ba):.3f} IQR=[{np.quantile(ba, .25):.3f},"
                    f"{np.quantile(ba, .75):.3f}] · null q95 median={np.median(nq):.3f} · "
                    f"separable {sep_frac:.0%}"
                )

    if not rows:
        print("no rows to write")
        return

    df_out = pd.DataFrame(rows)
    out_tsv = DATA_DIR / "d_s_classifier_powercheck.tsv"
    out_md = DATA_DIR / "d_s_classifier_powercheck.md"
    df_out.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}")

    # Aggregate to (model, axis) median ± IQR.
    md_lines: list[str] = ["# D/S classifier — powercheck (matched-n subsampling)\n"]
    md_lines.append(
        f"Reference axis: **{ref_axis}** at {target_n_d}D / {target_n_s}S prompts. "
        f"Non-ref axes subsampled to the same prompt counts; "
        f"**k={args.n_subsamples}** random draws per (model, axis). "
        f"Each draw: full StratifiedGroupKFold(k={args.n_splits}) CV + "
        f"{args.n_perm}-permutation null on the *subsampled* prompt set.\n"
    )
    md_lines.append(
        "**Question**: at the LN sample size, does HN still separate? "
        "If yes → LN's null is genuine (the design CAN detect a real "
        "signal at this n). If HN drops to chance → LN null is power-"
        "confounded.\n"
    )
    md_lines.append(
        "| model | axis | mode | n D/S | bal_acc median (IQR) | null q95 median | separable %ile |"
    )
    md_lines.append("|---|---|---|---|---|---:|---:|")
    for short in models:
        for axis in axes:
            sub = df_out[(df_out["model"] == short) & (df_out["axis"] == axis)]
            if sub.empty:
                continue
            ba = sub["bal_acc"].to_numpy()
            nq = sub["null_bal_q95"].to_numpy()
            sep_frac = float(sub["separable"].mean())
            mode = sub["mode"].iloc[0]
            n_d = int(sub["n_d_prompts"].iloc[0])
            n_s = int(sub["n_s_prompts"].iloc[0])
            md_lines.append(
                f"| {short} | {axis} | {mode} (k={len(sub)}) | {n_d}/{n_s} | "
                f"{np.median(ba):.3f} ([{np.quantile(ba, .25):.3f},"
                f"{np.quantile(ba, .75):.3f}]) | "
                f"{np.median(nq):.3f} | {sep_frac:.0%} |"
            )
    md_lines.append("")
    md_lines.append("## Reading\n")
    md_lines.append(
        f"- **{ref_axis} (mode=full)** is the question: a single point estimate "
        f"at this prompt-count budget.\n"
        f"- **HN (mode=matched)** is the positive-control answer at matched n. "
        f"If `separable %` ≈ 100%, the design has power; LN's null is real. "
        f"If `separable %` drops sharply (toward 50% or below), LN's null is "
        f"power-confounded and we need more LN-D prompts before concluding.\n"
        f"- **HP (mode=matched)** is the negative-control reference: HP was "
        f"already null at full n, so it should remain null at matched n.\n"
    )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {out_md}")


def main() -> None:
    args = _parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    for a in axes:
        if a not in AXIS_LABELS:
            raise SystemExit(f"unknown axis {a!r}; valid: {list(AXIS_LABELS)}")

    if args.powercheck:
        _powercheck_main(args, models, axes)
        return

    rows: list[dict] = []
    for short in models:
        if short not in MODEL_REGISTRY:
            print(f"[{short}] unknown model; skipping")
            continue
        M = MODEL_REGISTRY[short]
        if not M.emotional_data_path.exists():
            print(f"[{short}] no v3 emit data; skipping")
            continue
        try:
            df, X = load_emotional_features_stack(short, split_hn=False)
        except Exception as e:
            print(f"[{short}] load failed: {e}; skipping")
            continue
        print(f"[{short}] loaded {len(df)} rows, X{X.shape}")

        for axis in axes:
            sub_df, sub_X, y, groups = _filter_to_axis(df, X, axis)
            n_d = int((y == 1).sum())
            n_s = int((y == 0).sum())
            n_d_prompts = len(np.unique(groups[y == 1]))
            n_s_prompts = len(np.unique(groups[y == 0]))
            if n_d == 0 or n_s == 0:
                print(f"  [{short} {axis}] D={n_d}, S={n_s} — skipping")
                continue
            # Pipeline factory keeps each null-perm fit using a fresh
            # estimator with the same hyperparameters / seed.
            pca_cap = max(2, min(args.pca_cap, sub_X.shape[0] // 3))

            def factory(pca_cap=pca_cap, C=args.C, seed=args.seed) -> Pipeline:
                return _make_pipe(pca_cap, C, seed)

            try:
                bal_acc, auc, raw_acc = _cv_metrics(
                    factory(), sub_X, y, groups, args.n_splits, args.seed,
                )
                null_bal_q95, null_auc_q95 = _null_q95(
                    factory, sub_X, y, groups,
                    args.n_splits, args.n_perm, args.seed,
                )
            except Exception as e:
                print(f"  [{short} {axis}] CV failed: {e}")
                continue

            verdict = (
                "**SEPARABLE**" if bal_acc > null_bal_q95
                else "not separable"
            )
            row = {
                "model": short,
                "axis": axis,
                "n_d_prompts": n_d_prompts,
                "n_s_prompts": n_s_prompts,
                "n_d_rows": n_d,
                "n_s_rows": n_s,
                "pca_cap": pca_cap,
                "raw_acc": raw_acc,
                "bal_acc": bal_acc,
                "auc": auc,
                "null_bal_q95": null_bal_q95,
                "null_auc_q95": null_auc_q95,
                "separable": bal_acc > null_bal_q95,
            }
            rows.append(row)
            print(
                f"  [{short} {axis}] D={n_d_prompts}p×{n_d//n_d_prompts}s, "
                f"S={n_s_prompts}p×{n_s//n_s_prompts}s · "
                f"bal_acc={bal_acc:.3f} (null q95 {null_bal_q95:.3f}) · "
                f"AUC={auc:.3f} (null q95 {null_auc_q95:.3f}) · {verdict}"
            )

    if not rows:
        print("no rows to write")
        return

    df_out = pd.DataFrame(rows)
    out_tsv = DATA_DIR / "d_s_classifier_summary.tsv"
    out_md = DATA_DIR / "d_s_classifier_summary.md"
    df_out.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nwrote {out_tsv}")

    md_lines: list[str] = ["# D/S classifier — layer-stack hidden state\n"]
    md_lines.append(
        "Pipeline: PCA(cap=20) → StandardScaler → L2-logistic (C=0.1). "
        "CV: StratifiedGroupKFold(k=5) with prompt_id as group. Null: "
        f"{args.n_perm} within-group label shuffles → 95th-percentile "
        "CV bal_acc / AUC.\n"
    )
    md_lines.append(
        "**Verdict**: bal_acc > null_q95 → the dominance axis is "
        "linearly recoverable from layer-stack hidden state in this "
        "model. HN is the positive control (rule-3-redesign confirmed); "
        "HP is the null reference (post-hoc analysis showed no D/S "
        "split universally).\n"
    )
    md_lines.append("| model | axis | D/S prompts | bal_acc | null q95 | AUC | null q95 | verdict |")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for r in rows:
        verdict = "**SEPARABLE**" if r["separable"] else "not separable"
        md_lines.append(
            f"| {r['model']} | {r['axis']} | {r['n_d_prompts']}/{r['n_s_prompts']} | "
            f"{r['bal_acc']:.3f} | {r['null_bal_q95']:.3f} | "
            f"{r['auc']:.3f} | {r['null_auc_q95']:.3f} | {verdict} |"
        )
    md_lines.append("")
    md_lines.append("## Methodology notes\n")
    md_lines.append(
        "- Same-prompt seeds are forced into the same fold via "
        "`StratifiedGroupKFold(groups=prompt_id)`; without this, "
        "row-level CV leaks prompt-text features and inflates accuracy.\n"
    )
    md_lines.append(
        "- Permutation null shuffles labels *within groups* — preserves "
        "the per-prompt seed structure, only randomizes the D-vs-S "
        "assignment. This is the right null for *latent geometric* "
        "encoding (vs prompt-text-level memorization).\n"
    )
    md_lines.append(
        "- PCA cap = min(20, n_rows // 3). With ~120-160 rows per axis "
        "this lands at 20 for HN/LN, ~20 for HP. The classifier still "
        "has more degrees of freedom than the data; regularization (C=0.1) "
        "carries it.\n"
    )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

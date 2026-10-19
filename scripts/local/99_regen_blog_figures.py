"""Regenerate the introspection-via-kaomoji blog post's PNG figures.

Writes light + dark variants of the static figures embedded in
[a9l.im/blog/introspection-via-kaomoji](https://a9l.im/blog/introspection-via-kaomoji),
into the a9l.im checkout's blog-assets folder (``$BLOG_ASSETS_DIR`` if set,
else ``../../a9lim.github.io/static/blog-assets/introspection-via-kaomoji``
relative to this experiment):

    fig_v3_layerwise_emergence_compare_{light,dark}.png
    fig_emo_a_kaomoji_sim_{gemma,qwen,ministral,gpt_oss_20b,granite}_{light,dark}.png

Styling follows the a9l.im design system (its AGENTS.md "Design system"):
near-white / near-black grounds, hairline rules at 12% (light) / 13% (dark)
of the text color, Newsreader throughout (static instances cut by
``scripts/make_newsreader_instances.py`` into ``data/fonts/``), the
per-theme oxblood accent where the brand red was used (gemma's line), and
slate ``#8A8580`` for NB. Data hues are the shared extended palette,
unchanged. The cosine heatmaps keep matplotlib's ``RdBu_r`` divergent
colormap.

Run after re-running ``scripts/harness/10_emit_analysis.py`` and
``scripts/local/20_v3_layerwise_emergence.py`` for any model whose data has
changed. The layerwise path reads the per-model ``v3_layerwise_emergence.tsv``
that script 20 writes, and renders at the pixel size the post was laid out
for. The heatmap path reads the hidden-state sidecars and h_first cache
under ``data/local/hidden`` / ``data/local/cache``, which are gitignored and
local-only; with them absent it skips the heatmaps, says so, and leaves any
existing PNGs untouched.

Usage:
    python scripts/local/99_regen_blog_figures.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(STUDY_ROOT))

from llmoji_experiment.config import MODEL_REGISTRY  # noqa: E402
from llmoji_experiment.emotional_analysis import (  # noqa: E402
    load_emotional_features_stack,
    plot_kaomoji_cosine_heatmap,
    register_newsreader,
)
from interpretability_experiments.kaomoji.quadrants import QUADRANT_COLORS  # noqa: E402

# Output target — the a9l.im checkout beside this workspace.
BLOG_ASSETS_DIR = Path(os.environ.get("BLOG_ASSETS_DIR") or (
    STUDY_ROOT.parent.parent / "a9lim.github.io" / "static"
    / "blog-assets" / "introspection-via-kaomoji"
))
DATA_DIR = STUDY_ROOT / "data"
# Heatmap is per-model; regenerate for the full v3 main lineup.
HEATMAP_MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
# Layerwise comparison only includes models with v3_layerwise_emergence.tsv;
# script 20 wasn't re-run on the gpt_oss / granite additions because the
# layer-stack rep supersedes single-layer silhouette as the canonical
# read. Keep the 3-model trajectory as a depth-emergence illustration.
LAYERWISE_MODELS = ("gemma", "qwen", "ministral")
# Pixel size of the layerwise PNG the post was laid out for (3 panels).
LAYERWISE_PX = (2384, 732)
DPI = 150

# Site palette, per theme (a9l.im AGENTS.md "Design system").
SITE: dict[str, dict[str, str]] = {
    "light": {
        "canvas": "#FCFCFB", "text": "#161514", "secondary": "#42413F",
        "muted": "#666463", "accent": "#983432",
    },
    "dark": {
        "canvas": "#121111", "text": "#ECEBEA", "secondary": "#BEBDBC",
        "muted": "#949392", "accent": "#DC5960",
    },
}
# Hairline rules: the text color at this alpha over the canvas.
RULE_ALPHA = {"light": 0.12, "dark": 0.13}
# Slate (quadrant NB) moved with the redesign; every other quadrant color is
# the shared palette's, so override just this one for the blog.
SITE_SLATE = "#8A8580"
BLOG_QUADRANT_COLORS = {**QUADRANT_COLORS, "NB": SITE_SLATE}


def _over(fg: str, bg: str, alpha: float) -> str:
    """Opaque hex of ``fg`` at ``alpha`` composited over ``bg``."""
    f = [int(fg[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x * alpha + y * (1 - alpha)):02x}" for x, y in zip(f, b))


def _theme_params(theme: str) -> dict[str, object]:
    c = SITE[theme]
    rule = _over(c["text"], c["canvas"], RULE_ALPHA[theme])
    return {
        "figure.facecolor":  c["canvas"],
        "axes.facecolor":    c["canvas"],
        "savefig.facecolor": c["canvas"],
        "figure.edgecolor":  c["canvas"],
        "savefig.edgecolor": c["canvas"],
        "text.color":        c["text"],
        "axes.labelcolor":   c["secondary"],
        "axes.titlecolor":   c["text"],
        "xtick.color":       rule,
        "ytick.color":       rule,
        "xtick.labelcolor":  c["muted"],
        "ytick.labelcolor":  c["muted"],
        "axes.edgecolor":    rule,
        "grid.color":        rule,
        "axes.linewidth":    0.6,
        "grid.linewidth":    0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.labelcolor": c["secondary"],
    }


def with_theme(theme: str) -> None:
    if not register_newsreader():
        raise SystemExit(
            "Newsreader instances missing from data/fonts/; run "
            "scripts/make_newsreader_instances.py (see its docstring)"
        )
    plt.rcdefaults()
    plt.rcParams.update(_theme_params(theme))
    plt.rcParams.update({
        "font.family":       ["Newsreader"],
        "font.size":         12,
        "axes.labelsize":    13,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   12,
        "axes.unicode_minus": True,
    })


def line_palette(theme: str) -> dict[str, str]:
    """Per-model line colors for the layerwise comparison: the theme's accent
    for gemma (the brand red before), shared extended hues for the rest —
    secondary / green, plus two more for the 5-model lineup."""
    return {
        "gemma":       SITE[theme]["accent"],
        "qwen":        "#488ACB",
        "ministral":   "#009F68",
        "gpt_oss_20b": "#D49A1A",
        "granite":     "#8B5CF6",
    }


def plot_layerwise(theme: str) -> None:
    with_theme(theme)
    colors = line_palette(theme)
    per_model: dict[str, pd.DataFrame] = {}
    for short in LAYERWISE_MODELS:
        M = MODEL_REGISTRY[short]
        tsv = M.figures_dir / "v3_layerwise_emergence.tsv"
        if not tsv.exists():
            print(f"  [layerwise] no TSV at {tsv} — skipping {short}")
            continue
        per_model[short] = pd.read_csv(tsv, sep="\t")

    fig, axes = plt.subplots(
        1, 3, figsize=(LAYERWISE_PX[0] / DPI, LAYERWISE_PX[1] / DPI),
    )
    panels = [
        (axes[0], "silhouette",
         "Silhouette score (quadrants in PC1–PC2)"),
        (axes[1], "between_centroid_std_pc1",
         "Between-centroid std on PC1"),
        (axes[2], "between_centroid_std_pc2",
         "Between-centroid std on PC2"),
    ]
    for ax, ycol, ylabel in panels:
        ax.grid(axis="y", linewidth=0.6)
        ax.set_axisbelow(True)
        for short, metrics in per_model.items():
            layers = metrics["layer"].to_numpy()
            denom = max(1.0, float(layers.max() - layers.min()))
            depth = (layers - layers.min()) / denom
            ax.plot(depth, metrics[ycol],
                    color=colors[short], linewidth=1.8, label=short)
            ax.scatter(depth, metrics[ycol],
                       s=14, color=colors[short], linewidths=0)
        if ycol == "silhouette":
            ax.axhline(0, color=plt.rcParams["xtick.labelcolor"],
                       linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_xlabel("Fractional depth (layer normalized to [0, 1])",
                      fontstyle="italic")
        ax.set_ylabel(ylabel, fontstyle="italic")
        ax.tick_params(length=3)
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    out = BLOG_ASSETS_DIR / f"fig_v3_layerwise_emergence_compare_{theme}.png"
    # Fixed canvas, no bbox trimming: the pixel size stays what the post
    # expects whatever the font metrics do.
    fig.savefig(out, dpi=DPI, facecolor=plt.rcParams["figure.facecolor"])
    plt.close(fig)
    print(f"  wrote {out}")


def plot_heatmap(short: str, theme: str) -> bool:
    """Returns False (and writes nothing) when the hidden-state data this
    figure is computed from is not on this machine."""
    with_theme(theme)
    df, X = load_emotional_features_stack(
        short, which="h_first", split_hn=True,
    )
    if len(df) == 0:
        print(f"  [heatmap] no hidden-state rows for {short} "
              f"(data/local/hidden, data/local/cache missing) — skipped, "
              f"existing PNG left in place")
        return False
    out = BLOG_ASSETS_DIR / f"fig_emo_a_kaomoji_sim_{short}_{theme}.png"
    plot_kaomoji_cosine_heatmap(
        df, X, str(out), min_count=3,
        font_primary="Newsreader", label_fontstyle="italic",
        colors=BLOG_QUADRANT_COLORS,
    )
    print(f"  wrote {out}")
    return True


def main() -> None:
    if not BLOG_ASSETS_DIR.parent.exists():
        print(f"blog-assets parent does not exist: {BLOG_ASSETS_DIR.parent}")
        print("  set BLOG_ASSETS_DIR, or edit the default at the top of this "
              "script, if your a9l.im checkout lives somewhere else.")
        sys.exit(1)
    BLOG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    for theme in ("light", "dark"):
        print(f"\n=== theme={theme} ===")
        plot_layerwise(theme)
        for short in HEATMAP_MODELS:
            if not plot_heatmap(short, theme):
                skipped.append(f"{short}_{theme}")
    if skipped:
        print(f"\nSKIPPED {len(skipped)} heatmaps (no hidden-state data): "
              + ", ".join(skipped))


if __name__ == "__main__":
    main()

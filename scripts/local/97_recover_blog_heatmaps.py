"""Recover the blog's per-kaomoji cosine heatmaps (Figure A) from their
published PNGs, so 99_regen_blog_figures.py can redraw them without the
hidden-state sidecars they were computed from (data/local/hidden/ and
data/local/cache/ no longer exist).

Each published PNG is an ``imshow`` of the ordered similarity matrix in
``RdBu_r`` on [-1, 1], rows labelled ``face  n=count`` in the colour of the
face's dominant quadrant. Everything the figure shows comes back exactly:

- the matrix, cell by cell, as the colormap entry it was drawn with. The
  colormap has 256 entries, so each value is stored as the centre of its
  entry's range (within 1/256 of the original) and the diagonal as exactly
  1; redrawn with the same colormap every cell is identical;
- each row's count, read from its ``n=…`` digits;
- each row's quadrant, from its label colour;
- each row's face, matched against the faces today's emission table
  (``data/local/<model>/emotional_raw.jsonl``) has with n >= 3, the figure's
  threshold, rendered in the font chain the PNGs were drawn with.

Counts and colours are taken from the PNG, not recomputed: a few faces'
counts have drifted since the figure was made, and the figure should keep
what was published. Validates that every grid has exactly as many rows as
there are candidate faces, square cells, a unit diagonal and a symmetric
matrix, and stops if not.

    python scripts/local/97_recover_blog_heatmaps.py PNG_DIR

PNG_DIR holds the published ``fig_emo_a_kaomoji_sim_<model>_light.png`` (the
a9l.im repository has them at commit 08e3833^,
static/blog-assets/introspection-via-kaomoji/). Writes
data/blog/fig_emo_a_recovered.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from llmoji_experiment.emotional_analysis import load_emotional_metadata, per_face_dominant_quadrant  # noqa: E402
from interpretability_experiments.kaomoji.quadrants import QUADRANT_COLORS  # noqa: E402

MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
OUT = REPO / "data" / "blog" / "fig_emo_a_recovered.json"
MIN_COUNT = 3                      # 99_regen_blog_figures.py's threshold
GROUND = np.array([247, 248, 251])  # the published light figures' ground
LUT = matplotlib.colormaps["RdBu_r"](np.linspace(0, 1, 256), bytes=True)[:, :3].astype(int)  # as drawn
VALUES = -1 + (2 * np.arange(256) + 1) / 256   # centre of each colormap entry's value range
# The chain the published PNGs were drawn with (Matplotlib 3.10, before the
# blog figures moved to Newsreader), so candidate labels render like them.
PUBLISHED_CHAIN = ["Noto Sans CJK JP", "Arial Unicode MS", "DejaVu Sans", "DejaVu Serif", "Tahoma",
                   "Heiti TC", "Hiragino Sans", "Apple Symbols", "Noto Emoji", "Helvetica Neue"]
PALETTE = {q: np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) for q, c in QUADRANT_COLORS.items()}
LTR = "‭{}‬"


def longest_run(flags) -> tuple[int, int]:
    best, start = (0, 0, 0), None
    for i, f in enumerate([*flags, False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            if i - start > best[0]:
                best = (i - start, start, i - 1)
            start = None
    return best[1], best[2]


def colormap_index(im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest colormap entry and its L1 distance, per pixel."""
    flat = im.reshape(-1, 3)
    idx, dist = np.empty(len(flat), int), np.empty(len(flat), int)
    for s in range(0, len(flat), 200_000):
        d = np.abs(flat[s:s + 200_000, None, :] - LUT[None]).sum(-1)
        idx[s:s + 200_000], dist[s:s + 200_000] = d.argmin(1), d.min(1)
    return idx.reshape(im.shape[:2]), dist.reshape(im.shape[:2])


def bbox_crop(a: np.ndarray, thr: float = 0.15) -> np.ndarray:
    ys, xs = np.nonzero(a > thr)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else np.zeros((1, 1))


_renders: dict[str, np.ndarray] = {}


def render(text: str) -> np.ndarray:
    """Ink coverage of ``text`` at the figure's tick-label size (7 pt, 150 dpi)."""
    if text not in _renders:
        fig = plt.figure(figsize=(4, 0.5), dpi=150)
        fig.text(0.02, 0.5, LTR.format(text), fontsize=7, va="center", color="black")
        fig.canvas.draw()
        a = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float)
        plt.close(fig)
        _renders[text] = bbox_crop(1 - a.mean(-1) / 255)
    return _renders[text]


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation of lightly blurred, right-aligned ink,
    best over small shifts (tolerates hinting and anti-aliasing changes
    between Matplotlib releases)."""
    best = -1.0
    for dy in (-1, 0, 1):
        for dx in (-2, -1, 0, 1, 2):
            h, w = max(a.shape[0], b.shape[0]) + 4, max(a.shape[1], b.shape[1]) + 6
            A, B = np.zeros((h, w)), np.zeros((h, w))
            ya, yb = (h - a.shape[0]) // 2, (h - b.shape[0]) // 2 + dy
            A[ya:ya + a.shape[0], w - 3 - a.shape[1]:w - 3] = a
            B[yb:yb + b.shape[0], w - 3 - b.shape[1] + dx:w - 3 + dx] = b
            A, B = gaussian_filter(A, 0.8), gaussian_filter(B, 0.8)
            A, B = A - A.mean(), B - B.mean()
            best = max(best, float((A * B).sum() / max(np.sqrt((A * A).sum() * (B * B).sum()), 1e-9)))
    return best


def recover(model: str, png: Path) -> dict:
    df = load_emotional_metadata(model, split_hn=True)
    counts = df.groupby("first_word").size()
    dominant = per_face_dominant_quadrant(df)
    faces = [str(k) for k, v in counts.items() if v >= MIN_COUNT]
    n = len(faces)

    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    idx, dist = colormap_index(im)
    on = dist <= 3
    x0, x1 = longest_run(on.sum(0) > 0.15 * on.shape[0])
    y0, y1 = longest_run(on[:, x0:x1 + 1].sum(1) > 0.6 * (x1 - x0 + 1))
    cw, ch = (x1 - x0 + 1) / n, (y1 - y0 + 1) / n
    if abs(cw - ch) > 0.5:
        raise SystemExit(f"{model}: {n} candidate faces do not tile the grid into square cells ({cw:.2f} x {ch:.2f})")

    sim = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            cy, cx = int(y0 + (i + 0.5) * ch), int(x0 + (j + 0.5) * cw)
            if dist[cy, cx]:
                raise SystemExit(f"{model}: cell ({i}, {j}) is not exactly a colormap colour")
            sim[i, j] = VALUES[idx[cy, cx]]
    if not (np.all(np.diag(sim) == VALUES[-1]) and np.array_equal(sim, sim.T)):
        raise SystemExit(f"{model}: recovered matrix is not a unit-diagonal symmetric similarity matrix")
    np.fill_diagonal(sim, 1.0)

    quadrants = sorted({dominant[f] for f in faces} | {"NP", "LP", "NB", "HB"})
    rows = []
    for i in range(n):
        cy = y0 + (i + 0.5) * ch
        crop = im[int(cy - ch / 2):int(cy + ch / 2), 0:x0 - 12]   # left of the tick marks
        d = np.sqrt(((crop - GROUND) ** 2).sum(-1))
        color = np.median(crop[d > 0.8 * d.max()], 0)
        ink = bbox_crop(np.clip(d / np.sqrt(((color - GROUND) ** 2).sum()), 0, 1))
        q = min(quadrants, key=lambda qq: np.abs(PALETTE[qq] - color).sum())
        # The count sits right of the label's double space, its rightmost
        # blank run of 5 px or more (faces can hold wide blanks of their own,
        # "n=…" never does); score only counts rendered as wide as that piece.
        blank = (ink > 0.15).sum(0) == 0
        runs, start = [], None
        for x, b in enumerate([*blank, False]):
            if b and start is None:
                start = x
            if not b and start is not None:
                runs.append((x - start, start, x))
                start = None
        wide = [r for r in runs if r[0] >= 5]
        strip = ink[:, wide[-1][2]:] if wide else ink
        widths = {k: render(f"n={k}").shape[1] for k in range(MIN_COUNT, 2 * int(counts.max()) + 50)}
        near = [k for k, w in widths.items() if abs(w - strip.shape[1]) <= 3] or list(widths)
        count = max(near, key=lambda k: similarity(strip, render(f"n={k}")))
        rows.append({"ink": ink, "q": q, "n": count})

    gap = max(render("a  n=1").shape[1] - render("a").shape[1] - render("n=1").shape[1], 4)
    score = np.zeros((n, n))
    for i, r in enumerate(rows):
        face_ink = bbox_crop(r["ink"][:, :max(r["ink"].shape[1] - render(f"n={r['n']}").shape[1] - gap, 4)])
        for j, f in enumerate(faces):
            score[i, j] = similarity(face_ink, render(f))
    cost = (-score
            + 5.0 * np.array([[rows[i]["q"] != dominant[faces[j]] for j in range(n)] for i in range(n)])
            + 0.5 * np.abs(np.log([[rows[i]["n"]] for i in range(n)]) - np.log([[counts[f] for f in faces]])))
    ri, cj = linear_sum_assignment(cost)
    order = dict(zip(ri.tolist(), cj.tolist()))
    matched = [score[i, order[i]] for i in range(n)]
    print(f"{model}: {n} rows; face match min {min(matched):.2f}, mean {np.mean(matched):.2f}; "
          f"{sum(rows[i]['n'] != counts[faces[order[i]]] for i in range(n))} published counts differ from today's table")
    return {
        "faces": [faces[order[i]] for i in range(n)],
        "counts": [rows[i]["n"] for i in range(n)],
        "quadrants": [rows[i]["q"] for i in range(n)],
        "sim": np.round(sim, 4).tolist(),
    }


def main(png_dir: str) -> None:
    emoji = REPO / "data" / "fonts" / "NotoEmoji-Regular.ttf"
    fm.fontManager.addfont(str(emoji))
    matplotlib.rcParams["font.family"] = [f for f in PUBLISHED_CHAIN if f in {x.name for x in fm.fontManager.ttflist}]
    out = {m: recover(m, Path(png_dir) / f"fig_emo_a_kaomoji_sim_{m}_light.png") for m in MODELS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

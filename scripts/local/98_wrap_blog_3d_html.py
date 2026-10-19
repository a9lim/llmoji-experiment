"""Wrap raw plotly 3D HTML files with the a9l.im site-themed shell.

Two wrappers:

1. **Procrustes overlay**: takes the multi-scene raw plotly HTML output by
   `26_v3_quadrant_procrustes.py` and extracts only the overlay scene
   (scene6 for 5 models, scene N+1 generally) into a single full-pane
   plotly view, plus a static legend bar showing per-model markers.

2. **Per-face PCA with model toggle**: takes the multi-scene raw plotly
   HTML output by the per-face PCA generator, exposes a model
   segmented-control via shared-forms, and re-themes the active scene
   in-place using the site's CSS custom-property palette.

All three re-paint plotly's grid/text/background colors from the parent
document's `data-theme` attribute (with a MutationObserver), so the
figures track the site's light/dark toggle live. Palette and type follow
the a9l.im design system (AGENTS.md "Design system"): near-white / near-black
grounds, hairline rules, a single oxblood accent, Newsreader throughout.
The plotly palette is spelled out in the page script (``SITE_PALETTE``)
rather than read from tokens.js, so a figure stays correct whatever state
the shared layer is in; the fonts come from the site's ``/fonts/fonts.css``.

The one data color that moves is slate (quadrant NB): the producers baked
the pre-redesign ``QUADRANT_COLORS["NB"]`` into the raw HTMLs, so
``_retint_slate`` re-tints it (and the per-face RGB blends that contain
it) to the site's ``#8A8580`` at wrap time. A no-op once the producers
emit that value themselves.

Output goes to ``$BLOG_ASSETS_DIR`` if set, else the a9l.im checkout's
``static/blog-assets/introspection-via-kaomoji/`` (``~/Work/a9lim.github.io``).

Usage:
    python scripts/local/98_wrap_blog_3d_html.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
BLOG_ASSETS_DIR = Path(os.environ.get("BLOG_ASSETS_DIR") or (
    STUDY_ROOT.parent.parent / "a9lim.github.io" / "static"
    / "blog-assets" / "introspection-via-kaomoji"
))
RAW_PROCRUSTES = STUDY_ROOT / "figures" / "local" / "fig_v3_quadrant_procrustes_3d.html"
RAW_PER_FACE = STUDY_ROOT / "figures" / "local" / "fig_v3_per_face_pca_3d.html"
RAW_WILD_FACES = STUDY_ROOT / "figures" / "harness" / "wild_faces_pca_3d.html"
OUT_PROCRUSTES = BLOG_ASSETS_DIR / "fig_v3_quadrant_procrustes_3d.html"
OUT_PER_FACE = BLOG_ASSETS_DIR / "fig_v3_per_face_pca_3d.html"
OUT_WILD_FACES = BLOG_ASSETS_DIR / "fig_wild_faces_pca_3d.html"


from interpretability_experiments.kaomoji.quadrants import (  # noqa: E402
    QUADRANT_COLORS,
    QUADRANT_ORDER_SPLIT,
)

# Site slate (design system `extended.slate`). NB is the only quadrant whose
# color moved in the redesign; everything else stays the shared palette.
SITE_SLATE = "#8A8580"
SITE_QUADRANT_COLORS = {**QUADRANT_COLORS, "NB": SITE_SLATE}

# Active models, in order. The procrustes file currently emits 5 per-model
# scenes (scene1..scene5) + 1 overlay (scene6).
MODELS = ("gemma", "qwen", "ministral", "gpt_oss_20b", "granite")
# Plotly 3D marker symbols used by 26_v3_quadrant_procrustes.py.
MODEL_MARKERS = {
    "gemma":       "circle",
    "qwen":        "diamond",
    "ministral":   "square",
    "gpt_oss_20b": "cross",
    "granite":     "x",
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _nb_share(hover: str) -> float | None:
    """NB's share of a point's quadrant blend, read back from its hover text.

    Two formats exist: per-face PCA lists emission counts
    (``HP-S=77, HP-D=45, NP=28, NB=4``; share = NB / sum), and wild-faces
    PCA lists BoL shares (``BoL shares: NP 0.50, NB 0.45``; already shares,
    but only the top three, so an unlisted NB counts as 0 — it is smaller
    than every listed share, which bounds the error at a few RGB levels).
    Returns None when the text carries neither.
    """
    plain = re.sub(r"<[^>]+>", "|", hover)
    counts = re.findall(r"\b([A-Z]{2}(?:-[DS])?)=(\d+)\b", plain)
    if counts:
        total = sum(int(n) for _, n in counts)
        nb = sum(int(n) for q, n in counts if q == "NB")
        return nb / total if total else 0.0
    m = re.search(r"shares:\s*([^|]+)", plain)
    if m:
        pairs = re.findall(r"\b([A-Z]{2}(?:-[DS])?)\s+(\d*\.?\d+)", m.group(1))
        if pairs:
            return sum(float(v) for q, v in pairs if q == "NB")
    return None


def _retint_slate(plotly_block: str) -> str:
    """Move the baked-in NB slate to the site's slate.

    The producers (26, 97, 67) paint markers with ``QUADRANT_COLORS`` as of
    their last run. Solid NB traces map directly; per-face markers are an
    RGB-linear blend of the quadrant colors weighted by share, so each moves
    by ``share_NB * (new_slate - old_slate)``. No-op when the raw HTML
    already carries the site slate.
    """
    old = QUADRANT_COLORS["NB"].lower()
    new = SITE_SLATE.lower()
    if old == new:
        return plotly_block
    d = tuple(n - o for n, o in zip(_hex_to_rgb(new), _hex_to_rgb(old)))

    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*', plotly_block)
    if not m:
        raise SystemExit("could not find the Plotly.newPlot call to re-tint")
    data, end = json.JSONDecoder().raw_decode(plotly_block, m.end())

    n_solid = n_blend = n_skipped = 0
    for tr in data:
        marker = tr.get("marker") or {}
        col = marker.get("color")
        if isinstance(col, str):
            if col.lower() == old:
                marker["color"] = SITE_SLATE
                n_solid += 1
        elif isinstance(col, list) and isinstance(tr.get("hovertext"), list):
            out = []
            for c0, hover in zip(col, tr["hovertext"]):
                share = _nb_share(hover) if isinstance(hover, str) else None
                if share is None or not (isinstance(c0, str) and c0.startswith("#")):
                    out.append(c0)
                    n_skipped += 1
                    continue
                rgb = _hex_to_rgb(c0)
                out.append("#%02x%02x%02x" % tuple(
                    max(0, min(255, round(v + share * dv))) for v, dv in zip(rgb, d)
                ))
                n_blend += 1
            marker["color"] = out
    print(f"    slate {old} -> {new}: {n_solid} solid traces, "
          f"{n_blend} blended points, {n_skipped} left as-is")
    return (plotly_block[:m.end()]
            + json.dumps(data, separators=(",", ":"))
            + plotly_block[end:])


def _extract_plotly_block(raw: str) -> str:
    """Pull just the `<div>...plotly init...<script>...</script></div>` block
    out of a py-plotly-emitted full HTML file. Skips the surrounding
    `<html>`, `<head>`, `<body>` boilerplate so we can paste it inside our
    own shell. Re-tints the NB slate on the way through."""
    m = re.search(r"<body>\s*(<div>.*</div>)\s*</body>", raw, re.DOTALL)
    if not m:
        raise SystemExit(f"could not find <body><div>...</div></body> block in raw plotly HTML")
    return _retint_slate(m.group(1))


# Common shared head — fonts, tokens, base CSS. Forms only needed for the
# per-face wrapper but harmless on the procrustes one (it's a defer load).
# /fonts/fonts.css declares Newsreader (and Shippori Mincho for Japanese);
# the figure is served from the site origin, so the same-origin font-src
# and the plotly CDN allowance in the CSP are all it needs.
_SHARED_HEAD = """<head>
<meta charset="utf-8" />
<script src="/shared/tokens.js"></script>
<link rel="stylesheet" href="/fonts/fonts.css">
<link rel="stylesheet" href="/shared/base.css">
<script defer src="/shared/forms.js"></script>
"""

# Font stack for everything the figure draws (CSS and plotly alike).
FONT_STACK = "'Newsreader', 'Shippori Mincho', 'Hiragino Mincho ProN', serif"

# Common reset + plotly transparent-background overrides shared by both
# wrappers. The `_THEME_JS_PRE` block runs in both — we factor it once.
_BASE_CSS = """  html, body {
    margin: 0;
    padding: 0;
    background: transparent !important;
    overflow: hidden;
    height: 100%;
  }
  #caption-bar, #model-bar {
    display: flex;
    align-items: center;
    gap: 14px 22px;
    padding: 10px 16px 4px;
    flex-wrap: wrap;
    background: transparent;
    font-family: __FONT__;
    font-size: 15.5px;
    line-height: 1.3;
  }
  .legend-static {
    display: flex;
    gap: 6px 20px;
    align-items: center;
    color: var(--text);
    flex-wrap: wrap;
  }
  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 0;
  }
  .legend-mark {
    display: inline-block;
    width: 9px;
    height: 9px;
    background: var(--text-muted);
  }
  .legend-mark.mark-circle  { border-radius: 50%; }
  .legend-mark.mark-diamond { transform: rotate(45deg); }
  .legend-mark.mark-square  {}
  .legend-mark.mark-cross   { background: none; position: relative; }
  .legend-mark.mark-cross::before, .legend-mark.mark-cross::after {
    content: ''; position: absolute; left: 50%; top: 50%; width: 9px; height: 2px;
    background: var(--text-muted); transform: translate(-50%, -50%);
  }
  .legend-mark.mark-cross::after { transform: translate(-50%, -50%) rotate(90deg); }
  .legend-mark.mark-x { background: none; position: relative; }
  .legend-mark.mark-x::before, .legend-mark.mark-x::after {
    content: ''; position: absolute; left: 50%; top: 50%; width: 11px; height: 2px;
    background: var(--text-muted); transform: translate(-50%, -50%) rotate(45deg);
  }
  .legend-mark.mark-x::after { transform: translate(-50%, -50%) rotate(-45deg); }
  .subtitle {
    color: var(--text-muted);
    font-style: italic;
  }
  #model-bar .mode-btn {
    font-family: __FONT__;
    font-size: 15.5px;
    font-variation-settings: normal;
    letter-spacing: normal;
    text-transform: none;
    padding: 6px 14px;
    flex: 0 0 auto;
  }
  .plotly-graph-div {
    width: 100% !important;
    height: calc(100vh - 50px) !important;
    background: transparent !important;
  }
  .plot-container, .svg-container, .main-svg { background: transparent !important; }""".replace("__FONT__", FONT_STACK)


# Color + layout helpers used by all three wrappers (emitted verbatim into
# each page). `SITE_PALETTE` is the design system's light/dark table; the
# rule color is text at 12% (light) / 13% (dark), pre-composited over the
# canvas because plotly's WebGL scenes drop the alpha channel of grid colors.
_THEME_JS_HELPERS = """  var FONT = "__FONT__";
  var SITE_PALETTE = {
    light: { canvas: '#FCFCFB', elevated: '#FFFFFF', text: '#161514',
             secondary: '#42413F', muted: '#666463', accent: '#983432',
             ruleAlpha: 0.12 },
    dark:  { canvas: '#121111', elevated: '#1B1A1A', text: '#ECEBEA',
             secondary: '#BEBDBC', muted: '#949392', accent: '#DC5960',
             ruleAlpha: 0.13 }
  };

  function readTheme() {
    try {
      var t = window.parent && window.parent !== window
        ? window.parent.document.documentElement.dataset.theme
        : document.documentElement.dataset.theme;
      return t === 'dark' ? 'dark' : 'light';
    } catch (e) { return 'light'; }
  }

  // Pre-composite a 6-digit hex over an opaque background at alpha `a`.
  function compositeOver(fgHex, bgHex, a) {
    var fg = fgHex.replace('#', ''), bg = bgHex.replace('#', '');
    var out = '#';
    for (var i = 0; i < 6; i += 2) {
      var f = parseInt(fg.slice(i, i + 2), 16);
      var b = parseInt(bg.slice(i, i + 2), 16);
      out += ('0' + Math.round(f * a + b * (1 - a)).toString(16)).slice(-2);
    }
    return out;
  }

  function readThemeColors() {
    var theme = readTheme();
    document.documentElement.dataset.theme = theme;
    var p = SITE_PALETTE[theme];
    return {
      theme: theme,
      canvas:    p.canvas,
      text:      p.text,
      secondary: p.secondary,
      muted:     p.muted,
      elevated:  p.elevated,
      accent:    p.accent,
      // Hairline rule (gridlines, hover-label border) and the axis line,
      // which sits a step above it so the box stays legible.
      grid:      compositeOver(p.text, p.canvas, p.ruleAlpha),
      axisLine:  compositeOver(p.text, p.canvas, 0.30),
      // Drop-line color: text at ~44% over the canvas, subordinate to the
      // markers and slightly more present than the grid.
      dropLine:  compositeOver(p.text, p.canvas, 0.44),
    };
  }

  // Style a 3D scene in place: transparent ground, hairline grid and spikes,
  // Newsreader ticks and titles, a fixed cube camera.
  function styleScene(scene, c) {
    scene.bgcolor = 'rgba(0,0,0,0)';
    scene.domain = { x: [-0.08, 1.08], y: [-0.08, 1.08] };
    scene.aspectmode = 'cube';
    scene.camera = { eye: { x: 1.45, y: 1.45, z: 1.30 } };
    ['xaxis', 'yaxis', 'zaxis'].forEach(function(ax) {
      if (!scene[ax]) scene[ax] = {};
      scene[ax].gridcolor = c.grid;
      scene[ax].linecolor = c.axisLine;
      scene[ax].zerolinecolor = c.axisLine;
      scene[ax].color = c.text;
      scene[ax].showbackground = false;
      // Hover spikes: hairline in the muted text color (plotly's default is a
      // heavy bright line).
      scene[ax].spikecolor = c.muted;
      scene[ax].spikethickness = 1;
      scene[ax].tickfont = { color: c.muted, family: FONT, size: 12 };
      if (typeof scene[ax].title === 'string') scene[ax].title = { text: scene[ax].title };
      if (!scene[ax].title) scene[ax].title = {};
      scene[ax].title.font = { color: c.secondary, family: FONT, size: 14 };
    });
    return scene;
  }

  function baseLayout(scene, c) {
    return {
      scene: scene,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 0, r: 0, t: 0, b: 0 },
      showlegend: false,
      autosize: true,
      font: { color: c.text, family: FONT, size: 13 },
      hoverlabel: {
        bgcolor: c.elevated,
        bordercolor: c.grid,
        font: { family: FONT, color: c.text, size: 14 },
        align: 'left',
      },
    };
  }

  // Plotly rasterizes 3D text on first draw, so wait for Newsreader to load
  // (or give up after a beat and draw with the fallback).
  function whenFontsReady(cb) {
    var done = false;
    function go() { if (!done) { done = true; cb(); } }
    setTimeout(go, 2500);
    try {
      Promise.all([
        document.fonts.load('400 13px "Newsreader"'),
        document.fonts.load('600 13px "Newsreader"'),
        document.fonts.load('italic 400 13px "Newsreader"'),
      ]).then(go, go);
    } catch (e) { go(); }
  }""".replace("__FONT__", FONT_STACK.replace('"', '\\"'))


def _legend_marker_html() -> str:
    """Static legend bar markup with per-model marker shapes."""
    items = []
    for m in MODELS:
        sym = MODEL_MARKERS[m]
        # Map plotly symbols to CSS classes.
        cls_map = {"circle": "mark-circle", "diamond": "mark-diamond",
                   "square": "mark-square", "cross": "mark-cross", "x": "mark-x"}
        cls = cls_map.get(sym, "mark-circle")
        label = m.replace("_", "-")
        items.append(
            f'    <span class="legend-item"><span class="legend-mark {cls}"></span>{label}</span>'
        )
    return "\n".join(items)


def wrap_procrustes(
    raw_path: Path, out_path: Path,
    overlay_scene: str = "scene6",
    show_droplines: bool = False,
) -> None:
    """Wrap the procrustes-overlay raw HTML.

    Drop-line traces (sentinel ``name="_droplines"`` from
    ``26_v3_quadrant_procrustes.py``) are always present in the
    embedded data, but only rendered when ``show_droplines`` is true —
    they read as a depth-anchor for some viewers and as visual clutter
    for others, so the wrapper exposes the choice via ``--drop-lines``
    rather than rebaking the producer for each preference.
    """
    raw = raw_path.read_text(encoding="utf-8")
    plotly_block = _extract_plotly_block(raw)

    legend_html = _legend_marker_html()
    show_droplines_js = "true" if show_droplines else "false"

    out = f"""<html>
{_SHARED_HEAD}<style>
{_BASE_CSS}
</style></head>
<body>
<div id="caption-bar">
  <div class="legend-static">
{legend_html}
  </div>
  <span class="subtitle">All five models overlaid on gemma</span>
</div>
{plotly_block}
<script>
(function() {{
  var OVERLAY_SCENE = '{overlay_scene}';
  var SHOW_DROPLINES = {show_droplines_js};
  var div = null;
  var origData = null;
  var origLayout = null;

{_THEME_JS_HELPERS}

  function buildLayout(c) {{
    var origScene = origLayout[OVERLAY_SCENE] || origLayout.scene || {{}};
    var scene = styleScene(JSON.parse(JSON.stringify(origScene)), c);
    return baseLayout(scene, c);
  }}

  function render() {{
    if (!div || !origData || !window.Plotly) return;
    var c = readThemeColors();
    var traces = origData
      .filter(function (t) {{ return (t.scene || 'scene') === OVERLAY_SCENE; }})
      // Drop-line traces (sentinel name="_droplines" set by
      // scripts/local/26_v3_quadrant_procrustes.py) are opt-in via
      // the wrapper's --drop-lines flag; default off because the
      // overlay reads cleaner without depth stems.
      .filter(function (t) {{ return SHOW_DROPLINES || t.name !== '_droplines'; }})
      .map(function (t) {{
        var clone = JSON.parse(JSON.stringify(t));
        clone.scene = 'scene';
        if (clone.marker && clone.marker.line) {{
          clone.marker.line.color = c.canvas;
        }}
        // Drop-lines repaint per active theme — the baked rgba
        // mid-grey washes out to near-white when composited over
        // the light canvas, so we override at render time with the
        // palette's text color over the canvas instead.
        if (clone.name === '_droplines' && clone.line) {{
          clone.line.color = c.dropLine;
        }}
        return clone;
      }});
    Plotly.react(div, traces, buildLayout(c), {{ responsive: true, displayModeBar: false }});
    setTimeout(function() {{ Plotly.Plots.resize(div); }}, 50);
  }}

  function init() {{
    div = document.querySelector('.plotly-graph-div');
    if (!div || !div.data || !window.Plotly) {{
      return setTimeout(init, 50);
    }}
    origData = JSON.parse(JSON.stringify(div.data));
    origLayout = JSON.parse(JSON.stringify(div.layout));
    render();
    try {{
      if (window.parent && window.parent !== window) {{
        var obs = new MutationObserver(render);
        obs.observe(window.parent.document.documentElement, {{ attributes: true, attributeFilter: ['data-theme'] }});
      }}
    }} catch (e) {{}}
    window.addEventListener('resize', function() {{ if (window.Plotly && div) Plotly.Plots.resize(div); }});
  }}

  document.documentElement.dataset.theme = readTheme();
  window.addEventListener('load', function() {{ whenFontsReady(function() {{ setTimeout(init, 50); }}); }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(out, encoding="utf-8")
    print(f"  wrote {out_path}")


# Quadrant colors for the wild-faces legend bar. Sourced from the
# canonical ``interpretability_experiments.kaomoji.quadrants`` module so the static legend
# matches the per-face proportional-blend colors plotly draws.
WILD_QUADRANTS = QUADRANT_ORDER_SPLIT
WILD_QUADRANT_COLORS = {q: SITE_QUADRANT_COLORS[q] for q in WILD_QUADRANTS}
# Surface marker shapes match the plotly markers script 67 emits:
# circle = Claude Code journal only, diamond = any claude.ai export,
# square = neither (HF corpus only, from another contributor).
WILD_SURFACE_LEGEND = (
    ("circle",  "Claude Code"),
    ("diamond", "claude.ai"),
    ("square",  "other"),
)


def _wild_legend_html() -> str:
    """Static legend bar: 6 quadrant color swatches plus 3 surface-shape
    markers. Per-face plotly markers carry an RGB blend of these
    quadrant colors weighted by the BoL share, so the legend exposes
    the pure-quadrant reference even though no individual point lands
    exactly on it."""
    items = []
    for q in WILD_QUADRANTS:
        color = WILD_QUADRANT_COLORS[q]
        items.append(
            f'    <span class="legend-item">'
            f'<span class="legend-mark mark-circle" style="background: {color};"></span>'
            f'{q}</span>'
        )
    for shape, label in WILD_SURFACE_LEGEND:
        items.append(
            f'    <span class="legend-item">'
            f'<span class="legend-mark mark-{shape}"></span>'
            f'{label}</span>'
        )
    return "\n".join(items)


def wrap_wild_faces(raw_path: Path, out_path: Path) -> None:
    """Wrap the wild-faces PCA HTML keeping only the BoL-quadrant-by-
    surface scene (left, "scene"), dropping the KMeans cluster scene
    (right, "scene2"). Filters out the dummy legend-entry traces script
    67 emits with all-null coordinates for plotly's built-in legend
    since the wrapper renders a static HTML legend bar instead."""
    raw = raw_path.read_text(encoding="utf-8")
    plotly_block = _extract_plotly_block(raw)
    legend_html = _wild_legend_html()

    out = f"""<html>
{_SHARED_HEAD}<style>
{_BASE_CSS}
</style></head>
<body>
<div id="caption-bar">
  <div class="legend-static">
{legend_html}
  </div>
  <span class="subtitle">Color blends per-face BoL shares · shape = deployment surface</span>
</div>
{plotly_block}
<script>
(function() {{
  var div = null;
  var origData = null;
  var origLayout = null;

{_THEME_JS_HELPERS}

  function buildLayout(c) {{
    var origScene = origLayout.scene || {{}};
    var scene = styleScene(JSON.parse(JSON.stringify(origScene)), c);
    return baseLayout(scene, c);
  }}

  function render() {{
    if (!div || !origData || !window.Plotly) return;
    var c = readThemeColors();
    // Keep only scene1 traces with real data. Excludes scene2 (KMeans
    // cluster scene we're dropping) and the dummy legend-entry traces
    // script 67 emits with all-null coords for plotly's built-in
    // legend, which we render as a static HTML bar instead. Filter on
    // legendgroup since recent plotly serializes large arrays as
    // {{bdata,dtype,shape}} objects rather than plain JS arrays — so
    // a t.x.length check rejects every real data trace too.
    var traces = origData
      .filter(function (t) {{ return (t.scene || 'scene') === 'scene'; }})
      .filter(function (t) {{
        return t.legendgroup && t.legendgroup.indexOf('left-data-') === 0;
      }})
      .map(function (t) {{
        var clone = JSON.parse(JSON.stringify(t));
        clone.scene = 'scene';
        if (clone.marker && clone.marker.line) {{
          clone.marker.line.color = c.canvas;
        }}
        return clone;
      }});
    Plotly.react(div, traces, buildLayout(c), {{ responsive: true, displayModeBar: false }});
    setTimeout(function() {{ Plotly.Plots.resize(div); }}, 50);
  }}

  function init() {{
    div = document.querySelector('.plotly-graph-div');
    if (!div || !div.data || !window.Plotly) {{
      return setTimeout(init, 50);
    }}
    origData = JSON.parse(JSON.stringify(div.data));
    origLayout = JSON.parse(JSON.stringify(div.layout));
    render();
    try {{
      if (window.parent && window.parent !== window) {{
        var obs = new MutationObserver(render);
        obs.observe(window.parent.document.documentElement, {{ attributes: true, attributeFilter: ['data-theme'] }});
      }}
    }} catch (e) {{}}
    window.addEventListener('resize', function() {{ if (window.Plotly && div) Plotly.Plots.resize(div); }});
  }}

  document.documentElement.dataset.theme = readTheme();
  window.addEventListener('load', function() {{ whenFontsReady(function() {{ setTimeout(init, 50); }}); }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(out, encoding="utf-8")
    print(f"  wrote {out_path}")


def wrap_per_face(raw_path: Path, out_path: Path, subtitles: list[str]) -> None:
    """Wrap the per-face PCA HTML with a model-toggle bar.

    The raw plotly is expected to have one scene per model in MODELS
    order (scene, scene2, ..., sceneN), each containing per-face markers
    in PC1×PC2×PC3.
    """
    raw = raw_path.read_text(encoding="utf-8")
    plotly_block = _extract_plotly_block(raw)

    toggle_buttons = []
    for i, m in enumerate(MODELS):
        active = " active" if i == 0 else ""
        label = m.replace("_", "-")
        toggle_buttons.append(
            f'    <button class="mode-btn{active}" data-model="{i}">{label}</button>'
        )
    toggle_html = "\n".join(toggle_buttons)
    subtitles_js = "[" + ", ".join(f'"{s}"' for s in subtitles) + "]"

    out = f"""<html>
{_SHARED_HEAD}<style>
{_BASE_CSS}
</style></head>
<body>
<div id="model-bar">
  <div class="mode-toggles" id="model-toggles">
{toggle_html}
  </div>
  <span class="subtitle" id="model-subtitle">{subtitles[0]}</span>
</div>
{plotly_block}
<script>
(function() {{
  var SUBTITLES = {subtitles_js};
  var SCENES = ['scene', 'scene2', 'scene3', 'scene4', 'scene5'];
  var activeModel = 0;
  var div = null;
  var origData = null;
  var origLayout = null;

{_THEME_JS_HELPERS}

  function buildLayout(idx, c) {{
    var sceneKey = SCENES[idx];
    var origScene = origLayout[sceneKey] || origLayout.scene || {{}};
    var scene = styleScene(JSON.parse(JSON.stringify(origScene)), c);
    return baseLayout(scene, c);
  }}

  function showModel(idx) {{
    activeModel = idx;
    document.getElementById('model-subtitle').textContent = SUBTITLES[idx];
    if (!div || !origData || !window.Plotly) return;
    var c = readThemeColors();
    var sceneKey = SCENES[idx];
    var traces = origData
      .filter(function (t) {{ return (t.scene || 'scene') === sceneKey; }})
      .map(function (t) {{
        var clone = JSON.parse(JSON.stringify(t));
        clone.scene = 'scene';
        if (!clone.marker) clone.marker = {{}};
        if (!clone.marker.line) clone.marker.line = {{}};
        clone.marker.line.color = c.canvas;
        clone.marker.line.width = 0.5;
        return clone;
      }});
    Plotly.react(div, traces, buildLayout(idx, c), {{ responsive: true, displayModeBar: false }});
    setTimeout(function() {{ Plotly.Plots.resize(div); }}, 50);
  }}

  function init() {{
    div = document.querySelector('.plotly-graph-div');
    var toggles = document.getElementById('model-toggles');
    if (!div || !div.data || !window.Plotly || !toggles || typeof _forms === 'undefined') {{
      return setTimeout(init, 50);
    }}
    origData = JSON.parse(JSON.stringify(div.data));
    origLayout = JSON.parse(JSON.stringify(div.layout));
    _forms.bindModeGroup(toggles, 'model', function (val) {{
      showModel(parseInt(val, 10));
    }});
    showModel(0);
    try {{
      if (window.parent && window.parent !== window) {{
        var obs = new MutationObserver(function() {{ showModel(activeModel); }});
        obs.observe(window.parent.document.documentElement, {{ attributes: true, attributeFilter: ['data-theme'] }});
      }}
    }} catch (e) {{}}
    window.addEventListener('resize', function() {{ if (window.Plotly && div) Plotly.Plots.resize(div); }});
  }}

  document.documentElement.dataset.theme = readTheme();
  window.addEventListener('load', function() {{ whenFontsReady(function() {{ setTimeout(init, 50); }}); }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(out, encoding="utf-8")
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrap raw plotly 3D HTML files with the a9l.im "
                    "site-themed shell.",
    )
    parser.add_argument(
        "--drop-lines", action="store_true",
        help="Render the procrustes overlay's vertical depth-anchor "
             "drop lines (off by default — the producer always emits "
             "them, but the wrapper hides them unless this flag is "
             "set so the cleaner no-stem view is the default).",
    )
    args = parser.parse_args()

    if not BLOG_ASSETS_DIR.parent.exists():
        print(f"blog-assets parent does not exist: {BLOG_ASSETS_DIR.parent}", file=sys.stderr)
        sys.exit(1)
    BLOG_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== procrustes overlay ===")
    if not RAW_PROCRUSTES.exists():
        print(f"  missing {RAW_PROCRUSTES} — skipping", file=sys.stderr)
    else:
        wrap_procrustes(
            RAW_PROCRUSTES, OUT_PROCRUSTES,
            overlay_scene="scene6",
            show_droplines=args.drop_lines,
        )

    print("\n=== per-face PCA toggle ===")
    if not RAW_PER_FACE.exists():
        print(f"  missing {RAW_PER_FACE} — generate it first via scripts/local/97_build_per_face_pca_3d.py")
    else:
        # Subtitles read off the per-face JSON sidecar written by the
        # generator; default fallback is generic.
        sidecar = RAW_PER_FACE.with_suffix(".meta.json")
        if sidecar.exists():
            import json
            meta = json.loads(sidecar.read_text())
            subtitles = [meta[m] for m in MODELS]
        else:
            subtitles = [f"{m} per-face centroids" for m in MODELS]
        wrap_per_face(RAW_PER_FACE, OUT_PER_FACE, subtitles)

    print("\n=== wild faces PCA (BoL quadrant by deployment surface) ===")
    if not RAW_WILD_FACES.exists():
        print(f"  missing {RAW_WILD_FACES} — generate via scripts/67_wild_residual.py", file=sys.stderr)
    else:
        wrap_wild_faces(RAW_WILD_FACES, OUT_WILD_FACES)


if __name__ == "__main__":
    main()

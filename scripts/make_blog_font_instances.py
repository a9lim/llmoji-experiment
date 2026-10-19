"""Cut the static TTFs that matplotlib uses for the blog figures.

matplotlib cannot select axes of a variable font, so each style is pinned
with fontTools.varLib.instancer and renamed to a plain static family:

    Newsreader-Regular.ttf    wght 400, opsz 14 (a tick label's optical size)
    Newsreader-SemiBold.ttf   wght 600, opsz 14
    Newsreader-Italic.ttf     wght 400, opsz 14, italic
    STIXTwoText-Regular.ttf   wght 400
    STIXTwoMath-Regular.ttf   copied as is (already static)

STIX Two sets the Greek, operators and scripts Newsreader lacks, matching
a9l.im's fallback stack. Output goes to data/fonts/ beside
NotoEmoji-Regular.ttf, with each family's OFL text. Sources are the variable
Newsreader roman and italic (https://github.com/productiontype/Newsreader),
the variable STIX Two Text roman and STIX Two Math (google/fonts ofl/):

    python scripts/make_blog_font_instances.py NEWSREADER_ROMAN_VF.ttf \
        NEWSREADER_ITALIC_VF.ttf STIXTWOTEXT_VF.ttf STIXTWOMATH.ttf
"""
from __future__ import annotations

import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "fonts"
OPSZ = 14

# (source key, filename stem, axis pins, family, subfamily, usWeightClass, italic)
STYLES = (
    ("roman",  "Newsreader-Regular",  {"wght": 400, "opsz": OPSZ}, "Newsreader",          "Regular", 400, False),
    ("roman",  "Newsreader-SemiBold", {"wght": 600, "opsz": OPSZ}, "Newsreader SemiBold", "Regular", 600, False),
    ("italic", "Newsreader-Italic",   {"wght": 400, "opsz": OPSZ}, "Newsreader",          "Italic",  400, True),
    ("stix",   "STIXTwoText-Regular", {"wght": 400},               "STIX Two Text",       "Regular", 400, False),
)


def _set_names(font: TTFont, family: str, sub: str, full: str, ps: str) -> None:
    name = font["name"]
    # Drop typographic-family and variable-font records; the static family
    # is carried by IDs 1/2 alone so every loader reads the same thing.
    name.names = [n for n in name.names if n.nameID not in (1, 2, 3, 4, 6, 16, 17, 25)]
    for nid, val in ((1, family), (2, sub), (3, ps), (4, full), (6, ps)):
        name.setName(val, nid, 3, 1, 0x409)
        name.setName(val, nid, 1, 0, 0)


def main(roman: str, italic: str, stix_text: str, stix_math: str) -> None:
    import shutil
    sources = {"roman": roman, "italic": italic, "stix": stix_text}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, stem, pins, family, sub, weight_class, is_italic in STYLES:
        vf = TTFont(sources[key])
        inst = instancer.instantiateVariableFont(vf, pins, inplace=False)
        full = family if sub == "Regular" else f"{family} {sub}"
        _set_names(inst, family, sub, full, stem)
        inst["OS/2"].usWeightClass = weight_class
        if "STAT" in inst:
            del inst["STAT"]
        out = OUT_DIR / f"{stem}.ttf"
        inst.save(out)
        print(f"wrote {out}")
    out = OUT_DIR / "STIXTwoMath-Regular.ttf"
    shutil.copyfile(stix_math, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:5])

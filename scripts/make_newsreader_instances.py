"""Cut the static Newsreader TTFs that matplotlib uses for the blog figures.

matplotlib cannot select axes of a variable font, so each style is pinned
with fontTools.varLib.instancer (wght and opsz = 14, the optical size of a
tick label) and renamed to a plain static family:

    Newsreader-Regular.ttf    wght 400, roman
    Newsreader-SemiBold.ttf   wght 600, roman
    Newsreader-Italic.ttf     wght 400, italic

Output goes to data/fonts/ beside NotoEmoji-Regular.ttf; the OFL text stays
with the instances. Source is the upstream variable Newsreader (OFL,
https://github.com/productiontype/Newsreader), roman and italic TTFs:

    python scripts/make_newsreader_instances.py ROMAN_VF.ttf ITALIC_VF.ttf
"""
from __future__ import annotations

import sys
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "fonts"
OPSZ = 14

# (source key, filename stem, wght, family, subfamily, usWeightClass, italic)
STYLES = (
    ("roman",  "Newsreader-Regular",  400, "Newsreader",          "Regular",  400, False),
    ("roman",  "Newsreader-SemiBold", 600, "Newsreader SemiBold", "Regular",  600, False),
    ("italic", "Newsreader-Italic",   400, "Newsreader",          "Italic",   400, True),
)


def _set_names(font: TTFont, family: str, sub: str, full: str, ps: str) -> None:
    name = font["name"]
    # Drop typographic-family and variable-font records; the static family
    # is carried by IDs 1/2 alone so every loader reads the same thing.
    name.names = [n for n in name.names if n.nameID not in (1, 2, 3, 4, 6, 16, 17, 25)]
    for nid, val in ((1, family), (2, sub), (3, ps), (4, full), (6, ps)):
        name.setName(val, nid, 3, 1, 0x409)
        name.setName(val, nid, 1, 0, 0)


def main(roman: str, italic: str) -> None:
    sources = {"roman": roman, "italic": italic}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, stem, wght, family, sub, weight_class, is_italic in STYLES:
        vf = TTFont(sources[key])
        inst = instancer.instantiateVariableFont(
            vf, {"wght": wght, "opsz": OPSZ}, inplace=False,
        )
        full = family if sub == "Regular" else f"{family} {sub}"
        _set_names(inst, family, sub, full, stem)
        inst["OS/2"].usWeightClass = weight_class
        if "STAT" in inst:
            del inst["STAT"]
        out = OUT_DIR / f"{stem}.ttf"
        inst.save(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

#!/usr/bin/env python3
"""One-shot migration: per-run jsonl files → merged emotional_raw.jsonl.

Pre-2026-05-08 layout (deprecated):

    data/harness/claude-runs/run-{0..7}.jsonl                     # naturalistic v3
    data/harness/claude-runs-introspection/run-0.jsonl            # introspection v3
    data/harness/claude-runs-v4-extension/run-{0..7}.jsonl        # naturalistic v4-ext
    + matching run-N_summary.tsv companions

Post-2026-05-08 layout:

    data/harness/claude/emotional_raw.jsonl              # naturalistic (v3 + v4-ext)
    data/harness/claude_intro_v7/emotional_raw.jsonl     # introspection

Mirrors local's ``data/local/<model>/emotional_raw.jsonl`` convention so
harness and local pipelines have full schema parity.

Per-row schema rewrite:
  - ``response_text`` → ``text``  (parity with local)
  - ``preamble`` set to ``"none"`` for naturalistic / v4-ext rows that
    lacked the field; introspection rows already have ``preamble="introspection"``.
  - ``run_index`` field added, derived from the filename.

Rows are sorted by (run_index, prompt_id) for deterministic diffs.

Usage:
    python scripts/harness/migrate_to_merged_emotional_raw.py [--dry-run] [--no-delete]

By default, writes the new files and deletes the old dirs in one pass.
``--dry-run`` reports what would happen without touching disk.
``--no-delete`` writes the new files but leaves the old dirs in place
(useful for parity-checking before committing).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from llmoji_experiment.claude_gt import (
    CLAUDE_GT_DIR,
    CLAUDE_GT_INTRO_DIR,
    EMOTIONAL_RAW_FILENAME,
)
from llmoji_experiment.config import DATA_DIR

# Old dirs.
LEGACY_NATURALISTIC_DIR = DATA_DIR / "harness" / "claude-runs"
LEGACY_INTROSPECTION_DIR = DATA_DIR / "harness" / "claude-runs-introspection"
LEGACY_V4_EXT_DIR = DATA_DIR / "harness" / "claude-runs-v4-extension"

_RUN_FILENAME_RE = re.compile(r"^run-(\d+)\.jsonl$")


def _find_run_files(d: Path) -> list[tuple[int, Path]]:
    """[(run_index, path)] sorted by index. Empty if dir is missing."""
    if not d.exists():
        return []
    out: list[tuple[int, Path]] = []
    for p in d.iterdir():
        m = _RUN_FILENAME_RE.match(p.name)
        if m is None:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rewrite_row(row: dict, run_index: int, default_preamble: str) -> dict:
    """Apply the schema rewrites for one row.

    - response_text → text (rename)
    - preamble defaulted if missing
    - run_index stamped
    - field order canonicalized
    """
    out: dict = {}
    out["prompt_id"] = row.get("prompt_id")
    out["quadrant"] = row.get("quadrant")
    out["condition"] = row.get("condition", "direct")
    out["preamble"] = row.get("preamble", default_preamble)
    out["seed"] = row.get("seed", 0)
    if "prompt_text" in row:
        out["prompt_text"] = row["prompt_text"]
    # response_text → text rename. Both keys absent on error rows.
    if "text" in row:
        out["text"] = row["text"]
    elif "response_text" in row:
        out["text"] = row["response_text"]
    if "first_word" in row:
        out["first_word"] = row["first_word"]
    if "n_response_chars" in row:
        out["n_response_chars"] = row["n_response_chars"]
    if "model_id" in row:
        out["model_id"] = row["model_id"]
    if "ts" in row:
        out["ts"] = row["ts"]
    out["run_index"] = run_index
    # Forward any unknown fields verbatim (defensive).
    for k, v in row.items():
        if k in {
            "prompt_id", "quadrant", "condition", "preamble", "seed",
            "prompt_text", "text", "response_text", "first_word",
            "n_response_chars", "model_id", "ts", "run_index",
        }:
            continue
        out[k] = v
    return out


def _migrate_arm(
    legacy_dirs: list[Path],
    out_path: Path,
    default_preamble: str,
    *,
    dry_run: bool,
) -> int:
    """Pool rows from one or more legacy dirs into a single merged file.

    Returns the number of rows written (or that would be written).
    """
    pooled: list[dict] = []
    for d in legacy_dirs:
        runs = _find_run_files(d)
        for run_index, path in runs:
            rows = _read_jsonl(path)
            print(f"  read {path.relative_to(DATA_DIR.parent)}: {len(rows)} rows  "
                  f"(run_index={run_index})")
            for r in rows:
                pooled.append(_rewrite_row(r, run_index, default_preamble))

    # Deterministic order: (run_index, prompt_id, quadrant). prompt_id is
    # unique per (run_index, quadrant) under the single-condition single-
    # seed protocol, but include quadrant in the key to be defensive.
    pooled.sort(key=lambda r: (
        int(r.get("run_index", 0)),
        str(r.get("prompt_id", "")),
        str(r.get("quadrant", "")),
    ))

    print(f"  → {out_path.relative_to(DATA_DIR.parent)}: {len(pooled)} rows")
    if dry_run:
        print(f"  [dry-run] would write {len(pooled)} rows")
        return len(pooled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in pooled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {out_path}")
    return len(pooled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would happen without writing or deleting.",
    )
    parser.add_argument(
        "--no-delete", action="store_true",
        help="Write new files but leave old dirs in place. Use for "
             "parity-checking before committing the cleanup.",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Naturalistic arm: claude-runs/ + claude-runs-v4-extension/ → claude/")
    print("=" * 72)
    nat_count = _migrate_arm(
        [LEGACY_NATURALISTIC_DIR, LEGACY_V4_EXT_DIR],
        CLAUDE_GT_DIR / EMOTIONAL_RAW_FILENAME,
        default_preamble="none",
        dry_run=args.dry_run,
    )
    print()
    print("=" * 72)
    print("Introspection arm: claude-runs-introspection/ → claude_intro_v7/")
    print("=" * 72)
    intro_count = _migrate_arm(
        [LEGACY_INTROSPECTION_DIR],
        CLAUDE_GT_INTRO_DIR / EMOTIONAL_RAW_FILENAME,
        default_preamble="introspection",
        dry_run=args.dry_run,
    )

    total = nat_count + intro_count
    print()
    print("=" * 72)
    print(f"migration complete: {total} rows pooled "
          f"({nat_count} naturalistic + {intro_count} introspection)")
    print("=" * 72)

    if args.dry_run:
        print("\n[dry-run] no files written or deleted.")
        return
    if args.no_delete:
        print("\n[--no-delete] new files written; legacy dirs left in place. "
              "Run without --no-delete to clean up.")
        return

    print("\nRemoving legacy dirs:")
    for d in [LEGACY_NATURALISTIC_DIR, LEGACY_INTROSPECTION_DIR, LEGACY_V4_EXT_DIR]:
        if d.exists():
            shutil.rmtree(d)
            print(f"  rm -r {d.relative_to(DATA_DIR.parent)}")
        else:
            print(f"  (already gone) {d.relative_to(DATA_DIR.parent)}")
    print("\ndone.")


if __name__ == "__main__":
    main()

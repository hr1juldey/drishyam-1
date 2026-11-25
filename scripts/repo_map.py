#!/usr/bin/env python3
"""Repo map: every .py file -> top-level classes/functions + LOC. The ONLY valid
evidence for existence claims. Run: uv run python scripts/repo_map.py [--save]"""
import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "node_modules", "__pycache__", ".git", "runs", "artifacts", "embeddings"}


def main() -> None:
    rows = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in sorted(filenames):
            if not f.endswith(".py"):
                continue
            p = Path(dirpath) / f
            try:
                tree = ast.parse(p.read_text())
            except SyntaxError:
                rows.append((p, "SYNTAX-ERR", []))
                continue
            syms = [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
            loc = sum(1 for _ in p.open())
            rows.append((p.relative_to(ROOT), loc, syms))
    lines = [f"{p}  [{loc}L]  " + ", ".join(syms) for p, loc, syms in sorted(rows)]
    out = "\n".join(lines) + f"\nTOTAL: {len(rows)} py files\n"
    print(out)
    if "--save" in sys.argv:
        (ROOT / "docs" / "REPO_MAP.md").write_text(
            "# REPO_MAP (regenerate: uv run python scripts/repo_map.py --save; "
            "committed artifact — absence claims cite this)\n\n```\n" + out + "```\n")
        print("saved docs/REPO_MAP.md")


if __name__ == "__main__":
    main()

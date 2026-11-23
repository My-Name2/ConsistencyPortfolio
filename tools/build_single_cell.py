"""Bundle the valuation/ package into one self-contained script (and a one-cell notebook).

    python -B tools/build_single_cell.py

Writes valuation_single_cell.py and valuation_single_cell.ipynb at the repository root.
Rebuild after every change to the package so the cell stays in step with it.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "valuation"
# Dependency order: each module only uses names defined by modules above it.
MODULES = ["tags", "dq", "storage", "facts", "splits", "values", "periods", "prices", "ratios", "stats", "redflags", "pipeline", "sources", "workbook", "report", "reference", "cli"]
RELATIVE = re.compile(r"^(?P<indent>\s*)from \.[\w.]* import (?P<rest>.*)$")


def strip_module(name: str, text: str) -> str:
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "from __future__ import annotations":
            i += 1
            continue
        m = RELATIVE.match(line)
        if line.strip().startswith("from . import "):
            i += 1  # module objects are provided as namespaces where needed (see sources)
            continue
        if m:
            # Swallow a parenthesised multi-line import.
            if "(" in m.group("rest") and ")" not in m.group("rest"):
                while ")" not in lines[i]:
                    i += 1
            if m.group("indent"):
                out.append(f"{m.group('indent')}pass  # was: relative import (names are global in the single cell)")
            i += 1
            continue
        out.append(line)
        i += 1
    text = "\n".join(out) + "\n"
    if name == "cli":
        # The argparse entry point and its __main__ guard do not belong in a notebook cell.
        cut = text.index("\ndef main(")
        text = text[:cut] + "\n"
    if name == "sources":
        tickers = (PKG / "default_tickers.txt").read_text(encoding="utf-8")
        text += (
            "\n\nDEFAULT_TICKERS_TEXT = " + repr(tickers) + "\n\n\n"
            "def default_tickers() -> list[str]:\n"
            "    return parse_ticker_list(DEFAULT_TICKERS_TEXT)\n\n\n"
            "# cli.py refers to these as sources.<name>; in the single cell they are plain globals.\n"
            "import types as _types\n\n"
            "sources = _types.SimpleNamespace(\n"
            "    resolve=resolve, fetch_companyfacts=fetch_companyfacts, fetch_prices=fetch_prices,\n"
            "    default_tickers=default_tickers, read_ticker_file=read_ticker_file, parse_ticker_list=parse_ticker_list, ticker_map=ticker_map,\n"
            ")\n"
        )
    return text


def build() -> str:
    parts = [(ROOT / "tools" / "single_cell_header.py").read_text(encoding="utf-8")]
    for name in MODULES:
        body = strip_module(name, (PKG / f"{name}.py").read_text(encoding="utf-8"))
        parts.append(f"\n# {'=' * 30} valuation/{name}.py {'=' * 30}\n{body}")
    parts.append((ROOT / "tools" / "single_cell_footer.py").read_text(encoding="utf-8"))
    source = "".join(parts)
    ast.parse(source)  # fail loudly if the bundle does not compile
    leftovers = [l for l in source.splitlines() if RELATIVE.match(l)]
    if leftovers:
        raise SystemExit(f"relative imports left in the bundle: {leftovers[:3]}")
    return source


def main() -> None:
    source = build()
    (ROOT / "valuation_single_cell.py").write_text(source, encoding="utf-8")
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "id": "valuation-review", "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "valuation_single_cell.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"wrote valuation_single_cell.py ({len(source.splitlines())} lines) and valuation_single_cell.ipynb")


if __name__ == "__main__":
    sys.exit(main())

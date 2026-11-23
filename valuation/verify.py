"""Cross-check a recalculated workbook against the Python engine (used by tests)."""

from __future__ import annotations

import io
import math

import pandas as pd
from openpyxl import load_workbook

from .pipeline import ratio_inputs
from .ratios import RATIOS, compute_all
from .stats import SERIES
from .workbook import STAT_COLUMNS


def _blank(v) -> bool:
    if v is None or v == "":
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _same(a, b, tol=1e-6) -> bool:
    if _blank(a) and _blank(b):
        return True
    if _blank(b):
        b = None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if hasattr(a, "date") and hasattr(b, "date"):
        return pd.Timestamp(a).normalize() == pd.Timestamp(b).normalize()
    return str(a) == str(b)


def compare(review, recalculated: bytes) -> list[str]:
    wb = load_workbook(io.BytesIO(recalculated), data_only=True)
    problems: list[str] = []
    # TTM_Stats
    ws = wb["TTM_Stats"]
    keys = [k for k, _ in STAT_COLUMNS]
    stats = review.stats.set_index("series")
    for i, s in enumerate(SERIES):
        r = 5 + i
        for c, key in enumerate(keys, start=1):
            if key in ("series", "basis", "focus", "start", "daily_ratio"):
                continue
            got, want = ws.cell(row=r, column=c).value, stats.at[s.key, key]
            if not _same(got, want):
                problems.append(f"TTM_Stats {s.key}.{key}: workbook {got!r} vs python {want!r}")
    # TTM_Quarterly series values
    ws = wb["TTM_Quarterly"]
    header = {ws.cell(row=1, column=c).value: c for c in range(1, ws.max_column + 1)}
    ttm = review.ttm.reset_index(drop=True)
    for s in SERIES:
        unit = {"money": "", "shares": " (shares)", "per_share": " (per share)"}[s.unit]
        col = header[f"{s.label} [{s.basis}]{unit}"]
        for i in range(len(ttm)):
            got, want = ws.cell(row=i + 2, column=col).value, ttm.at[i, s.key]
            if not _same(got, want):
                problems.append(f"TTM_Quarterly row {i + 2} {s.key}: workbook {got!r} vs python {want!r}")
    # Ratios_Quarterly
    ws = wb["Ratios_Quarterly"]
    for i, row in ttm.iterrows():
        values = compute_all(ratio_inputs(row))
        for k, ratio in enumerate(RATIOS):
            got = ws.cell(row=i + 2, column=6 + k).value
            want = values[ratio.key][0]
            if not _same(got, want, tol=1e-6):
                problems.append(f"Ratios_Quarterly row {i + 2} {ratio.key}: workbook {got!r} vs python {want!r}")
    return problems

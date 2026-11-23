"""Compare observation counts with a CompanyCharts-style reference workbook.

The reference reports, per ticker and metric M in Rev, EPS, FCF, GP, CFO, EQ, the columns
"{M} TTM #Obs", "#Inc", "#Rec", "#Inc Adj", "#Rec Adj" over a trailing window of observations.

The written spec's rules are the defaults. Each reference convention found by comparing the
two is a switch in `ReferenceOptions` so its effect can be seen; none changes the review itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

METRIC_COLUMNS = {"Rev": "ttm_revenue", "EPS": "eps_ttm", "FCF": "ttm_fcf", "GP": "ttm_gross_profit", "CFO": "ttm_ocf", "EQ": "equity"}
FIELDS = ["#Obs", "#Inc", "#Rec", "#Inc Adj", "#Rec Adj"]


@dataclass
class ReferenceOptions:
    window: int = 57  # trailing TTM observations the reference counts over
    equity_window: int = 57  # the reference uses 60 for equity (no TTM build needed)
    record_counts_first: bool = False  # spec: the first observation is the reference, not a record
    emulate_first_ytd: bool = False  # reference artifact: first window value of YTD-built series (CFO, FCF) includes the prior YTD
    missing_capex_as_zero: bool = False  # reference fills FCF = CFO where capex is missing

    @classmethod
    def reference_compatible(cls) -> "ReferenceOptions":
        return cls(window=57, equity_window=60, record_counts_first=True, emulate_first_ytd=True, missing_capex_as_zero=True)


def _num(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


def window_counts(values: list, window: int, record_counts_first: bool = False) -> dict:
    vals = list(values[-window:]) if window else list(values)
    vals = [float(v) if _num(v) else None for v in vals]
    present = [v for v in vals if v is not None]
    pairs = [(vals[i], vals[i + 1]) for i in range(len(vals) - 1) if vals[i] is not None and vals[i + 1] is not None]

    def records(seq):
        n, best = 0, None
        for v in seq:
            if best is None:
                n += 1 if record_counts_first else 0
                best = v
            else:
                n += 1 if v > best else 0
                best = max(best, v)
        return n

    return {
        "#Obs": len(present),
        "#Inc": sum(1 for a, b in pairs if b > a),
        "#Rec": records(present),
        "#Inc Adj": sum(1 for a, b in pairs if a > 0 and b > 0 and b > a),
        "#Rec Adj": records([v for v in present if v > 0]),
    }


def _first_window_ytd(review, key: str, ttm: pd.DataFrame, window: int) -> tuple[int | None, float | None]:
    """Index of the first non-blank value in the window and the YTD the reference adds to it."""
    col = {"ocf": "ttm_ocf", "capex": "ttm_capex"}[key]
    values = pd.to_numeric(ttm[col], errors="coerce")
    start = max(0, len(ttm) - window)
    idx = next((i for i in range(start, len(ttm)) if pd.notna(values.iloc[i])), None)
    if idx is None:
        return None, None
    cal = review.calendar
    quarter = next((q for q in cal.quarters if q.end == pd.Timestamp(ttm.at[idx, "quarter_end"])), None)
    four = cal.previous_quarters(quarter, 4) if quarter is not None else None
    if not four:
        return idx, None
    earliest = four[0]
    if earliest.k == 1:
        return idx, 0.0
    prior = cal.quarter(earliest.fy, earliest.k - 1)
    row = review.tables[key].get(cal.years[earliest.fy].start, prior.end) if prior is not None else None
    return idx, (None if row is None else float(row["value"]))


def series_for(review, metric: str, options: ReferenceOptions) -> list:
    ttm = review.ttm.reset_index(drop=True)
    if ttm.empty:
        return []
    window = options.equity_window if metric == "EQ" else options.window
    if metric == "FCF":
        ocf = pd.to_numeric(ttm["ttm_ocf"], errors="coerce")
        capex = pd.to_numeric(ttm["ttm_capex"], errors="coerce")
        if options.missing_capex_as_zero:
            capex = capex.where(capex.notna() | ocf.isna(), 0.0)
        values = list(ocf - capex)
        if options.emulate_first_ytd:
            i, extra_ocf = _first_window_ytd(review, "ocf", ttm, window)
            _, extra_capex = _first_window_ytd(review, "capex", ttm, window)
            if i is not None and extra_ocf is not None and _num(values[i]):
                values[i] = values[i] + extra_ocf - (extra_capex or 0.0)
        return values
    values = list(pd.to_numeric(ttm[METRIC_COLUMNS[metric]], errors="coerce"))
    if metric == "CFO" and options.emulate_first_ytd:
        i, extra = _first_window_ytd(review, "ocf", ttm, window)
        if i is not None and extra is not None:
            values[i] = values[i] + extra
    return values


def review_counts(review, options: ReferenceOptions | None = None) -> dict[str, dict]:
    options = options or ReferenceOptions()
    out = {}
    for metric in METRIC_COLUMNS:
        window = options.equity_window if metric == "EQ" else options.window
        out[metric] = window_counts(series_for(review, metric, options), window, options.record_counts_first)
    return out


def load_reference(path) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=str)
    if "Ticker" not in frame.columns:
        raise ValueError("Reference workbook needs a 'Ticker' column.")
    return frame.set_index(frame["Ticker"].str.strip().str.upper())


def compare(reviews: dict, reference: pd.DataFrame, options: ReferenceOptions | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(agreement per metric and field, per-ticker mine/theirs) for reviews keyed by ticker."""
    options = options or ReferenceOptions()
    rows = []
    for ticker, review in reviews.items():
        if ticker not in reference.index or review is None or review.ttm.empty:
            continue
        mine = review_counts(review, options)
        for metric in METRIC_COLUMNS:
            row = {"Ticker": ticker, "metric": metric}
            for field in FIELDS:
                col = f"{metric} TTM {field}"
                theirs = pd.to_numeric(str(reference.at[ticker, col]).replace(",", ""), errors="coerce") if col in reference.columns else np.nan
                row[f"mine {field}"] = mine[metric][field]
                row[f"theirs {field}"] = theirs
            rows.append(row)
    per = pd.DataFrame(rows)
    summary = []
    for metric in METRIC_COLUMNS:
        sub = per[per["metric"] == metric] if not per.empty else per
        if sub.empty:
            continue
        same = sub["mine #Obs"] == sub["theirs #Obs"]
        for field in FIELDS:
            a, b = sub[f"mine {field}"], sub[f"theirs {field}"]
            both = a.notna() & b.notna()
            d = (a - b)[both]
            ds = (a - b)[both & same]
            summary.append({
                "metric": metric, "field": field, "tickers": int(both.sum()),
                "exact": round(float((d == 0).mean()), 3) if len(d) else np.nan,
                "within 1": round(float((d.abs() <= 1).mean()), 3) if len(d) else np.nan,
                "same #Obs exact": round(float((ds == 0).mean()), 3) if len(ds) else np.nan,
                "rank corr": round(float(a[both].rank().corr(b[both].rank())), 3) if both.sum() > 2 else np.nan,
            })
    return pd.DataFrame(summary), per

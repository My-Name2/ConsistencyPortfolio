"""Compare observation counts with a CompanyCharts-style reference workbook (built from Zacks data).

The reference reports, per ticker and metric M in Rev, EPS, FCF, GP, CFO, EQ, the columns
"{M} TTM #Obs", "#Inc", "#Rec", "#Inc Adj", "#Rec Adj" over a trailing window, plus a series
fingerprint ("{M} TTM Slope", "R2", "CV%") that pins down the exact values it counted.

The written spec's rules are the defaults. Each reference convention is a switch in
`ReferenceOptions`; none changes the review, the workbook or the review statistics.
Each metric has its own series builder below so metric-specific conventions stay separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

METRIC_COLUMNS = {"Rev": "ttm_revenue", "EPS": "eps_ttm", "FCF": "ttm_fcf", "GP": "ttm_gross_profit", "CFO": "ttm_ocf", "EQ": "equity"}
FIELDS = ["#Obs", "#Inc", "#Rec", "#Inc Adj", "#Rec Adj"]
MONEY_METRICS = {"Rev", "FCF", "GP", "CFO", "EQ"}  # reference fingerprints are in $ millions
ALIGN_COLUMNS = {"ttm_revenue": "Revenue TTM Total", "ttm_ocf": "CFO TTM Total", "ttm_net_income": "Net Income TTM Total"}


@dataclass
class ReferenceOptions:
    window: int = 57  # trailing TTM observations the reference counts over
    equity_window: int = 57  # the reference uses 60 for equity (no TTM build needed)
    record_counts_first: bool = False  # spec: the first observation is the reference, not a record
    emulate_first_ytd: bool = False  # reference artifact: the first window value of YTD-built series (CFO, FCF) includes the prior YTD
    missing_capex_as_zero: bool = False  # reference fills FCF = CFO where capex is missing
    align_window: bool = False  # end the window at the quarter the reference's latest TTM totals match (drops our newer quarters)

    @classmethod
    def reference_compatible(cls) -> "ReferenceOptions":
        return cls(window=57, equity_window=60, record_counts_first=True, emulate_first_ytd=True, missing_capex_as_zero=True, align_window=True)


def _num(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


# ---------------------------------------------------------------- counting

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


def fingerprint(values: list, window: int, metric: str) -> dict:
    """Slope (per observation), R2 and CV% (population sd) of the window's values, as the reference computes them."""
    vals = [float(v) for v in values[-window:] if _num(v)]
    if metric in MONEY_METRICS:
        vals = [v / 1e6 for v in vals]
    if len(vals) < 3:
        return {"slope": np.nan, "r2": np.nan, "cv": np.nan, "last": vals[-1] if vals else np.nan}
    y = np.asarray(vals)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - (intercept + slope * x)) ** 2).sum() / ss_tot if ss_tot else np.nan
    cv = y.std() / y.mean() * 100 if y.mean() else np.nan
    return {"slope": float(slope), "r2": float(r2), "cv": float(cv), "last": float(y[-1])}


# ---------------------------------------------------------------- alignment

def alignment_offset(review, reference_row: pd.Series | None) -> int:
    """How many of our latest TTM rows are newer than the reference's latest quarter (0 or more).

    Uses the reference's latest totals (revenue, CFO, net income, $ millions): if they match one of
    our earlier rows on at least two metrics (within 0.5%) and do not match our latest row, the
    window ends at that earlier row.
    """
    if reference_row is None or review.ttm.empty:
        return 0
    ttm = review.ttm.reset_index(drop=True)

    def matches(i):
        hits = 0
        for col, ref_col in ALIGN_COLUMNS.items():
            theirs = pd.to_numeric(str(reference_row.get(ref_col, "")).replace(",", ""), errors="coerce")
            ours = ttm.at[i, col] if col in ttm else None
            if _num(theirs) and _num(ours) and theirs != 0 and abs(float(ours) / 1e6 - theirs) <= 0.005 * abs(theirs):
                hits += 1
        return hits

    n = len(ttm)
    if matches(n - 1) >= 2:
        return 0
    for back in (1, 2):
        if n - 1 - back >= 0 and matches(n - 1 - back) >= 2:
            return back
    return 0


# ---------------------------------------------------------------- per-metric series builders

def _ttm(review) -> pd.DataFrame:
    return review.ttm.reset_index(drop=True)


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


def series_rev(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    return list(pd.to_numeric(ttm["ttm_revenue"], errors="coerce"))


def series_gp(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    return list(pd.to_numeric(ttm["ttm_gross_profit"], errors="coerce"))


def series_cfo(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    values = list(pd.to_numeric(ttm["ttm_ocf"], errors="coerce"))
    if options.emulate_first_ytd:
        i, extra = _first_window_ytd(review, "ocf", ttm, options.window)
        if i is not None and extra is not None:
            values[i] = values[i] + extra
    return values


def series_fcf(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    ocf = pd.to_numeric(ttm["ttm_ocf"], errors="coerce")
    capex = pd.to_numeric(ttm["ttm_capex"], errors="coerce")
    if options.missing_capex_as_zero:
        capex = capex.where(capex.notna() | ocf.isna(), 0.0)
    values = list(ocf - capex)
    if options.emulate_first_ytd:
        i, extra_ocf = _first_window_ytd(review, "ocf", ttm, options.window)
        _, extra_capex = _first_window_ytd(review, "capex", ttm, options.window)
        if i is not None and extra_ocf is not None and _num(values[i]):
            values[i] = values[i] + extra_ocf - (extra_capex or 0.0)
    return values


def series_eps(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    return list(pd.to_numeric(ttm["eps_ttm"], errors="coerce"))


def series_eq(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    return list(pd.to_numeric(ttm["equity"], errors="coerce"))


SERIES_BUILDERS = {"Rev": series_rev, "EPS": series_eps, "FCF": series_fcf, "GP": series_gp, "CFO": series_cfo, "EQ": series_eq}


def series_for(review, metric: str, options: ReferenceOptions, offset: int = 0) -> list:
    ttm = _ttm(review)
    if ttm.empty:
        return []
    if offset:
        ttm = ttm.iloc[: len(ttm) - offset].reset_index(drop=True)
    return SERIES_BUILDERS[metric](review, ttm, options)


def review_counts(review, options: ReferenceOptions | None = None, reference_row: pd.Series | None = None) -> dict[str, dict]:
    options = options or ReferenceOptions()
    offset = alignment_offset(review, reference_row) if options.align_window else 0
    out = {}
    for metric in METRIC_COLUMNS:
        window = options.equity_window if metric == "EQ" else options.window
        values = series_for(review, metric, options, offset)
        counts = window_counts(values, window, options.record_counts_first)
        counts["offset"] = offset
        counts["fingerprint"] = fingerprint(values, window, metric)
        out[metric] = counts
    return out


# ---------------------------------------------------------------- comparison

def load_reference(path) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype=str)
    if "Ticker" not in frame.columns:
        raise ValueError("Reference workbook needs a 'Ticker' column.")
    return frame.set_index(frame["Ticker"].str.strip().str.upper())


def _ref_num(reference: pd.DataFrame, ticker: str, col: str):
    if col not in reference.columns:
        return np.nan
    return pd.to_numeric(str(reference.at[ticker, col]).replace(",", "").replace("%", ""), errors="coerce")


def per_ticker(reviews: dict, reference: pd.DataFrame, options: ReferenceOptions | None = None) -> pd.DataFrame:
    options = options or ReferenceOptions()
    rows = []
    for ticker, review in reviews.items():
        if ticker not in reference.index or review is None or review.ttm.empty:
            continue
        mine = review_counts(review, options, reference.loc[ticker])
        for metric in METRIC_COLUMNS:
            row = {"Ticker": ticker, "metric": metric, "offset": mine[metric]["offset"]}
            for field in FIELDS:
                row[f"mine {field}"] = mine[metric][field]
                row[f"theirs {field}"] = _ref_num(reference, ticker, f"{metric} TTM {field}")
            fp = mine[metric]["fingerprint"]
            row.update({"mine slope": fp["slope"], "mine r2": fp["r2"], "mine cv": fp["cv"],
                        "theirs slope": _ref_num(reference, ticker, f"{metric} TTM Slope"),
                        "theirs r2": _ref_num(reference, ticker, f"{metric} TTM R2"),
                        "theirs cv": _ref_num(reference, ticker, f"{metric} TTM CV%")})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for metric in METRIC_COLUMNS:
        sub = per[per["metric"] == metric] if not per.empty else per
        if sub.empty:
            continue
        same = sub["mine #Obs"] == sub["theirs #Obs"]
        slope_ok = ((sub["mine slope"] - sub["theirs slope"]).abs() <= 0.005 * sub["theirs slope"].abs()) & ((sub["mine r2"] - sub["theirs r2"]).abs() <= 0.002)
        for field in FIELDS:
            a, b = sub[f"mine {field}"], sub[f"theirs {field}"]
            both = a.notna() & b.notna()
            d = (a - b)[both]
            ds = (a - b)[both & same]
            summary.append({
                "metric": metric, "field": field, "tickers": int(both.sum()),
                "exact": round(float((d == 0).mean()), 3) if len(d) else np.nan,
                "within 1": round(float((d.abs() <= 1).mean()), 3) if len(d) else np.nan,
                "within 2": round(float((d.abs() <= 2).mean()), 3) if len(d) else np.nan,
                "same #Obs exact": round(float((ds == 0).mean()), 3) if len(ds) else np.nan,
                "series match": round(float(slope_ok[both].mean()), 3) if both.any() else np.nan,
            })
    return pd.DataFrame(summary)


def compare(reviews: dict, reference: pd.DataFrame, options: ReferenceOptions | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(agreement per metric and field, per-ticker mine/theirs) for reviews keyed by ticker."""
    per = per_ticker(reviews, reference, options)
    return summarize(per), per

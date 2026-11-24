"""Compare observation counts with a CompanyCharts-style reference workbook (built from Zacks data).

The reference reports, per ticker and metric M in Rev, EPS, FCF, GP, CFO, EQ, the columns
"{M} TTM #Obs", "#Inc", "#Rec", "#Inc Adj", "#Rec Adj" over a trailing window, plus a series
fingerprint ("{M} TTM Slope", "R2", "CV%") that pins down the exact values it counted.

The written spec's rules are the defaults. Each reference convention is a switch in
`ReferenceOptions`; none changes the review, the workbook or the review statistics.
Each metric has its own series builder below so metric-specific conventions stay separate.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

METRIC_COLUMNS = {"Rev": "ttm_revenue", "EPS": "eps_ttm", "FCF": "ttm_fcf", "GP": "ttm_gross_profit", "CFO": "ttm_ocf", "EQ": "equity"}
FIELDS = ["#Obs", "#Inc", "#Rec", "#Inc Adj", "#Rec Adj"]
MONEY_METRICS = {"Rev", "FCF", "GP", "CFO", "EQ"}  # reference fingerprints are in $ millions (EPS too when it counts net income)
# Comparative-only quarters (engine flag INCLUDE_COMPARATIVE_QUARTERS) the reference counts, measured on 486 tickers:
# revenue, gross profit and net income the latest two; CFO and FCF (built from YTD data) all; equity the latest eight.
COMPARATIVE_DEPTH = {"Rev": 2, "EPS": 2, "GP": 2, "FCF": None, "CFO": None, "EQ": 8}
ALIGN_COLUMNS = {"ttm_revenue": "Revenue TTM Total", "ttm_ocf": "CFO TTM Total", "ttm_net_income": "Net Income TTM Total"}


# Single-line total revenue tags some filers use for whole spans of years instead of Revenues / contract revenue
# (CMG FoodAndBeverageRevenue, O / ARE / EXR RealEstateRevenueNet, FANG OilAndGasRevenue, HCA patient service revenue).
REV_FILL_TAGS = ("RealEstateRevenueNet", "FoodAndBeverageRevenue", "OilAndGasRevenue", "HealthCareOrganizationRevenue",
                 "HealthCareOrganizationPatientServiceRevenueLessProvisionForBadDebts", "OperatingLeasesIncomeStatementLeaseRevenue")
BANK_INTEREST_INCOME_TAGS = ("InterestAndDividendIncomeOperating", "InterestIncomeOperating", "InterestAndFeeIncomeLoansAndLeases")
BANK_INTEREST_EXPENSE_TAGS = ("InterestExpense", "InterestExpenseOperating")
BANK_NET_INTEREST_TAGS = ("InterestIncomeExpenseNet",)
BANK_NONINTEREST_TAGS = ("NoninterestIncome",)


@dataclass
class ReferenceOptions:
    window: int = 57  # trailing TTM observations the reference counts over
    equity_window: int = 57  # the reference uses 60 for equity (no TTM build needed)
    record_counts_first: bool = False  # spec: the first observation is the reference, not a record
    emulate_first_ytd: bool = False  # reference artifact: the first window value of YTD-built series (CFO, FCF) includes the prior YTD
    missing_capex_as_zero: bool = False  # reference fills FCF = CFO where capex is missing
    align_window: bool = False  # end the window at the quarter the reference's latest TTM totals match (drops our newer quarters)
    # With the engine's opt-in INCLUDE_COMPARATIVE_QUARTERS rows (quarters before the first filing, reported only as
    # comparatives): keep only the latest N of them (an int, or {metric: N}); None keeps all the engine added.
    comparative_depth: int | dict | None = None
    # The reference's window is a date span, not a row count: quarters ending in the as-of month or the `window` x 3
    # months before it (June 2012 - September 2026 for 57; equity 60 from September 2011). Calendar filers get 57;
    # filers whose quarters end in Feb/May/Aug/Nov get 56 (May 2012 is before the span and their Aug 2026 quarter was
    # not reported yet; equity 59). Counted in quarter steps from our latest quarter (a 10-Q missing from Input A still
    # counts), plus the quarters the reference holds beyond ours when its latest totals match none of our last rows;
    # the window is that many of our latest rows. See dated_window().
    window_by_date: bool = False
    # DIAGNOSTIC ONLY (never in reference_compatible()): alignment by the reference's own fingerprints. Among windows
    # ending 0, 1 or 2 rows before our latest, take the one whose slope/R2 match the reference on more metrics than
    # the value-based offset does. It picks the window with the answer it is scored against, so report it separately.
    align_by_fingerprint: bool = False
    eps_counts_net_income: bool = False  # the reference's "EPS TTM" counts are computed on TTM net income, not per-share EPS
    # Equity (EQ). The reference counts the balance-sheet value at each of the last 60 fiscal quarter ends; its
    # "EQ TTM" fingerprint is the 4-quarter mean of those values over 57 observations (the same 60 quarter ends).
    equity_quarter_ends: bool = False  # EQ = equity at every fiscal quarter end (also quarters before our first TTM row), not at valuation dates
    equity_total: bool = False  # EQ includes noncontrolling interest (total equity), not parent equity only
    equity_first_reported: bool = False  # EQ uses each quarter end's value as first reported, not as later restated
    equity_fingerprint_mean4: bool = False  # compare EQ's fingerprint as the 4-quarter mean over `window` observations
    equity_history_years: int = 0  # with equity_quarter_ends: only quarter ends after the as-of month's start this many years earlier (0 = no limit)
    eps_net_income: str = "parent"  # with eps_counts_net_income: "parent" (the review's NI attributable to parent) or "common" (NI available to common stockholders)
    # Revenue (Rev) and gross profit (GP).
    rev_bank_gross: bool = False  # banks (noninterest income + interest income tagged): Rev = interest income + noninterest income (gross), not net of interest expense
    rev_fill_tags: tuple = ()  # fill blank TTM revenue from these single-line total-revenue tags (first available per period)
    gp_bank_net: bool = False  # banks: GP = revenue net of interest expense (net interest income + noninterest income)
    gp_revenue_when_gm100: bool = False  # where the reference's GM % is 100 (financials without a cost of revenue), GP = Rev

    @classmethod
    def reference_compatible(cls) -> "ReferenceOptions":
        return cls(window=57, equity_window=60, record_counts_first=True, emulate_first_ytd=True, missing_capex_as_zero=True, align_window=True,
                   eps_counts_net_income=True, eps_net_income="common", equity_quarter_ends=True, equity_total=True, equity_first_reported=True,
                   equity_fingerprint_mean4=True, equity_history_years=15, window_by_date=True, comparative_depth=dict(COMPARATIVE_DEPTH),
                   rev_bank_gross=True, rev_fill_tags=REV_FILL_TAGS, gp_bank_net=True, gp_revenue_when_gm100=True)


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


def fingerprint(values: list, window: int, metric: str, money: bool | None = None) -> dict:
    """Slope (per observation), R2 and CV% (population sd) of the window's values, as the reference computes them."""
    vals = [float(v) for v in values[-window:] if _num(v)]
    if (metric in MONEY_METRICS) if money is None else money:
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

def _totals_hits(ttm: pd.DataFrame, i: int, reference_row: pd.Series) -> int:
    """On how many of revenue, CFO and net income our row i matches the reference's latest TTM totals (within 0.5%)."""
    hits = 0
    for col, ref_col in ALIGN_COLUMNS.items():
        theirs = pd.to_numeric(str(reference_row.get(ref_col, "")).replace(",", ""), errors="coerce")
        ours = ttm.at[i, col] if col in ttm else None
        if _num(theirs) and _num(ours) and theirs != 0 and abs(float(ours) / 1e6 - theirs) <= 0.005 * abs(theirs):
            hits += 1
    return hits


def alignment_offset(review, reference_row: pd.Series | None) -> int:
    """How many of our latest TTM rows are newer than the reference's latest quarter (0 or more).

    Uses the reference's latest totals (revenue, CFO, net income, $ millions): if they match one of
    our earlier rows on at least two metrics (within 0.5%) and do not match our latest row, the
    window ends at that earlier row.
    """
    if reference_row is None or review.ttm.empty:
        return 0
    ttm = review.ttm.reset_index(drop=True)
    n = len(ttm)
    if _totals_hits(ttm, n - 1, reference_row) >= 2:
        return 0
    for back in (1, 2):
        if n - 1 - back >= 0 and _totals_hits(ttm, n - 1 - back, reference_row) >= 2:
            return back
    return 0


def reference_is_newer(review, reference_row: pd.Series | None) -> bool:
    """The reference holds a quarter we lack (a 10-Q missing from Input A, or reported after the as-of date): its
    latest totals match none of our last three rows on two metrics. (Also true where the definitions differ; callers
    cap what they do with it.)"""
    if reference_row is None or review.ttm.empty:
        return False
    ttm = review.ttm.reset_index(drop=True)
    return all(_totals_hits(ttm, i, reference_row) < 2 for i in range(max(0, len(ttm) - 3), len(ttm)))


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


def _tag_ttm(review, ttm: pd.DataFrame, tags: tuple) -> np.ndarray:
    """TTM values per row of `ttm` from a fallback chain of us-gaap duration tags (first available per period), built
    with the engine's own period logic. Cached on the review."""
    cache = review.__dict__.setdefault("_reference_tag_ttm", {})
    key = (tuple(tags), tuple(pd.to_datetime(ttm["quarter_end"])))
    if key not in cache:
        from .dq import DQLog
        from .periods import quarterly_values, ttm_flow
        from .tags import US, MetricDef
        from .values import build_metric_table

        metric = MetricDef("reference:" + "+".join(tags), "+".join(tags), "duration", "money", True, {US: tuple(tags)}, {})
        table = build_metric_table(review.cf, metric, review.basis, DQLog())
        out = np.full(len(ttm), np.nan)
        if not table.table.empty:
            cal = review.calendar
            by_end = {q.end: q for q in cal.quarters}
            qv = quarterly_values(table, cal, DQLog())
            for i, end in enumerate(pd.to_datetime(ttm["quarter_end"])):
                q = by_end.get(end)
                v = ttm_flow(table, qv, cal, q).value if q is not None else None
                out[i] = np.nan if v is None else float(v)
        cache[key] = out
    return cache[key]


def _fill(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = values.copy()
    blank = np.isnan(out)
    out[blank] = fallback[blank]
    return out


def _bank_series(review, ttm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """(gross revenue, net revenue) for a bank-like filer, or None.

    A filer is treated as a bank when it tags noninterest income and interest income over at least half of the
    rows. Gross = interest income + noninterest income (else net revenue + interest expense); net = our revenue
    (else net interest income + noninterest income)."""
    nonint = _tag_ttm(review, ttm, BANK_NONINTEREST_TAGS)
    income = _tag_ttm(review, ttm, BANK_INTEREST_INCOME_TAGS)
    if min(np.isfinite(nonint).sum(), np.isfinite(income).sum()) < max(1, len(ttm) // 2):
        return None
    ours = pd.to_numeric(ttm["ttm_revenue"], errors="coerce").to_numpy(dtype=float)
    net = _fill(ours, _tag_ttm(review, ttm, BANK_NET_INTEREST_TAGS) + nonint)
    gross = _fill(income + nonint, net + _tag_ttm(review, ttm, BANK_INTEREST_EXPENSE_TAGS))
    return gross, net


def series_rev(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    values = pd.to_numeric(ttm["ttm_revenue"], errors="coerce").to_numpy(dtype=float)
    if options.rev_fill_tags and np.isnan(values).any():
        values = _fill(values, _tag_ttm(review, ttm, tuple(options.rev_fill_tags)))
    if options.rev_bank_gross:
        bank = _bank_series(review, ttm)
        if bank is not None:
            values = bank[0]
    return list(values)


def series_gp(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    values = pd.to_numeric(ttm["ttm_gross_profit"], errors="coerce").to_numpy(dtype=float)
    gm = getattr(review, "_reference_gm", None)
    if options.gp_revenue_when_gm100 and gm is not None and _num(gm) and abs(float(gm) - 100) < 1e-6:
        return series_rev(review, ttm, options)
    if options.gp_bank_net:
        bank = _bank_series(review, ttm)
        if bank is not None:
            return list(bank[1])
    return list(values)


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


NI_COMMON_TAG = "NetIncomeLossAvailableToCommonStockholdersBasic"


def _ni_available_to_common(review, ttm: pd.DataFrame) -> list:
    """TTM net income available to common stockholders: the review's TTM NI attributable to parent less the TTM
    of (NI - NI available to common), the latter taken only from periods where both are tagged and built with the
    review's period rules (YTD differencing, FY - 9M). Where that TTM difference cannot be built, NI is used as is,
    so tagging gaps never mix the two tags inside one quarter."""
    from .dq import DQLog
    from .periods import quarterly_values, ttm_flow
    from .tags import IFRS, US, MetricDef
    from .values import MetricTable, build_metric_table

    base = list(pd.to_numeric(ttm["ttm_net_income"], errors="coerce"))
    ni = review.tables["net_income"]
    common = build_metric_table(review.cf, MetricDef("ni_common", "NI available to common", "duration", "money", True, {US: (NI_COMMON_TAG,), IFRS: ()}), review.basis, DQLog())
    if ni.table.empty or common.table.empty:
        return base
    both = ni.table.merge(common.table[["start", "end", "value"]], on=["start", "end"], suffixes=("", "_common"))
    if both.empty:
        return base
    diff = MetricTable(ni.metric, both.assign(value=both["value"].astype(float) - both["value_common"].astype(float)).drop(columns="value_common"))
    cal = review.calendar
    qv = quarterly_values(diff, cal, DQLog())
    by_end = {q.end: q for q in cal.quarters}
    out = []
    for end, value in zip(ttm["quarter_end"], base):
        q = by_end.get(pd.Timestamp(end))
        d = ttm_flow(diff, qv, cal, q).value if q is not None and _num(value) else None
        out.append(value - d if _num(value) and _num(d) else value)
    return out


def series_eps(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    # The reference's "EPS TTM" fingerprint (slope, R2, CV%) equals TTM net income's in $ millions
    # (AAPL 1520.44 / 0.8742 / 38.86%, MSFT 1872.63 / 0.8564 / 68.94%), so its counts use net income.
    if not options.eps_counts_net_income:
        return list(pd.to_numeric(ttm["eps_ttm"], errors="coerce"))
    if options.eps_net_income == "common":
        # Its net income is after preferred dividends: JPM 778.82 / 0.8662 (ours with NI to parent 780.43 / 0.8705);
        # latest "Net Income TTM Total" 63,634 = NI available to common (NI attributable to parent 65,067).
        return _ni_available_to_common(review, ttm)
    return list(pd.to_numeric(ttm["ttm_net_income"], errors="coerce"))


def quarter_end_equity(review, last_quarter_end, total: bool = False, first_reported: bool = False, since=None) -> list:
    """Equity at every fiscal quarter end after `since` up to `last_quarter_end` (None where no balance sheet is tagged at that instant).

    `total` adds noncontrolling interest back (total equity); `first_reported` takes each instant's value from the
    first filing that reported it (a 10-Q or 10-K balance sheet), as the reference does, instead of later restatements.
    """
    last = pd.Timestamp(last_quarter_end)
    after = pd.Timestamp(since) if since is not None else pd.Timestamp.min
    ends = sorted({q.end for q in review.calendar.quarters if after < q.end <= last})
    equity, nci = review.tables["equity"], review.tables.get("nci")
    column = "first_value" if first_reported else "value"
    out = []
    for end in ends:
        row = equity.get(None, end)
        if row is None or not _num(float(row[column])):
            out.append(None)
            continue
        value = float(row[column])
        n = nci.get(None, end) if total and nci is not None else None
        if n is not None:
            # Rows derived as total - NCI subtracted NCI's latest value; add that back to recover the tagged total.
            value += float(n["value"] if str(row["tag"]).startswith("derived:") else n[column])
        out.append(value)
    return out


def _equity_since(review, last, options: ReferenceOptions):
    """Earliest quarter end the reference's equity history holds (exclusive), or None.

    Its quarterly history covers quarters ending after the start of the as-of month N years earlier (as-of 2026-09-14:
    after 2011-09-01): May-quarter filers whose 60th quarter back ends 2011-08-31 (ACN, COST, MU 2011-09-01) have 59
    observations there, PEP (2011-09-03) 60. Where our latest quarter is older than a filing deadline allows (a 10-Q
    missing from the inputs: AEP, DUK, SO end 2026-03-31), the reference holds the newer quarter and its history
    starts that many quarters earlier.
    """
    if not options.equity_history_years:
        return None
    as_of = pd.Timestamp(review.inputs.as_of)
    missing = max(0, int(((as_of - last).days - 45) // 91))  # quarters the reference has after our latest
    return as_of.replace(day=1) - pd.DateOffset(years=options.equity_history_years, months=3 * missing)


def series_eq(review, ttm: pd.DataFrame, options: ReferenceOptions) -> list:
    if options.equity_quarter_ends:
        last = pd.Timestamp(ttm["quarter_end"].iloc[-1])
        return quarter_end_equity(review, last, options.equity_total, options.equity_first_reported, _equity_since(review, last, options))
    return list(pd.to_numeric(ttm["equity"], errors="coerce"))


SERIES_BUILDERS = {"Rev": series_rev, "EPS": series_eps, "FCF": series_fcf, "GP": series_gp, "CFO": series_cfo, "EQ": series_eq}


def _comparative_kept(ttm: pd.DataFrame, depth, metric: str) -> pd.DataFrame:
    """Drop the earliest comparative-only rows beyond `depth` (they lead the table)."""
    if isinstance(depth, dict):
        depth = depth.get(metric)
    if depth is None or "comparative" not in ttm:
        return ttm
    n = int(ttm["comparative"].fillna(False).astype(bool).sum())
    drop = max(0, n - int(depth))
    return ttm.iloc[drop:].reset_index(drop=True) if drop else ttm


def dated_window(review, metric: str, options: ReferenceOptions, offset: int, window: int, reference_row: pd.Series | None = None) -> int:
    """How many of our latest rows the reference's date span covers, at most `window`.

    The span holds the quarters ending in the as-of month or the 3 x `window` months before it (from June 2012 for
    57 quarters as of September 2026, September 2011 for 60). It is counted in quarter steps back from our (aligned)
    latest quarter end, so a quarter missing from our rows still counts. When the reference is newer, the quarters it
    holds beyond ours count too: those whose 10-Q was due (45 days) by the as-of date, at least one."""
    ttm = _comparative_kept(_ttm(review), options.comparative_depth, metric)
    if len(ttm) <= offset:
        return 0
    latest = pd.Timestamp(ttm["quarter_end"].iloc[len(ttm) - 1 - offset])
    as_of = pd.Timestamp(review.inputs.as_of)
    first_month = as_of.year * 12 + as_of.month - 3 * window
    slots = max(0, (latest.year * 12 + latest.month - first_month) // 3 + 1)
    if offset == 0 and reference_is_newer(review, reference_row):
        due = sum(1 for j in range(1, 5) if latest + pd.DateOffset(months=3 * j) + pd.Timedelta(days=45) <= as_of)
        slots += max(1, due)
    return min(window, slots)


def series_for(review, metric: str, options: ReferenceOptions, offset: int = 0, window: int | None = None) -> list:
    """The metric's TTM series (oldest first), ending `offset` rows before our latest; `window` overrides the
    options' window for builders that depend on where the window starts (CFO/FCF first-YTD emulation)."""
    if window is not None and metric != "EQ" and window != options.window:
        options = dataclasses.replace(options, window=window)
    ttm = _comparative_kept(_ttm(review), options.comparative_depth, metric)
    if ttm.empty:
        return []
    if offset:
        ttm = ttm.iloc[: len(ttm) - offset].reset_index(drop=True)
    return SERIES_BUILDERS[metric](review, ttm, options)


def fingerprint_matches(counts: dict[str, dict], reference_row: pd.Series | None) -> int:
    """Metrics whose window fingerprint matches the reference's (slope within 0.5%, R2 within 0.002)."""
    if reference_row is None:
        return 0
    hits = 0
    for metric, c in counts.items():
        fp = c["fingerprint"]
        slope = pd.to_numeric(str(reference_row.get(f"{metric} TTM Slope", "")).replace(",", ""), errors="coerce")
        r2 = pd.to_numeric(str(reference_row.get(f"{metric} TTM R2", "")).replace(",", ""), errors="coerce")
        if _num(slope) and _num(r2) and _num(fp["slope"]) and _num(fp["r2"]):
            hits += abs(fp["slope"] - slope) <= 0.005 * abs(slope) and abs(fp["r2"] - r2) <= 0.002
    return hits


def review_counts(review, options: ReferenceOptions | None = None, reference_row: pd.Series | None = None) -> dict[str, dict]:
    options = options or ReferenceOptions()
    offset = alignment_offset(review, reference_row) if options.align_window else 0
    if reference_row is not None:  # the reference's latest gross margin, read by series_gp (gp_revenue_when_gm100)
        review.__dict__["_reference_gm"] = pd.to_numeric(str(reference_row.get("GM %", "")).replace(",", "").replace("%", ""), errors="coerce")
    out = _counts_at(review, options, offset, reference_row)
    if options.align_by_fingerprint and reference_row is not None:
        best = fingerprint_matches(out, reference_row)
        for other in (0, 1, 2):
            if other == offset or len(review.ttm) - other < 3:
                continue
            alt = _counts_at(review, options, other, reference_row)
            hits = fingerprint_matches(alt, reference_row)
            if hits > best:
                best, out = hits, alt
    return out


def _counts_at(review, options: ReferenceOptions, offset: int, reference_row: pd.Series | None) -> dict[str, dict]:
    out = {}
    for metric in METRIC_COLUMNS:
        window = options.equity_window if metric == "EQ" else options.window
        if options.window_by_date:
            window = dated_window(review, metric, options, offset, window, reference_row)
        values = series_for(review, metric, options, offset, window=window) if window else []
        counts = window_counts(values, window, options.record_counts_first)
        counts["offset"] = offset
        money = metric in MONEY_METRICS or (metric == "EPS" and options.eps_counts_net_income)
        if metric == "EQ" and options.equity_fingerprint_mean4:
            counts["fingerprint"] = fingerprint(list(pd.Series(values, dtype=float).rolling(4).mean()), options.window, metric, money=money)
        else:
            counts["fingerprint"] = fingerprint(values, window, metric, money=money)
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

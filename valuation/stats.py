"""RAW and ADJUSTED trend statistics over quarter-end valuation dates.

The workbook reproduces every number here with live formulas; tests check the
two agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SeriesDef:
    key: str  # column in the TTM table
    label: str
    basis: str  # "TTM" or "Per share"
    family: str  # groups the TTM and per-share versions of one metric
    focus: bool
    daily_ratio: str | None  # daily ratio whose valid-day count is reported
    unit: str  # "money", "per_share", "shares"


SERIES: list[SeriesDef] = [
    SeriesDef("ttm_revenue", "Revenue", "TTM", "revenue", True, "ps", "money"),
    SeriesDef("ttm_gross_profit", "Gross profit", "TTM", "gross_profit", True, "pgp", "money"),
    SeriesDef("ttm_net_income", "Earnings (NI attributable to parent)", "TTM", "earnings", True, "pe", "money"),
    SeriesDef("equity", "Parent equity", "TTM", "equity", True, "pb", "money"),
    SeriesDef("ttm_ocf", "Operating cash flow", "TTM", "ocf", True, "pcfo", "money"),
    SeriesDef("ttm_fcf", "Free cash flow", "TTM", "fcf", True, "pfcf", "money"),
    SeriesDef("revenue_ps", "Revenue per share", "Per share", "revenue", True, "ps", "per_share"),
    SeriesDef("gp_ps", "Gross profit per share", "Per share", "gross_profit", True, "pgp", "per_share"),
    SeriesDef("eps_ttm", "Diluted EPS", "Per share", "earnings", True, "pe", "per_share"),
    SeriesDef("bvps", "Book value per share", "Per share", "equity", True, "pb", "per_share"),
    SeriesDef("ocf_ps", "Operating cash flow per share", "Per share", "ocf", True, "pcfo", "per_share"),
    SeriesDef("fcf_ps", "Free cash flow per share", "Per share", "fcf", True, "pfcf", "per_share"),
    SeriesDef("ttm_da", "D&A", "TTM", "da", False, None, "money"),
    SeriesDef("ttm_ebitda", "EBITDA", "TTM", "ebitda", False, "ev_ebitda", "money"),
    SeriesDef("ttm_operating_income", "Operating income", "TTM", "operating_income", False, "ev_ebit", "money"),
    SeriesDef("ttm_dividends_paid", "Dividends paid", "TTM", "dividends", False, "div_yield", "money"),
    SeriesDef("da_ps", "D&A per share", "Per share", "da", False, None, "per_share"),
    SeriesDef("ebitda_ps", "EBITDA per share", "Per share", "ebitda", False, "ev_ebitda", "per_share"),
    SeriesDef("oi_ps", "Operating income per share", "Per share", "operating_income", False, "ev_ebit", "per_share"),
    SeriesDef("dps_ttm", "Dividends per share", "Per share", "dividends", False, "div_yield", "per_share"),
    SeriesDef("ttm_capex", "Capital expenditures", "TTM", "capex", False, None, "money"),
    SeriesDef("avg_diluted_shares", "Average diluted shares", "TTM", "shares", False, None, "shares"),
]
SERIES_BY_KEY = {s.key: s for s in SERIES}
FAMILIES = list(dict.fromkeys(s.family for s in SERIES))


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def compute_stats(dates: list, values: list) -> dict:
    """Statistics for one series. `values[i]` is a number or None for each valuation date."""
    vals = [float(v) if _is_num(v) else None for v in values]
    present = [i for i, v in enumerate(vals) if v is not None]
    out: dict = {"start": dates[0] if dates else None}
    pairs = [(vals[i], vals[i + 1]) for i in range(len(vals) - 1) if vals[i] is not None and vals[i + 1] is not None]

    # RAW
    out["raw_n"] = len(present)
    out["raw_comparisons"] = len(pairs)
    out["raw_increases"] = sum(1 for a, b in pairs if b > a)
    out["raw_decreases"] = sum(1 for a, b in pairs if b < a)
    out["raw_unchanged"] = sum(1 for a, b in pairs if b == a)
    out["raw_increase_rate"] = out["raw_increases"] / len(pairs) if pairs else None
    records, running = 0, None
    for i in present:
        v = vals[i]
        if running is not None and v > running:
            records += 1
        running = v if running is None else max(running, v)
    out["raw_record_highs"] = records

    latest_i = present[-1] if present else None
    latest = vals[latest_i] if latest_i is not None else None
    earlier = [vals[i] for i in present[:-1]]
    if latest is None:
        out["raw_latest_record"] = "n/a (no values)"
    elif not earlier:
        out["raw_latest_record"] = "n/a (single value)"
    else:
        out["raw_latest_record"] = "yes" if latest > max(earlier) else "no"
    out["raw_latest_rank"] = (1 + sum(1 for i in present if vals[i] > latest)) if latest is not None else None

    # ADJUSTED
    eligible = [i for i in present if vals[i] > 0]
    adj_pairs = [(a, b) for a, b in pairs if a > 0 and b > 0]
    out["adj_n"] = len(eligible)
    out["adj_comparisons"] = len(adj_pairs)
    out["adj_pairs_skipped"] = len(pairs) - len(adj_pairs)
    out["adj_increases"] = sum(1 for a, b in adj_pairs if b > a)
    out["adj_decreases"] = sum(1 for a, b in adj_pairs if b < a)
    out["adj_unchanged"] = sum(1 for a, b in adj_pairs if b == a)
    out["adj_increase_rate"] = out["adj_increases"] / len(adj_pairs) if adj_pairs else None
    records, running = 0, None
    for i in eligible:
        v = vals[i]
        if running is not None and v > running:
            records += 1
        running = v if running is None else max(running, v)
    out["adj_record_highs"] = records
    if latest is None:
        out["adj_latest_record"] = "n/a (no values)"
    elif latest <= 0:
        out["adj_latest_record"] = "excluded (≤ 0)"
    else:
        earlier_eligible = [vals[i] for i in eligible if i != latest_i]
        out["adj_latest_record"] = "yes" if earlier_eligible and latest > max(earlier_eligible) else ("n/a (single value)" if not earlier_eligible else "no")
    out["adj_excluded"] = sum(1 for i in present if vals[i] <= 0)
    out["adj_sign_changes"] = sum(1 for a, b in pairs if (a > 0 and b < 0) or (a < 0 and b > 0))

    # COMMON
    out["latest"] = latest
    out["latest_date"] = dates[latest_i] if latest_i is not None else None
    prior_i = present[-2] if len(present) >= 2 else None
    out["prior"] = vals[prior_i] if prior_i is not None else None
    out["prior_date"] = dates[prior_i] if prior_i is not None else None
    if out["prior"] is None or latest is None:
        out["pct_change"] = None
    elif out["prior"] == 0:
        out["pct_change"] = "NM"
    else:
        out["pct_change"] = (latest - out["prior"]) / abs(out["prior"])
    if present:
        min_i = min(present, key=lambda i: (vals[i], i))
        max_i = min(present, key=lambda i: (-vals[i], i))
        out.update(min=vals[min_i], min_date=dates[min_i], max=vals[max_i], max_date=dates[max_i])
    else:
        out.update(min=None, min_date=None, max=None, max_date=None)
    return out


def stats_table(ttm: pd.DataFrame, start_index: int, daily_counts: dict[str, int] | None = None) -> pd.DataFrame:
    rows = []
    sub = ttm.iloc[start_index:]
    dates = list(sub["valuation_date"])
    for s in SERIES:
        values = list(sub[s.key]) if s.key in sub else [None] * len(sub)
        stats = compute_stats(dates, values)
        stats.update(series=s.key, label=s.label, basis=s.basis, focus=s.focus, family=s.family)
        stats["daily_obs"] = (daily_counts or {}).get(s.daily_ratio) if s.daily_ratio else None
        stats["daily_ratio"] = s.daily_ratio
        rows.append(stats)
    return pd.DataFrame(rows)

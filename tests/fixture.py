"""SYNTHETIC test fixture: a made-up company, not real data.

Calendar fiscal years 2019-2024 plus 2025 Q1-Q3. Quarterly values follow
simple formulas so every TTM, derived quarter and ratio can be checked by
hand. A 2-for-1 split takes effect on 2022-06-15: filings made after it
restate share counts and EPS for every period they report. Input C prices
are NOT split-adjusted, so the close halves on the split date.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

SPLIT_DATE = pd.Timestamp("2022-06-15")
FY_YEARS = range(2019, 2025)
CURRENT_YEAR = 2025
AS_OF = date(2025, 11, 21)
PRE_SPLIT_SHARES = 1_000_000.0


def q_index(year: int, k: int) -> int:
    return (year - 2019) * 4 + (k - 1)


def revenue(year, k):
    return 1_000_000.0 + 10_000.0 * q_index(year, k)


def net_income(year, k):
    return -900_000.0 if (year, k) == (2020, 2) else 0.10 * revenue(year, k)


def ocf(year, k):
    return net_income(year, k) + 20_000.0


CAPEX, DA = 15_000.0, 5_000.0


def operating_income(year, k):
    return net_income(year, k) + 10_000.0


def shares_true(year, k):
    """Weighted diluted shares on the post-split basis."""
    return 2 * PRE_SPLIT_SHARES


def quarter_bounds(year, k):
    start = date(year, 3 * (k - 1) + 1, 1)
    end = (date(year, 3 * k + 1, 1) if k < 4 else date(year + 1, 1, 1)) - timedelta(days=1)
    return start, end


def filed_date(year, k):
    _, end = quarter_bounds(year, k)
    return end + timedelta(days=60 if k == 4 else 40)


def _fact(store, tag, unit, start, end, val, accn, form, filed, fy, fp):
    item = {"end": end.isoformat(), "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed.isoformat()}
    if start is not None:
        item["start"] = start.isoformat()
    store.setdefault(tag, {"label": tag, "units": {}})["units"].setdefault(unit, []).append(item)


def companyfacts() -> dict:
    gaap: dict = {}
    dei: dict = {}
    n = 0

    def basis(filed: date) -> float:
        return 1.0 if pd.Timestamp(filed) >= SPLIT_DATE else 0.5  # pre-split filings report half the shares

    periods = [(y, k) for y in FY_YEARS for k in (1, 2, 3, 4)] + [(CURRENT_YEAR, k) for k in (1, 2, 3)]
    for year, k in periods:
        n += 1
        filed = filed_date(year, k)
        accn = f"0000000001-{str(year)[2:]}-{n:06d}"
        form = "10-K" if k == 4 else "10-Q"
        b = basis(filed)
        years = [year, year - 1] if k < 4 else [year, year - 1, year - 2]
        for y in years:
            if y < 2018 + 1 and k < 4:
                continue
            if y < 2017:
                continue
            fy_start = date(y, 1, 1)
            _, q_end = quarter_bounds(y, k)
            q_start, _ = quarter_bounds(y, k)
            have = lambda yy: yy >= 2019  # noqa: E731  (history begins with fiscal 2019 values)
            if not have(y):
                continue
            # income statement: 3M (10-Q only) and YTD / FY
            ytd_k = range(1, k + 1)
            ytd = {
                "Revenues": sum(revenue(y, j) for j in ytd_k),
                "NetIncomeLoss": sum(net_income(y, j) for j in ytd_k),
                "OperatingIncomeLoss": sum(operating_income(y, j) for j in ytd_k),
                "NetCashProvidedByUsedInOperatingActivities": sum(ocf(y, j) for j in ytd_k),
                "PaymentsToAcquirePropertyPlantAndEquipment": CAPEX * k,
                "DepreciationDepletionAndAmortization": DA * k,
            }
            for tag, val in ytd.items():
                _fact(gaap, tag, "USD", fy_start, q_end, val, accn, form, filed, year, "FY" if k == 4 else f"Q{k}")
            shares_rep = shares_true(y, k) * b
            ytd_shares = shares_true(y, k) * b
            _fact(gaap, "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", fy_start, q_end, ytd_shares, accn, form, filed, year, f"Q{k}")
            _fact(gaap, "EarningsPerShareDiluted", "USD/shares", fy_start, q_end, round(ytd["NetIncomeLoss"] / ytd_shares, 2), accn, form, filed, year, f"Q{k}")
            if k < 4 and k > 1:
                for tag, fn in (("Revenues", revenue), ("NetIncomeLoss", net_income), ("OperatingIncomeLoss", operating_income)):
                    _fact(gaap, tag, "USD", q_start, q_end, fn(y, k), accn, form, filed, year, f"Q{k}")
                _fact(gaap, "WeightedAverageNumberOfDilutedSharesOutstanding", "shares", q_start, q_end, shares_rep, accn, form, filed, year, f"Q{k}")
            # balance sheet at the period end
            q_index_now = q_index(y, k)
            bs = {
                "CashAndCashEquivalentsAtCarryingValue": 500_000.0 + 1_000.0 * q_index_now,
                "Assets": 5_000_000.0 + 10_000.0 * q_index_now,
                "Liabilities": 2_000_000.0,
                "StockholdersEquity": 3_000_000.0 + 10_000.0 * q_index_now,
                "LongTermDebtNoncurrent": 1_000_000.0,
                "Goodwill": 200_000.0,
            }
            if y == year:
                for tag, val in bs.items():
                    _fact(gaap, tag, "USD", None, q_end, val, accn, form, filed, year, f"Q{k}")
        _fact(dei, "EntityCommonStockSharesOutstanding", "shares", None, filed - timedelta(days=5), 2 * PRE_SPLIT_SHARES * b, accn, form, filed, year, f"Q{k}")
    return {"cik": 1, "entityName": "SYNTHETIC TEST CO (not a real company)", "facts": {"us-gaap": gaap, "dei": dei}}


def prices() -> pd.DataFrame:
    days = pd.bdate_range("2018-06-01", "2025-11-21")
    adjusted = 10.0 + 0.01 * np.arange(len(days))  # as-of (post-split) basis
    raw = np.where(days < SPLIT_DATE, adjusted * 2, adjusted)
    frame = pd.DataFrame({"date": days, "close": raw, "dividend": 0.0})
    for d in pd.date_range("2018-06-15", "2025-11-21", freq="QS-FEB") + pd.Timedelta(days=14):
        hit = frame.index[frame["date"] >= d]
        if len(hit):
            i = hit[0]
            frame.loc[i, "dividend"] = 0.20 if frame.loc[i, "date"] < SPLIT_DATE else 0.10
    return frame

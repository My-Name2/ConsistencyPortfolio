"""Input C: daily closes, dividends with ex-dates, and the stated adjustment basis."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .dq import DQLog
from .splits import SplitBasis

STALE_TRADING_DAYS = 5


# ------------------------------------------------------------------ exchange calendar
# Used only to tell a data gap from an exchange holiday. It follows NYSE rules;
# a listing on another exchange can produce false gap entries, which are logged
# as "possible" gaps for that reason.

def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + (month == 12), month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _observed(d: date, saturday_to_friday: bool = True) -> date | None:
    if d.weekday() == 5:
        return d - timedelta(days=1) if saturday_to_friday else None
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


SPECIAL_CLOSURES = {
    date(1994, 4, 27), date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11), date(2007, 1, 2), date(2012, 10, 29), date(2012, 10, 30),
    date(2018, 12, 5), date(2025, 1, 9),
}


@lru_cache(maxsize=None)
def exchange_holidays(year: int) -> frozenset[date]:
    days = {
        _observed(date(year, 1, 1), saturday_to_friday=False),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        days.add(_observed(date(year, 6, 19)))
    days |= {d for d in SPECIAL_CLOSURES if d.year == year}
    days.discard(None)
    return frozenset(days)


def expected_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    if start > end:
        return pd.DatetimeIndex([])
    days = pd.bdate_range(start, end)
    holidays = set()
    for year in range(start.year, end.year + 1):
        holidays |= exchange_holidays(year)
    return days[[d.date() not in holidays for d in days]]


# ------------------------------------------------------------------ loading

@dataclass
class PriceMeta:
    ticker: str = ""
    split_adjusted: bool | None = None
    dividend_adjusted: bool | None = None
    currency: str = ""

    @classmethod
    def from_json(cls, source) -> "PriceMeta":
        if source is None:
            return cls()
        if isinstance(source, dict):
            data = source
        elif isinstance(source, (str, Path)):
            data = json.loads(Path(source).read_text(encoding="utf-8"))
        else:
            raw = source.read()
            data = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        return cls(
            ticker=str(data.get("ticker", "") or ""),
            split_adjusted=_as_bool(data.get("split_adjusted")),
            dividend_adjusted=_as_bool(data.get("dividend_adjusted")),
            currency=str(data.get("currency", "") or "").upper(),
        )


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return None


@dataclass
class PriceData:
    meta: PriceMeta
    close_raw: pd.Series  # valid closes as supplied, indexed by date
    invalid_days: list[pd.Timestamp]
    dividends_raw: pd.DataFrame  # ex_date, amount as supplied
    splits: pd.DataFrame  # date, ratio (optional, from Input C)
    close: pd.Series = field(default=None)  # current split basis
    dividends: pd.DataFrame = field(default=None)  # current split basis
    uncertain_days: pd.Series = field(default=None)  # True where the split basis is unknown
    missing_days: list[pd.Timestamp] = field(default_factory=list)

    @property
    def first_date(self) -> pd.Timestamp:
        return self.close_raw.index.min()

    @property
    def last_date(self) -> pd.Timestamp:
        return self.close_raw.index.max()


def _read_table(source) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    raw = source.getvalue() if hasattr(source, "getvalue") else source.read()
    return pd.read_csv(io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw))


def _col(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in frame.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def load_prices(prices_source, meta: PriceMeta, dividends_source=None, dq: DQLog | None = None) -> PriceData:
    frame = _read_table(prices_source)
    date_col = _col(frame, "date")
    close_col = _col(frame, "close", "closing_price", "price")
    if date_col is None or close_col is None:
        raise ValueError("Input C must have 'date' and 'close' columns.")
    frame["_date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    frame["_close"] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna(subset=["_date"]).sort_values("_date").drop_duplicates("_date", keep="last")
    valid = frame["_close"].notna() & (frame["_close"] > 0)
    event_cols = [c for c in (_col(frame, "dividend", "dividends", "dividend_per_share", "dps"), _col(frame, "split_ratio", "split", "stock_splits", "splits")) if c is not None]
    has_event = frame[event_cols].apply(pd.to_numeric, errors="coerce").fillna(0).ne(0).any(axis=1) if event_cols else pd.Series(False, index=frame.index)
    # A row that only carries a dividend or split event is not a missing close.
    invalid_days = list(frame.loc[~valid & ~(frame["_close"].isna() & has_event), "_date"])
    close_raw = frame.loc[valid].set_index("_date")["_close"].astype(float)
    close_raw.index.name = "date"

    div_col = _col(frame, "dividend", "dividends", "dividend_per_share", "dps")
    dividends = pd.DataFrame(columns=["ex_date", "amount"])
    if div_col is not None:
        d = frame[["_date", div_col]].copy()
        d["amount"] = pd.to_numeric(d[div_col], errors="coerce")
        d = d[d["amount"].notna() & (d["amount"] != 0)]
        dividends = pd.DataFrame({"ex_date": d["_date"].to_numpy(), "amount": d["amount"].to_numpy()})
    extra = _read_table(dividends_source)
    if not extra.empty:
        ex_col = _col(extra, "ex_date", "exdate", "ex-date", "date")
        amt_col = _col(extra, "amount", "dividend", "dividends", "dps", "value")
        if ex_col is None or amt_col is None:
            raise ValueError("Dividend table must have 'ex_date' and 'amount' columns.")
        more = pd.DataFrame(
            {"ex_date": pd.to_datetime(extra[ex_col], errors="coerce").dt.normalize(), "amount": pd.to_numeric(extra[amt_col], errors="coerce")}
        ).dropna()
        dividends = pd.concat([dividends, more], ignore_index=True).drop_duplicates(["ex_date", "amount"])
    dividends = dividends.sort_values("ex_date").reset_index(drop=True)

    split_col = _col(frame, "split_ratio", "split", "stock_splits", "splits")
    splits = pd.DataFrame(columns=["date", "ratio"])
    if split_col is not None:
        s = frame[["_date", split_col]].copy()
        s["ratio"] = pd.to_numeric(s[split_col], errors="coerce")
        s = s[s["ratio"].notna() & (s["ratio"] > 0) & (s["ratio"] != 1)]
        splits = pd.DataFrame({"date": s["_date"].to_numpy(), "ratio": s["ratio"].to_numpy()})

    # Rows without a valid close are logged by apply_basis, limited to the span the review uses.
    return PriceData(meta, close_raw, invalid_days, dividends, splits)


def apply_basis(prices: PriceData, basis: SplitBasis, dq: DQLog, check_from: pd.Timestamp | None = None) -> None:
    """Put closes and dividends on the as-of split basis; find data gaps."""
    close = prices.close_raw.copy()
    dividends = prices.dividends_raw.copy()
    gap_start = max(prices.first_date, check_from) if check_from is not None else prices.first_date
    uncertain = pd.Series(False, index=close.index)
    if prices.meta.split_adjusted is False and basis.events:
        factors = pd.Series([basis.price_factor(d) for d in close.index], index=close.index)
        close = close / factors
        if not dividends.empty:
            dividends["amount"] = dividends["amount"] / dividends["ex_date"].map(basis.price_factor)
        uncertain = pd.Series([basis.uncertain(d) is not None for d in close.index], index=close.index)
        for event in basis.events:
            if not event.price_located and event.window_start is not None:
                dq.add(
                    "price_basis",
                    f"split {event.label} ex-date not found",
                    f"No close-to-close move near 1/{event.factor:g} between {event.window_start.date()} and {event.window_end.date()} in Input C. "
                    "Price ratios inside that window are left blank because the price basis is uncertain.",
                    period=event.window_end,
                    severity="warning",
                )
        dq.add("price_basis", "split adjustment applied", f"Input C stated not split-adjusted; closes before each ex-date divided by the cumulative factor ({basis.describe()}).")
    elif prices.meta.split_adjusted is True:
        dq.add("price_basis", "split basis", "Input C stated split-adjusted; closes used as supplied and assumed adjusted through the as-of date. Dividends assumed on the same basis.")
        _check_unexplained_jumps(close, dq, stated_adjusted=True)
    if prices.meta.split_adjusted is False:
        _check_unexplained_jumps(close, dq, stated_adjusted=False)
    if prices.meta.dividend_adjusted:
        dq.add(
            "price_basis",
            "dividend-adjusted prices",
            "Input C stated dividend-adjusted: historical closes are reduced for later dividends, so historical price ratios and yields "
            "are not on traded prices. Not reversed (no method supplied).",
            severity="warning",
        )
    for day in prices.invalid_days:
        if day >= gap_start:
            dq.add("price_gap", "missing close", f"Input C row for {day.date()} has no valid close; day skipped (no carry-forward).", period=day, severity="warning")
    prices.close = close
    prices.dividends = dividends
    prices.uncertain_days = uncertain

    expected = expected_trading_days(gap_start, prices.last_date)
    present = set(prices.close_raw.index)
    invalid = set(prices.invalid_days)
    missing = [d for d in expected if d not in present and d not in invalid]
    prices.missing_days = missing + sorted(d for d in invalid if d >= gap_start)
    for run in _runs(missing):
        dq.add(
            "price_gap",
            "possible missing trading day(s)",
            f"No Input C row for {run[0].date()}" + (f" to {run[-1].date()} ({len(run)} days)" if len(run) > 1 else "")
            + " (exchange calendar assumption: NYSE). Days skipped; no price carried across.",
            period=run[0],
            severity="warning",
        )


def _check_unexplained_jumps(close: pd.Series, dq: DQLog, stated_adjusted: bool) -> None:
    ratio = (close / close.shift(1)).dropna()
    big = ratio[(ratio < 1 / 1.8) | (ratio > 1.8)]
    for day, r in big.items():
        dq.add(
            "price_basis",
            "large one-day price move",
            f"Close changed by a factor of {r:.3f} on {day.date()}"
            + (" although Input C is stated split-adjusted" if stated_adjusted else " after split adjustment")
            + "; possible unrecorded split or data error. Not adjusted.",
            period=day,
            severity="warning",
        )


def _runs(days: list[pd.Timestamp]) -> list[list[pd.Timestamp]]:
    runs: list[list[pd.Timestamp]] = []
    for d in days:
        if runs and (d - runs[-1][-1]).days <= 4:
            runs[-1].append(d)
        else:
            runs.append([d])
    return runs


# ------------------------------------------------------------------ lookups

@dataclass
class PriceLookup:
    date: pd.Timestamp | None
    close: float | None
    raw_close: float | None
    factor: float
    stale_days: int
    reason: str


def price_on_or_before(prices: PriceData, when: pd.Timestamp, allow_stale: bool) -> PriceLookup:
    """Close for a valuation date (allow_stale=False) or the as-of snapshot (allow_stale=True)."""
    if prices is None or prices.close is None or prices.close.empty:
        return PriceLookup(None, None, None, 1.0, 0, "no price data")
    if when < prices.first_date:
        return PriceLookup(None, None, None, 1.0, 0, f"before price coverage (starts {prices.first_date.date()})")
    sub = prices.close[prices.close.index <= when]
    day = sub.index.max()
    gap = expected_trading_days(day + pd.Timedelta(days=1), when)
    stale = len(gap)
    raw = float(prices.close_raw.loc[day])
    adj = float(prices.close.loc[day])
    factor = raw / adj if adj else 1.0
    if bool(prices.uncertain_days.get(day, False)):
        return PriceLookup(day, None, raw, factor, stale, "split basis uncertain on this date")
    if allow_stale:
        reason = f"stale: {stale} trading days old" if stale > STALE_TRADING_DAYS else ""
        return PriceLookup(day, adj, raw, factor, stale, reason)
    if stale > 0:
        where = "after price coverage ends" if when > prices.last_date else "no close on an expected trading day"
        return PriceLookup(None, None, None, 1.0, stale, f"{where} ({gap[0].date()}); price not carried forward")
    return PriceLookup(day, adj, raw, factor, 0, "")


def ttm_dividends(prices: PriceData | None, end: pd.Timestamp) -> tuple[float | None, int, str]:
    """Sum of dividends with ex-dates in (end - 1 year, end]."""
    if prices is None or prices.close is None:
        return None, 0, "no price data"
    start = end - pd.DateOffset(years=1)
    if start < prices.first_date - pd.Timedelta(days=7):
        return None, 0, f"dividend history in Input C starts {prices.first_date.date()}; trailing year not covered"
    if end > prices.last_date + pd.Timedelta(days=7):
        return None, 0, f"dividend history in Input C ends {prices.last_date.date()}"
    d = prices.dividends
    mask = (d["ex_date"] > start) & (d["ex_date"] <= end)
    return float(d.loc[mask, "amount"].sum()), int(mask.sum()), ""

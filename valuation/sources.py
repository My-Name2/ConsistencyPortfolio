"""Fetch Inputs A and C for a ticker into memory.

Input A: SEC companyfacts JSON (data.sec.gov). Input C: Yahoo Finance chart
API daily history. Its `close` is split-adjusted and not dividend-adjusted;
dividends are cash amounts on their ex-dates (on the same split basis) and
split events carry their ex-dates. Nothing is written to disk here.

SEC asks automated clients to identify themselves. Set SEC_USER_AGENT to your
own "Name contact@domain" string; the default is a generic client name.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import pandas as pd
import requests

from . import filings
from .prices import PriceMeta

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DEFAULT_SEC_AGENT = "ValuationReview research-client admin@example.com"


def sec_headers() -> dict:
    return {"User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_AGENT), "Accept-Encoding": "gzip, deflate"}


def _get(url: str, headers: dict, params: dict | None = None, attempts: int = 4, timeout: int = 60, retry_not_found: bool = False) -> requests.Response:
    delay = 1.0
    last_error = None
    for _ in range(attempts):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            if response.status_code == 404 and not retry_not_found:
                response.raise_for_status()
            last_error = RuntimeError(f"HTTP {response.status_code} from {url}")
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:  # network hiccup: retry with backoff
            last_error = exc
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


# ------------------------------------------------------------------ ticker lists

def default_tickers() -> list[str]:
    text = resources.files("valuation").joinpath("default_tickers.txt").read_text(encoding="utf-8")
    return parse_ticker_list(text)


def parse_ticker_list(text: str) -> list[str]:
    """Accept one-per-line, comma-separated or quoted lists (e.g. 'AAPL','MSFT')."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for part in line.replace(";", ",").replace("\t", ",").split(","):
            for token in part.split():
                ticker = token.strip().strip("'\"[]()").upper()
                if ticker and ticker not in out:
                    out.append(ticker)
    return out


def read_ticker_file(path: str | Path) -> list[str]:
    return parse_ticker_list(Path(path).read_text(encoding="utf-8"))


_TICKER_MAP: dict[str, dict] | None = None


def ticker_map() -> dict[str, dict]:
    """Ticker -> {cik, title} from SEC's company_tickers.json (cached in memory)."""
    global _TICKER_MAP
    if _TICKER_MAP is None:
        data = _get(SEC_TICKERS_URL, sec_headers()).json()
        _TICKER_MAP = {}
        for item in data.values():
            _TICKER_MAP[str(item["ticker"]).upper()] = {"cik": str(item["cik_str"]).zfill(10), "title": item.get("title", "")}
    return _TICKER_MAP


def resolve(identifier: str) -> tuple[str, str]:
    """Ticker or CIK -> (ticker, 10-digit CIK)."""
    ident = identifier.strip().upper()
    if ident.isdigit():
        cik = ident.zfill(10)
        ticker = next((t for t, v in ticker_map().items() if v["cik"] == cik), "")
        return ticker, cik
    mapping = ticker_map()
    for candidate in (ident, ident.replace(".", "-"), ident.replace("-", ".")):
        if candidate in mapping:
            return candidate, mapping[candidate]["cik"]
    raise KeyError(
        f"{identifier}: not found in SEC company_tickers.json (the ticker may have changed or the company may no longer file); "
        "enter its CIK instead or update the ticker list"
    )


# ------------------------------------------------------------------ Input A

@dataclass
class FetchedFacts:
    cik: str
    data: dict
    url: str
    retrieved: str


def fetch_companyfacts(cik: str) -> FetchedFacts:
    url = SEC_FACTS_URL.format(cik=str(cik).zfill(10))
    response = _get(url, sec_headers())
    return FetchedFacts(str(cik).zfill(10), response.json(), url, datetime.now(timezone.utc).isoformat(timespec="seconds"))


# Input A supplement (see filings.py): periodic filings that companyfacts has not indexed are read from
# each filing's own XBRL instance on EDGAR. False = only detect them and warn under Data_Quality.
FILL_UNINDEXED_FILINGS = True


def fetch_submissions(cik: str) -> dict:
    return _get(filings.SUBMISSIONS_URL.format(cik=str(cik).zfill(10)), sec_headers()).json()


def unindexed_filings(cik: str, companyfacts: dict, as_of, read: bool = True) -> list[filings.UnindexedFiling]:
    """Periodic filings on or before `as_of` missing from `companyfacts`; with `read`, their XBRL facts are parsed."""
    found = filings.find_unindexed(companyfacts, fetch_submissions(cik), as_of)
    for f in found:
        if not read:
            f.instance = "FILL_UNINDEXED_FILINGS is off"
            continue
        folder = f.accn.replace("-", "")
        try:
            index = _get(filings.ARCHIVE_INDEX_URL.format(cik=int(cik), folder=folder), sec_headers(), retry_not_found=True).json()
            name = filings.pick_instance(index, f.primary_document)
            if not name:
                f.instance = "no XBRL instance in the filing folder"
                continue
            data = _get(filings.ARCHIVE_FILE_URL.format(cik=int(cik), folder=folder, name=name), sec_headers(), timeout=180, retry_not_found=True).content
            f.rows = filings.parse_instance(data, cik, f.accn, f.form, f.filed)
        except Exception as exc:  # one unreadable filing must not stop the review
            f.instance = f"could not read: {type(exc).__name__}: {exc}"
            continue
        f.facts = sum(len(items) for tags in f.rows.values() for by_unit in tags.values() for items in by_unit.values())
        f.instance = name if f.facts else f"{name} has no non-dimensional facts for this CIK"
    return found


# ------------------------------------------------------------------ Input C

@dataclass
class FetchedPrices:
    symbol: str
    frame: pd.DataFrame  # date, close, dividend, split_ratio
    meta: PriceMeta
    url: str
    retrieved: str
    exchange: str


def yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def fetch_prices(ticker: str, start: str = "1990-01-01") -> FetchedPrices:
    symbol = yahoo_symbol(ticker)
    url = YAHOO_CHART_URL.format(symbol=symbol)
    params = {
        "period1": int(pd.Timestamp(start, tz="UTC").timestamp()),
        "period2": int(time.time()) + 86400,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    response = _get(url, {"User-Agent": "Mozilla/5.0"}, params=params)
    payload = response.json().get("chart", {})
    if payload.get("error") or not payload.get("result"):
        raise RuntimeError(f"Yahoo returned no data for {symbol}: {payload.get('error')}")
    result = payload["result"][0]
    meta = result.get("meta", {})
    tz = meta.get("exchangeTimezoneName") or "America/New_York"

    def to_dates(seconds) -> pd.DatetimeIndex:
        return pd.to_datetime(seconds, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()

    stamps = result.get("timestamp") or []
    closes = (result.get("indicators", {}).get("quote") or [{}])[0].get("close") or []
    frame = pd.DataFrame({"date": to_dates(stamps), "close": pd.to_numeric(pd.Series(closes, dtype="float64"), errors="coerce")})
    frame = frame.drop_duplicates("date", keep="last")
    events = result.get("events", {}) or {}
    dividends = events.get("dividends", {}) or {}
    splits = events.get("splits", {}) or {}
    div = pd.DataFrame({"date": to_dates([d["date"] for d in dividends.values()]), "dividend": [float(d["amount"]) for d in dividends.values()]}) if dividends else pd.DataFrame(columns=["date", "dividend"])
    spl = (
        pd.DataFrame({"date": to_dates([s["date"] for s in splits.values()]), "split_ratio": [float(s["numerator"]) / float(s["denominator"]) for s in splits.values()]})
        if splits
        else pd.DataFrame(columns=["date", "split_ratio"])
    )
    frame = frame.merge(div.groupby("date", as_index=False)["dividend"].sum(), on="date", how="outer") if not div.empty else frame.assign(dividend=0.0)
    frame = frame.merge(spl.groupby("date", as_index=False)["split_ratio"].prod(), on="date", how="outer") if not spl.empty else frame.assign(split_ratio=float("nan"))
    frame = frame.sort_values("date").reset_index(drop=True)
    price_meta = PriceMeta(ticker=ticker, split_adjusted=True, dividend_adjusted=False, currency=str(meta.get("currency") or "").upper())
    return FetchedPrices(symbol, frame, price_meta, response.url.split("?")[0], datetime.now(timezone.utc).isoformat(timespec="seconds"), str(meta.get("exchangeName") or ""))

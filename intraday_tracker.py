import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


DEFAULT_NAV = 100.0
DEFAULT_REFRESH_SECONDS = 60
DEFAULT_PORTFOLIO_CSV = """ticker,target_weight
AAPL,0.20
MSFT,0.20
NVDA,0.20
AMZN,0.20
GOOGL,0.20
"""


@dataclass
class QuoteResult:
    price: float
    prev_close: Optional[float]
    timestamp: Optional[pd.Timestamp]
    source: str


def load_portfolio(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("portfolio.csv"):
        df = pd.read_csv("portfolio.csv")
    else:
        df = pd.read_csv(StringIO(DEFAULT_PORTFOLIO_CSV))

    required = {"ticker", "target_weight"}
    missing = required - set(df.columns.str.lower())
    if missing:
        normalized_columns = {c.lower(): c for c in df.columns}
        if required - set(normalized_columns):
            raise ValueError("Portfolio file must contain 'ticker' and 'target_weight' columns.")
        df = df.rename(columns={normalized_columns["ticker"]: "ticker", normalized_columns["target_weight"]: "target_weight"})

    if "ticker" not in df.columns or "target_weight" not in df.columns:
        df.columns = [c.lower() for c in df.columns]

    df = df[["ticker", "target_weight"]].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce")
    df = df.dropna(subset=["ticker", "target_weight"])
    if df.empty:
        raise ValueError("Portfolio file is empty after parsing.")

    total_weight = df["target_weight"].sum()
    if total_weight <= 0:
        raise ValueError("Portfolio weights must sum to a positive number.")

    df["normalized_weight"] = df["target_weight"] / total_weight
    return df.sort_values("ticker").reset_index(drop=True)


def get_provider_name(provider: str) -> str:
    return {
        "yfinance": "Yahoo Finance via yfinance",
        "alpaca": "Alpaca",
        "finnhub": "Finnhub",
        "polygon": "Polygon",
    }.get(provider, provider)


def fetch_quotes_yfinance(tickers: List[str]) -> Dict[str, QuoteResult]:
    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    hist = yf.download(
        tickers=tickers,
        period="2d",
        interval="1m",
        group_by="ticker",
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=True,
    )

    results: Dict[str, QuoteResult] = {}
    if len(tickers) == 1:
        hist = pd.concat({tickers[0]: hist}, axis=1)

    for ticker in tickers:
        if ticker not in hist.columns.get_level_values(0):
            continue
        frame = hist[ticker].dropna(how="all").copy()
        if frame.empty:
            continue

        frame.index = pd.to_datetime(frame.index)
        last_close = float(frame["Close"].dropna().iloc[-1])
        last_ts = pd.Timestamp(frame["Close"].dropna().index[-1])

        day_groups = frame["Close"].dropna().groupby(frame["Close"].dropna().index.date)
        prev_close = None
        if len(day_groups) >= 2:
            prev_close = float(day_groups.last().iloc[-2])
        elif len(frame["Close"].dropna()) >= 2:
            prev_close = float(frame["Close"].dropna().iloc[-2])

        results[ticker] = QuoteResult(
            price=last_close,
            prev_close=prev_close,
            timestamp=last_ts,
            source="yfinance",
        )

    return results


def fetch_quotes_alpaca(tickers: List[str], api_key: str, api_secret: str, feed: str) -> Dict[str, QuoteResult]:
    url = "https://data.alpaca.markets/v2/stocks/quotes/latest"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret}
    params = {"symbols": ",".join(tickers), "feed": feed}
    response = requests.get(url, headers=headers, params=params, timeout=20)
    response.raise_for_status()
    data = response.json().get("quotes", {})

    prev_url = "https://data.alpaca.markets/v2/stocks/snapshots"
    prev_response = requests.get(prev_url, headers=headers, params={"symbols": ",".join(tickers), "feed": feed}, timeout=20)
    prev_response.raise_for_status()
    snapshots = prev_response.json().get("snapshots", {})

    results: Dict[str, QuoteResult] = {}
    for ticker, quote in data.items():
        price = quote.get("ap") or quote.get("bp")
        if price is None:
            continue
        ts = pd.to_datetime(quote.get("t"), utc=True) if quote.get("t") else None
        prev_close = snapshots.get(ticker, {}).get("prevDailyBar", {}).get("c")
        results[ticker] = QuoteResult(price=float(price), prev_close=prev_close, timestamp=ts, source="alpaca")
    return results


def fetch_quotes_finnhub(tickers: List[str], api_key: str) -> Dict[str, QuoteResult]:
    results: Dict[str, QuoteResult] = {}
    for ticker in tickers:
        response = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": ticker, "token": api_key},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        price = data.get("c")
        if not price:
            continue
        ts = pd.to_datetime(data.get("t"), unit="s", utc=True) if data.get("t") else None
        results[ticker] = QuoteResult(
            price=float(price),
            prev_close=data.get("pc"),
            timestamp=ts,
            source="finnhub",
        )
    return results


def fetch_quotes_polygon(tickers: List[str], api_key: str) -> Dict[str, QuoteResult]:
    results: Dict[str, QuoteResult] = {}
    for ticker in tickers:
        response = requests.get(
            f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
            params={"apiKey": api_key},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json().get("ticker", {})
        minute = data.get("min", {})
        prev_day = data.get("prevDay", {})
        day = data.get("day", {})
        price = minute.get("c") or day.get("c")
        if price is None:
            continue
        updated = data.get("updated")
        ts = pd.to_datetime(updated, unit="ms", utc=True) if updated else None
        results[ticker] = QuoteResult(
            price=float(price),
            prev_close=prev_day.get("c"),
            timestamp=ts,
            source="polygon",
        )
    return results


def fetch_quotes(provider: str, tickers: List[str], settings: Dict[str, str]) -> Dict[str, QuoteResult]:
    if provider == "yfinance":
        return fetch_quotes_yfinance(tickers)
    if provider == "alpaca":
        return fetch_quotes_alpaca(tickers, settings["api_key"], settings["api_secret"], settings["feed"])
    if provider == "finnhub":
        return fetch_quotes_finnhub(tickers, settings["api_key"])
    if provider == "polygon":
        return fetch_quotes_polygon(tickers, settings["api_key"])
    raise ValueError(f"Unsupported provider: {provider}")


def compute_nav(quotes_df: pd.DataFrame, start_nav: float) -> float:
    return float(start_nav * (quotes_df["normalized_weight"] * (quotes_df["price"] / quotes_df["base_price"])).sum())


def init_base_prices(quotes_df: pd.DataFrame) -> pd.DataFrame:
    return quotes_df[["ticker", "price"]].rename(columns={"price": "base_price"})


def ensure_session_state(portfolio: pd.DataFrame, quotes_df: pd.DataFrame) -> None:
    portfolio_signature = tuple(zip(portfolio["ticker"], portfolio["normalized_weight"].round(10)))
    existing_signature = st.session_state.get("portfolio_signature")
    if existing_signature != portfolio_signature:
        st.session_state["portfolio_signature"] = portfolio_signature
        st.session_state["base_prices"] = init_base_prices(quotes_df)
        st.session_state["nav_history"] = []
        st.session_state["last_nav"] = None


def build_quotes_frame(portfolio: pd.DataFrame, quotes: Dict[str, QuoteResult]) -> pd.DataFrame:
    rows = []
    for _, row in portfolio.iterrows():
        quote = quotes.get(row["ticker"])
        if quote is None:
            continue
        rows.append(
            {
                "ticker": row["ticker"],
                "target_weight": row["target_weight"],
                "normalized_weight": row["normalized_weight"],
                "price": quote.price,
                "prev_close": quote.prev_close,
                "timestamp": quote.timestamp,
                "source": quote.source,
            }
        )
    if not rows:
        raise RuntimeError("No quotes were returned for the portfolio.")
    return pd.DataFrame(rows)


def append_nav_history(nav: float, now_ts: pd.Timestamp) -> None:
    history = st.session_state.get("nav_history", [])
    history.append({"timestamp": now_ts, "nav": nav})
    cutoff = now_ts - pd.Timedelta(days=1)
    history = [x for x in history if x["timestamp"] >= cutoff]
    st.session_state["nav_history"] = history


def main() -> None:
    st.set_page_config(page_title="Synthetic ETF Tracker", layout="wide")
    st.title("Synthetic ETF Tracker")
    st.caption("Tracks a custom weighted basket and estimates a synthetic ETF NAV on a roughly one-minute refresh.")

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox(
            "Data source",
            ["yfinance", "alpaca", "finnhub", "polygon"],
            index=0,
            format_func=get_provider_name,
        )
        start_nav = st.number_input("Starting NAV", min_value=1.0, value=DEFAULT_NAV, step=1.0)
        refresh_seconds = st.slider("Refresh cadence (seconds)", min_value=30, max_value=300, value=DEFAULT_REFRESH_SECONDS, step=30)
        uploaded_file = st.file_uploader("Portfolio CSV", type=["csv"])

        provider_settings: Dict[str, str] = {}
        if provider == "alpaca":
            provider_settings["api_key"] = st.text_input("Alpaca API key", value=os.getenv("ALPACA_API_KEY", ""), type="password")
            provider_settings["api_secret"] = st.text_input("Alpaca API secret", value=os.getenv("ALPACA_API_SECRET", ""), type="password")
            provider_settings["feed"] = st.selectbox("Alpaca feed", ["iex", "delayed_sip", "sip"], index=0)
        elif provider == "finnhub":
            provider_settings["api_key"] = st.text_input("Finnhub API key", value=os.getenv("FINNHUB_API_KEY", ""), type="password")
        elif provider == "polygon":
            provider_settings["api_key"] = st.text_input("Polygon API key", value=os.getenv("POLYGON_API_KEY", ""), type="password")

    try:
        portfolio = load_portfolio(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if provider != "yfinance":
        missing_keys = [k for k, v in provider_settings.items() if k != "feed" and not v]
        if missing_keys:
            st.warning(f"Missing credentials for {get_provider_name(provider)}: {', '.join(missing_keys)}")
            st.stop()

    tickers = portfolio["ticker"].tolist()

    try:
        quotes = fetch_quotes(provider, tickers, provider_settings)
        quotes_df = build_quotes_frame(portfolio, quotes)
    except Exception as exc:
        st.error(f"Quote fetch failed: {exc}")
        st.stop()

    ensure_session_state(portfolio, quotes_df)
    base_prices = st.session_state["base_prices"]
    quotes_df = quotes_df.merge(base_prices, on="ticker", how="left")
    quotes_df["base_price"] = quotes_df["base_price"].fillna(quotes_df["price"])

    nav = compute_nav(quotes_df, start_nav)
    last_nav = st.session_state.get("last_nav")
    minute_return = ((nav / last_nav) - 1.0) if last_nav else 0.0

    quotes_df["day_return"] = (quotes_df["price"] / quotes_df["prev_close"] - 1.0).where(quotes_df["prev_close"].notna())
    day_nav = float(start_nav * (quotes_df["normalized_weight"] * (quotes_df["price"] / quotes_df["prev_close"])).fillna(quotes_df["normalized_weight"]).sum())
    day_return = (day_nav / start_nav) - 1.0

    now_ts = pd.Timestamp.now(tz=timezone.utc)
    append_nav_history(nav, now_ts)
    st.session_state["last_nav"] = nav

    quotes_df["weight_pct"] = quotes_df["normalized_weight"] * 100.0
    quotes_df["nav_contribution"] = start_nav * quotes_df["normalized_weight"] * (quotes_df["price"] / quotes_df["base_price"])
    quotes_df["day_contribution_pct"] = quotes_df["normalized_weight"] * quotes_df["day_return"]
    quotes_df = quotes_df.sort_values("weight_pct", ascending=False).reset_index(drop=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Synthetic NAV", f"{nav:,.2f}")
    col2.metric("1-minute return", f"{minute_return * 100:.2f}%")
    col3.metric("Day return", f"{day_return * 100:.2f}%")
    col4.metric("Tracked names", f"{len(quotes_df)}/{len(portfolio)}")

    history_df = pd.DataFrame(st.session_state["nav_history"])
    if not history_df.empty:
        history_df = history_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        st.line_chart(history_df.set_index("timestamp")["nav"], height=320)

    st.subheader("Constituents")
    display_cols = [
        "ticker",
        "price",
        "prev_close",
        "weight_pct",
        "day_return",
        "day_contribution_pct",
        "nav_contribution",
        "timestamp",
        "source",
    ]
    st.dataframe(
        quotes_df[display_cols].rename(
            columns={
                "price": "Price",
                "prev_close": "Prev Close",
                "weight_pct": "Weight %",
                "day_return": "Day Return",
                "day_contribution_pct": "Weighted Day Return",
                "nav_contribution": "NAV Contribution",
                "timestamp": "Quote Time",
                "source": "Source",
                "ticker": "Ticker",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Portfolio File Format")
    st.code(DEFAULT_PORTFOLIO_CSV.strip(), language="csv")

    st.info(
        "Yahoo mode is convenient and key-free but should be treated as delayed/near-live and best-effort. "
        "API providers can offer better live coverage, but plan limits and exchange entitlements still matter."
    )

    # Lightweight client-side refresh without extra dependencies.
    refresh_ms = refresh_seconds * 1000
    st.markdown(
        f"""
        <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {refresh_ms});
        </script>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

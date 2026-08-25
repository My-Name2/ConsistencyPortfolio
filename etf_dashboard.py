from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:  # pragma: no cover
    # Altair can fail to import on a newly released Python runtime. Native
    # Streamlit charts keep the hosted app usable in that case.
    alt = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


APP_DIR = Path(__file__).resolve().parent
PORTFOLIO_PATH = APP_DIR / "portfolio.csv"
DEFAULT_INCEPTION = date(2026, 8, 14)
DEFAULT_NAV = 100.0
DEFAULT_RISK_FREE = 0.04


def load_portfolio(path: Path = PORTFOLIO_PATH) -> pd.DataFrame:
    portfolio = pd.read_csv(path)
    portfolio.columns = [str(column).strip().lower() for column in portfolio.columns]
    required = {"ticker", "target_weight"}
    if not required.issubset(portfolio.columns):
        raise ValueError("portfolio.csv must contain ticker and target_weight columns.")

    portfolio = portfolio[["ticker", "target_weight"]].copy()
    portfolio["ticker"] = portfolio["ticker"].astype(str).str.strip().str.upper()
    portfolio["target_weight"] = pd.to_numeric(portfolio["target_weight"], errors="coerce")
    portfolio = portfolio.dropna(subset=["ticker", "target_weight"])
    portfolio = portfolio[portfolio["target_weight"] > 0]
    if portfolio.empty:
        raise ValueError("The portfolio has no positive weights.")

    portfolio = portfolio.groupby("ticker", as_index=False)["target_weight"].sum()
    portfolio["weight"] = portfolio["target_weight"] / portfolio["target_weight"].sum()
    return portfolio.sort_values("weight", ascending=False).reset_index(drop=True)


def extract_close(data: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    tickers = list(tickers)
    if isinstance(data.columns, pd.MultiIndex):
        level_zero = set(data.columns.get_level_values(0))
        level_one = set(data.columns.get_level_values(1))
        if "Close" in level_zero:
            close = data["Close"]
        elif "Close" in level_one:
            close = data.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Yahoo Finance returned no Close data.")
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.columns = [str(column).upper() for column in close.columns]
    return close.reindex(columns=tickers)


@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers: tuple[str, ...], start: date, end: date, adjusted: bool) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run pip install -r requirements.txt.")
    raw = yf.download(
        list(tickers),
        start=start - timedelta(days=5),
        end=end + timedelta(days=1),
        auto_adjust=adjusted,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    close = extract_close(raw, tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.loc[close.index >= pd.Timestamp(start)]
    return close.apply(pd.to_numeric, errors="coerce").sort_index()


def build_index(
    prices: pd.DataFrame, portfolio: pd.DataFrame, starting_nav: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = portfolio.set_index("ticker")["weight"]
    available = prices.notna().sum().rename("observations")
    first_prices = prices.apply(lambda series: series.dropna().iloc[0] if series.notna().any() else pd.NA)
    relative = prices.divide(first_prices, axis="columns")
    # Hold an unavailable sleeve flat until its first quote, and flag it in coverage.
    relative = relative.ffill().fillna(1.0).reindex(columns=weights.index)
    nav = starting_nav * relative.mul(weights, axis="columns").sum(axis=1).rename("Synthetic ETF")

    details = pd.DataFrame(index=weights.index)
    details["Weight"] = weights
    details["Base price"] = first_prices.reindex(weights.index)
    details["Latest price"] = prices.iloc[-1].reindex(weights.index)
    details["Price return"] = relative.iloc[-1].reindex(weights.index) - 1.0
    details["Contribution"] = details["Weight"] * details["Price return"]
    details["Observations"] = available.reindex(weights.index).fillna(0).astype(int)
    details.index.name = "Ticker"
    return pd.DataFrame({"Synthetic ETF": nav}), details.reset_index()


def metrics(series: pd.Series, benchmark: pd.Series | None, risk_free: float) -> dict[str, float]:
    series = series.dropna()
    if series.empty:
        return {}
    daily = series.pct_change().dropna()
    years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
    total_return = series.iloc[-1] / series.iloc[0] - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    annual_vol = daily.std(ddof=1) * (252**0.5) if len(daily) > 1 else 0.0
    sharpe = ((daily.mean() * 252) - risk_free) / annual_vol if annual_vol else float("nan")
    drawdown = series / series.cummax() - 1
    result = {
        "Total return": total_return,
        "CAGR": cagr,
        "Annualized volatility": annual_vol,
        "Sharpe ratio": sharpe,
        "Maximum drawdown": drawdown.min(),
        "Positive days": (daily > 0).mean() if len(daily) else float("nan"),
    }
    if benchmark is not None and not benchmark.empty:
        aligned = pd.concat([daily.rename("portfolio"), benchmark.pct_change().rename("benchmark")], axis=1).dropna()
        if len(aligned) > 1:
            result["Correlation to SPY"] = aligned["portfolio"].corr(aligned["benchmark"])
            result["Beta to SPY"] = aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var()
            result["Active return"] = total_return - (benchmark.iloc[-1] / benchmark.iloc[0] - 1)
    return result


def format_metric(value: float, name: str) -> str:
    if pd.isna(value):
        return "n/a"
    if "ratio" in name.lower() or "beta" in name.lower():
        return f"{value:.2f}"
    return f"{value:.1%}"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #08111f;
            --panel: #101c2d;
            --panel-2: #15243a;
            --line: #263a54;
            --muted: #8fa3ba;
            --text: #edf4fb;
            --cyan: #61d4e6;
            --green: #54d39b;
            --red: #ff7f86;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #07101d 0%, #0b1626 48%, #0a1220 100%);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #0a1525;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        [data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 2rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        h1 { font-size: 2.2rem !important; margin-bottom: 0.25rem !important; }
        h2 { font-size: 1.25rem !important; margin-top: 1.6rem !important; }
        h3 { font-size: 0.95rem !important; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted) !important; }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(20, 37, 59, .95), rgba(14, 27, 44, .95));
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, .16);
        }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; }
        [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1.6rem; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
        [data-testid="stExpander"] { border-color: var(--line); background: rgba(16, 28, 45, .55); }
        .hero {
            display: flex; justify-content: space-between; gap: 1rem; align-items: flex-end;
            padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; border-radius: 14px;
            background: radial-gradient(circle at 90% 10%, rgba(97,212,230,.16), transparent 36%), linear-gradient(135deg, #12253b, #0e1929);
            border: 1px solid var(--line); box-shadow: 0 14px 36px rgba(0,0,0,.18);
        }
        .hero-kicker { color: var(--cyan); font-size: .72rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; }
        .hero-title { color: var(--text); font-size: 1.7rem; font-weight: 700; margin-top: .3rem; }
        .hero-copy { color: var(--muted); max-width: 700px; font-size: .92rem; margin-top: .35rem; }
        .hero-meta { text-align: right; color: var(--muted); font-size: .78rem; line-height: 1.7; white-space: nowrap; }
        .hero-meta strong { color: var(--text); }
        .section-note { color: var(--muted); font-size: .86rem; line-height: 1.55; margin: -.35rem 0 1rem; }
        .positive { color: var(--green); font-weight: 700; }
        .negative { color: var(--red); font-weight: 700; }
        @media (max-width: 800px) {
            .hero { display: block; }
            .hero-meta { text-align: left; margin-top: .8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def performance_chart(chart: pd.DataFrame):
    if alt is None:
        return None
    frame = chart.reset_index().rename(columns={chart.index.name or "index": "Date"})
    frame = frame.melt("Date", var_name="Series", value_name="Indexed value").dropna()
    return (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Indexed value:Q", title="Indexed value", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=["Synthetic ETF", "SPY"], range=["#61d4e6", "#a4b3c6"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Series:N"), alt.Tooltip("Indexed value:Q", format=".2f")],
        )
        .properties(height=360)
        .interactive()
    )


def active_return_chart(chart: pd.DataFrame):
    if alt is None:
        return None
    if not {"Synthetic ETF", "SPY"}.issubset(chart.columns):
        return None
    active = (chart["Synthetic ETF"] / chart["SPY"] - 1.0).mul(100).rename("Active return")
    frame = active.rename_axis("Date").reset_index()
    return (
        alt.Chart(frame)
        .mark_area(
            line={"color": "#54d39b"},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#54d39b", offset=0), alt.GradientStop(color="#183a42", offset=1)],
                x1=1,
                x2=1,
                y1=0,
                y2=1,
            ),
        )
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Active return:Q", title="Active return vs SPY (%)"),
            tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Active return:Q", format=".2f", title="Spread (%)")],
        )
        .properties(height=230)
        .interactive()
    )


def main() -> None:
    st.set_page_config(page_title="Custom ETF | Research Dashboard", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    portfolio = load_portfolio()

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="hero-kicker">Portfolio research terminal · synthetic ETF</div>
            <div class="hero-title">Quality Compounders Basket</div>
            <div class="hero-copy">A transparent, buy-and-hold simulation of the supplied 50-stock basket, with daily total-return-style attribution and a direct SPY benchmark.</div>
          </div>
          <div class="hero-meta"><strong>50 constituents</strong><br/>Target-weighted<br/>Daily observations</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Research controls")
        inception = st.date_input(
            "Inception date",
            value=DEFAULT_INCEPTION,
            min_value=date(2010, 1, 1),
            max_value=date.today(),
            help="The original image has no embedded timestamp. This defaults to 2026-08-14, the date associated with the ETF work in this project.",
        )
        starting_nav = st.number_input("Starting NAV", min_value=1.0, value=DEFAULT_NAV, step=10.0)
        risk_free = st.number_input("Annual risk-free rate", min_value=0.0, max_value=0.25, value=DEFAULT_RISK_FREE, step=0.005, format="%.3f")
        adjusted = st.checkbox("Include dividends and splits", value=True, help="Adjusted prices make this a total-return-style comparison rather than a price-only index.")
        show_constituents = st.slider("Constituents to show", min_value=5, max_value=len(portfolio), value=15)
        if st.button("Refresh downloaded data"):
            download_prices.clear()
            st.rerun()

    if inception > date.today():
        st.error("Inception date cannot be in the future.")
        return

    tickers = tuple(portfolio["ticker"].tolist())
    benchmark_tickers = tickers + ("SPY",)
    with st.spinner("Downloading daily price history..."):
        try:
            prices = download_prices(benchmark_tickers, inception, date.today(), adjusted)
        except Exception as exc:
            st.error(f"Could not download price history: {exc}")
            st.info("Check your internet connection, then use Refresh downloaded data.")
            return

    available_portfolio = prices.reindex(columns=tickers)
    missing = [ticker for ticker in tickers if available_portfolio[ticker].notna().sum() == 0]
    if missing:
        st.error("No price history was returned for: " + ", ".join(missing))
        return

    index, details = build_index(available_portfolio, portfolio, starting_nav)
    spy = prices["SPY"].dropna() if "SPY" in prices else pd.Series(dtype=float)
    if not spy.empty:
        spy = starting_nav * spy / spy.iloc[0]
        chart = pd.concat([index, spy.rename("SPY")], axis=1).dropna(how="all")
    else:
        chart = index

    portfolio_metrics = metrics(index["Synthetic ETF"], spy, risk_free)
    latest_date = index.index[-1].date()
    latest_nav = index["Synthetic ETF"].iloc[-1]
    first_nav = index["Synthetic ETF"].iloc[0]
    spy_total = (spy.iloc[-1] / spy.iloc[0] - 1) if not spy.empty else float("nan")

    st.markdown(
        f"<div class='section-note'>Observation window: <strong>{inception}</strong> to <strong>{latest_date}</strong>. The basket and SPY are both rebased to {starting_nav:,.0f}; active return is the basket's cumulative return minus SPY's cumulative return over the same dates.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Executive performance")
    headline = st.columns(5)
    headline[0].metric("Synthetic NAV", f"{latest_nav:,.2f}", f"{latest_nav / first_nav - 1:.1%} since start")
    headline[1].metric("Basket return", f"{portfolio_metrics.get('Total return', float('nan')):.1%}")
    headline[2].metric("SPY return", f"{spy_total:.1%}" if not pd.isna(spy_total) else "n/a", "same window")
    active_return = portfolio_metrics.get("Active return", float("nan"))
    headline[3].metric("Return vs SPY", f"{active_return:+.1%}" if not pd.isna(active_return) else "n/a", "outperformance" if active_return >= 0 else "underperformance")
    headline[4].metric("CAGR", f"{portfolio_metrics.get('CAGR', float('nan')):.1%}", str(latest_date))

    st.subheader("Cumulative return vs SPY")
    st.markdown("<div class='section-note'>The distance between the lines is the investor-facing result: how much the basket has gained or lost relative to simply owning SPY. A rising gap means the basket is adding value; a falling gap means SPY is ahead.</div>", unsafe_allow_html=True)
    if alt is not None:
        st.altair_chart(performance_chart(chart), use_container_width=True)
    else:
        st.line_chart(chart, y_label="Indexed value", x_label="Date", height=360)

    active_chart = active_return_chart(chart) if alt is not None else None
    if active_chart is not None:
        st.subheader("Active return spread")
        st.markdown("<div class='section-note'>This is the cumulative percentage-point spread versus SPY, not a volatility-adjusted alpha estimate. It includes the effect of the supplied weights, price movement, dividends, and splits under the selected data mode.</div>", unsafe_allow_html=True)
        st.altair_chart(active_chart, use_container_width=True)
    elif {"Synthetic ETF", "SPY"}.issubset(chart.columns):
        st.subheader("Active return spread")
        st.markdown("<div class='section-note'>This is the cumulative percentage-point spread versus SPY, not a volatility-adjusted alpha estimate. It includes the effect of the supplied weights, price movement, dividends, and splits under the selected data mode.</div>", unsafe_allow_html=True)
        active = (chart["Synthetic ETF"] / chart["SPY"] - 1.0).mul(100).rename("Active return vs SPY")
        st.line_chart(active, y_label="Active return (%)", x_label="Date", height=230)

    left, right = st.columns(2)
    with left:
        st.subheader("Risk and efficiency")
        st.markdown("<div class='section-note'>Sharpe uses the risk-free rate from the sidebar. Drawdown measures the decline from the basket's prior high-water mark.</div>", unsafe_allow_html=True)
        metric_rows = [
            ("CAGR", "CAGR"),
            ("Annualized volatility", "Annualized volatility"),
            ("Sharpe ratio", "Sharpe ratio"),
            ("Maximum drawdown", "Maximum drawdown"),
            ("Positive days", "Positive days"),
            ("Correlation to SPY", "Correlation to SPY"),
            ("Beta to SPY", "Beta to SPY"),
        ]
        risk_table = pd.DataFrame({"Metric": [row[0] for row in metric_rows], "Value": [format_metric(portfolio_metrics.get(row[1], float("nan")), row[1]) for row in metric_rows]})
        st.dataframe(risk_table, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Drawdown")
        st.markdown("<div class='section-note'>Peak-to-trough loss for the basket and SPY.</div>", unsafe_allow_html=True)
        drawdowns = chart.divide(chart.cummax()).subtract(1).rename(columns=lambda column: f"{column} drawdown")
        st.area_chart(drawdowns, y_label="Drawdown", x_label="Date", height=260)

    st.subheader("Attribution by constituent")
    st.markdown("<div class='section-note'>Contribution is the constituent's return multiplied by its normalized target weight. It is the additive contribution to the basket's total return, so the table explains what drove the result rather than just ranking raw stock returns.</div>", unsafe_allow_html=True)
    display = details.sort_values("Contribution", ascending=False).copy()
    display["Weight"] = display["Weight"].map(lambda value: f"{value:.2%}")
    display["Price return"] = display["Price return"].map(lambda value: f"{value:.2%}")
    display["Contribution"] = display["Contribution"].map(lambda value: f"{value:.2%}")
    display["Base price"] = display["Base price"].map(lambda value: "n/a" if pd.isna(value) else f"${value:,.2f}")
    display["Latest price"] = display["Latest price"].map(lambda value: "n/a" if pd.isna(value) else f"${value:,.2f}")
    display = display.rename(columns={"Base price": "Inception price", "Price return": "Return", "Observations": "Trading days"})
    st.dataframe(display.head(show_constituents), hide_index=True, use_container_width=True)

    short_history = details[details["Observations"] < len(index) - 5]
    if not short_history.empty:
        st.warning("Short or delayed history: " + ", ".join(short_history["Ticker"].tolist()) + ". Their sleeve is held flat before its first available quote, so earlier inception dates are approximate for those names.")

    with st.expander("Basket and methodology"):
        st.write(f"The basket contains {len(portfolio)} tickers. Raw supplied weights sum to {portfolio['target_weight'].sum():.6f}; the dashboard normalizes them to {portfolio['weight'].sum():.2%}.")
        st.code(" + ".join(f"{row.ticker} {row.weight:.6f}" for row in portfolio.itertuples(index=False)), language="text")
        st.markdown("**NAV calculation:** each constituent is rebased to 1.00 on its first available quote, multiplied by its normalized target weight, and summed. Yahoo Finance adjusted prices are used by default, so the result is a total-return-style proxy. It excludes fees, taxes, bid/ask spreads, slippage, rebalancing, and corporate-action execution effects.")

    st.download_button("Download contribution CSV", data=details.to_csv(index=False).encode("utf-8"), file_name="custom_etf_contributions.csv", mime="text/csv")


if __name__ == "__main__":
    main()

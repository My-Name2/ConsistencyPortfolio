from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
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
DEFAULT_SCORE_PATH = APP_DIR / "consistency_snapshot.csv"
DEFAULT_INCEPTION = date(2026, 8, 14)
DEFAULT_NAV = 100.0
DEFAULT_RISK_FREE = 0.04


def _parse_weight(value: object) -> float:
    """Accept decimal weights and values copied from percentage-formatted tables."""
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace(",", "").replace("$", "")
    if text.endswith("%"):
        try:
            return float(text[:-1].strip()) / 100.0
        except ValueError:
            return float("nan")
    return pd.to_numeric(text, errors="coerce")


def load_portfolio(path: Path = PORTFOLIO_PATH) -> pd.DataFrame:
    portfolio = pd.read_csv(path, sep=None, engine="python")
    portfolio.columns = [str(column).strip().lower() for column in portfolio.columns]
    if "ticker" not in portfolio.columns:
        raise ValueError("Portfolio data must contain a ticker column.")

    weight_column = next(
        (
            column
            for column in ("target_weight", "target weight", "weight", "allocation")
            if column in portfolio.columns
        ),
        None,
    )
    if weight_column is None:
        raise ValueError("portfolio.csv must contain ticker and target_weight columns.")

    portfolio = portfolio[["ticker", weight_column]].rename(columns={weight_column: "target_weight"}).copy()
    portfolio["ticker"] = portfolio["ticker"].astype(str).str.strip().str.upper()
    portfolio["target_weight"] = portfolio["target_weight"].map(_parse_weight)
    portfolio = portfolio.dropna(subset=["ticker", "target_weight"])
    portfolio = portfolio[portfolio["target_weight"] > 0]
    if portfolio.empty:
        raise ValueError("No positive numeric weights were found. Check that the CSV includes data rows after the header.")

    portfolio = portfolio.groupby("ticker", as_index=False)["target_weight"].sum()
    portfolio["weight"] = portfolio["target_weight"] / portfolio["target_weight"].sum()
    return portfolio.sort_values("weight", ascending=False).reset_index(drop=True)


def _normalized_column_name(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _find_column(columns: Iterable[object], *candidates: str) -> str | None:
    normalized = {_normalized_column_name(column): str(column) for column in columns}
    for candidate in candidates:
        if _normalized_column_name(candidate) in normalized:
            return normalized[_normalized_column_name(candidate)]
    return None


def _as_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna().abs()
    if not valid.empty and valid.max() <= 1.5:
        values = values * 100.0
    return values


def load_consistency_snapshot(source: object | None = None) -> pd.DataFrame:
    """Load CompanyCharts-style score data and standardize the useful fields."""
    if source is None:
        if DEFAULT_SCORE_PATH.exists():
            snapshot = pd.read_csv(DEFAULT_SCORE_PATH)
        else:
            return pd.DataFrame()
    elif isinstance(source, (str, Path)):
        snapshot = pd.read_csv(source)
    else:
        snapshot = pd.read_csv(BytesIO(source.getvalue()))

    ticker_column = _find_column(snapshot.columns, "Ticker", "Symbol")
    adjusted_column = _find_column(snapshot.columns, "ConsistencyScore adjusted %Max")
    ttm_column = _find_column(snapshot.columns, "ConsistencyScore TTM adjusted %Max")
    if ticker_column is None or (adjusted_column is None and ttm_column is None):
        return pd.DataFrame()

    result = pd.DataFrame({"Ticker": snapshot[ticker_column].astype(str).str.strip().str.upper()})
    if adjusted_column is not None:
        result["Adjusted %Max"] = _as_percent(snapshot[adjusted_column])
    if ttm_column is not None:
        result["TTM adjusted %Max"] = _as_percent(snapshot[ttm_column])
    result = result[result["Ticker"].ne("")].drop_duplicates("Ticker")
    score_columns = [column for column in result.columns if column != "Ticker"]
    result["Average %Max"] = result[score_columns].mean(axis=1, skipna=True)
    return result


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
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            background: transparent !important;
            caret-color: var(--cyan) !important;
        }
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="input"],
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background: #122238 !important;
            border: 1px solid #2d4865 !important;
            border-radius: 9px !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within,
        [data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
            border-color: var(--cyan) !important;
            box-shadow: 0 0 0 1px var(--cyan) !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: var(--muted) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background: #1a3049 !important;
            border: 1px solid #3a5b7d !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
            color: var(--cyan) !important;
            background: transparent !important;
        }
        [data-testid="stSidebar"] [data-testid="stCheckbox"] div[role="checkbox"] {
            background: var(--cyan) !important;
            border-color: var(--cyan) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
            background: var(--cyan) !important;
            border-color: var(--cyan) !important;
            box-shadow: 0 0 0 2px rgba(97, 212, 230, .18) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] div[role="progressbar"] {
            background: var(--cyan) !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"] {
            background: #122238 !important;
            border-color: #2d4865 !important;
            color: var(--text) !important;
        }
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
        .brand-lockup { display: flex; align-items: center; gap: .65rem; }
        .brand-mark {
            display: inline-flex; align-items: center; justify-content: center;
            width: 2.2rem; height: 2.2rem; border-radius: 8px;
            color: #08111f; background: var(--cyan); font-size: .82rem; font-weight: 900;
            letter-spacing: -.08em; box-shadow: 0 0 24px rgba(97,212,230,.28);
        }
        .brand-name { color: var(--text); font-size: .82rem; font-weight: 800; letter-spacing: .12em; }
        .brand-subtitle { color: var(--cyan); font-size: .62rem; text-transform: uppercase; letter-spacing: .16em; margin-top: .15rem; }
        .hero-kicker { color: var(--cyan); font-size: .72rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 700; }
        .hero-title { color: var(--text); font-size: 1.7rem; font-weight: 700; margin-top: .3rem; }
        .hero-copy { color: var(--muted); max-width: 700px; font-size: .92rem; margin-top: .35rem; }
        .hero-meta { text-align: right; color: var(--muted); font-size: .78rem; line-height: 1.7; white-space: nowrap; }
        .hero-meta strong { color: var(--text); }
        .research-strip {
            display: flex; justify-content: space-between; gap: 1rem; align-items: center;
            padding: .65rem .9rem; margin: -.45rem 0 1.25rem; border-left: 3px solid var(--cyan);
            background: rgba(16, 28, 45, .72); color: var(--muted); font-size: .78rem;
        }
        .research-strip strong { color: var(--text); letter-spacing: .08em; }
        .brand-footer {
            margin-top: 2rem; padding-top: .9rem; border-top: 1px solid var(--line);
            color: var(--muted); font-size: .72rem; letter-spacing: .03em;
        }
        .brand-footer strong { color: var(--cyan); letter-spacing: .12em; }
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


def consistency_chart(frame: pd.DataFrame):
    if alt is None or frame.empty:
        return None
    return (
        alt.Chart(frame)
        .mark_circle(opacity=0.86, stroke="#dbeafe", strokeWidth=0.5)
        .encode(
            x=alt.X("Average %Max:Q", title="Average ConsistencyScore %Max", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Price return:Q", title="Price return (%)", axis=alt.Axis(format=".0%")),
            size=alt.Size("Weight:Q", title="Portfolio weight", scale=alt.Scale(range=[50, 900])),
            color=alt.Color("Weight:Q", title="Portfolio weight", scale=alt.Scale(scheme="tealblues")),
            tooltip=[
                alt.Tooltip("Average %Max:Q", format=".1f"),
                alt.Tooltip("Price return:Q", format=".1%"),
            ],
        )
        .properties(height=330)
        .interactive()
    )


def main() -> None:
    st.set_page_config(page_title="CompanyCharts | Synthetic ETF Research", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    try:
        portfolio = load_portfolio()
    except FileNotFoundError:
        st.error("Portfolio data is not configured yet.")
        st.info("Upload portfolio.csv to the same GitHub folder as etf_dashboard.py. The score snapshot is optional.")
        return
    except ValueError as exc:
        st.error("The portfolio data could not be read.")
        st.info("Use a CSV with a ticker column and a target_weight, weight, or allocation column. Weights may be decimals such as 0.022532 or percentages such as 2.2532%.")
        st.caption(f"Details: {exc}")
        return

    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="brand-lockup">
              <span class="brand-mark">CC</span>
              <div><div class="brand-name">COMPANYCHARTS</div><div class="brand-subtitle">Independent equity research</div></div>
            </div>
            <div class="hero-kicker">ConsistencyScore research · synthetic ETF</div>
            <div class="hero-title">Quality Compounders Basket</div>
            <div class="hero-copy">A transparent, buy-and-hold simulation of the supplied 50-stock basket, with daily attribution, a direct SPY benchmark, and an optional ConsistencyScore %Max research lens.</div>
          </div>
          <div class="hero-meta"><strong>RESEARCH DESK</strong><br/>ConsistencyScore + %Max<br/>50-stock model portfolio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='research-strip'><span><strong>COMPANYCHARTS RESEARCH NOTE</strong> &nbsp;|&nbsp; Fundamental consistency translated into a transparent portfolio monitor.</span><span>Signal: <strong>CONSISTENCY + COMPOUNDING</strong></span></div>",
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
        show_constituents = st.slider("Constituents to show", min_value=5, max_value=min(15, len(portfolio)), value=min(10, len(portfolio)))
        st.markdown("### CompanyCharts research lens")
        uploaded_scores = st.file_uploader(
            "ConsistencyScore snapshot CSV",
            type=["csv"],
            help="Optional override. Upload a newer private research export to replace the default snapshot.",
        )
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
    consistency_snapshot = load_consistency_snapshot(uploaded_scores) if uploaded_scores is not None else load_consistency_snapshot()
    if not consistency_snapshot.empty:
        details = details.merge(consistency_snapshot, on="Ticker", how="left")
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
    for score_column in ["Adjusted %Max", "TTM adjusted %Max", "Average %Max"]:
        if score_column in display.columns:
            display[score_column] = display[score_column].map(lambda value: "n/a" if pd.isna(value) else f"{value:.1f}%")
    display = display.rename(columns={"Base price": "Inception price", "Price return": "Return", "Observations": "Trading days"})
    st.dataframe(display.head(show_constituents), hide_index=True, use_container_width=True)

    if not consistency_snapshot.empty:
        st.subheader("CompanyCharts ConsistencyScore lens")
        score_source = "uploaded snapshot" if uploaded_scores is not None else "bundled research snapshot"
        st.markdown(
            f"<div class='section-note'>Scores are joined by ticker from the {score_source}. The research lens summarizes the persistence and direction of company fundamentals across available history, with normalization intended to improve comparability between companies.</div>",
            unsafe_allow_html=True,
        )
        lens = details.dropna(subset=["Average %Max"]).copy()
        if not lens.empty:
            score_cards = st.columns(3)
            score_cards[0].metric("Matched names", f"{len(lens)} / {len(details)}")
            score_cards[1].metric("Median average %Max", f"{lens['Average %Max'].median():.1f}%")
            score_cards[2].metric("Median TTM adjusted %Max", f"{lens['TTM adjusted %Max'].median():.1f}%" if "TTM adjusted %Max" in lens else "n/a")
            chart_frame = lens[["Ticker", "Weight", "Price return", "Average %Max"]].copy()
            if alt is not None:
                st.altair_chart(consistency_chart(chart_frame), use_container_width=True)
            else:
                st.scatter_chart(chart_frame, x="Average %Max", y="Price return", size="Weight", color="Weight", height=330)
            st.caption("Interpretation: the upper-right area combines stronger historical consistency with a positive observed return; this is descriptive attribution, not a forward-return guarantee.")

    short_history = details[details["Observations"] < len(index) - 5]
    if not short_history.empty:
        st.warning("Short or delayed history: " + ", ".join(short_history["Ticker"].tolist()) + ". Their sleeve is held flat before its first available quote, so earlier inception dates are approximate for those names.")

    with st.expander("Basket and methodology"):
        st.write(f"The model portfolio contains {len(portfolio)} constituents. The underlying weights are maintained in the portfolio configuration and are intentionally not reproduced in the public research view.")
        st.markdown("**NAV calculation:** each constituent is rebased to 1.00 on its first available quote, multiplied by its configured portfolio weight, and summed. Yahoo Finance adjusted prices are used by default, so the result is a total-return-style proxy. It excludes fees, taxes, bid/ask spreads, slippage, rebalancing, and corporate-action execution effects.")

    with st.expander("CompanyCharts ConsistencyScore and %Max methodology"):
        st.markdown(
            "**ConsistencyScore:** CompanyCharts' research framework evaluates the persistence and direction of fundamental operating trends over time. The adjusted views normalize the research signal across companies with different amounts of usable history."
        )
        st.markdown(
            "The dashboard presents the score as a research lens alongside realized portfolio results. It is descriptive, not a guarantee of future returns, and the underlying scoring specification is not reproduced in this public dashboard."
        )
        st.markdown(
            "The snapshot may contain multiple CompanyCharts score views, including a company-total lens and broader operating-quality views. This dashboard shows the available views without exposing the underlying construction details."
        )
        st.caption("CompanyCharts branding identifies the research framework represented here; the market-data series are downloaded independently from Yahoo Finance.")

    st.markdown(
        "<div class='brand-footer'><strong>COMPANYCHARTS</strong> &nbsp;|&nbsp; ConsistencyScore research dashboard &nbsp;|&nbsp; Synthetic ETF analytics for educational and research use.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

"""Valuation and fundamentals review GUI.

Run:  python -B -m streamlit run valuation_app.py --server.headless true --browser.gatherUsageStats false

Inputs are fetched into memory (SEC companyfacts, Yahoo chart history) or
uploaded; nothing is written to disk. Workbooks and CSVs are offered as
downloads built in memory.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # keep __pycache__ folders off the disk

import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from datetime import date  # noqa: E402

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from valuation import sources  # noqa: E402
from valuation.cli import fetch_inputs  # noqa: E402
from valuation.pipeline import ReviewInputs, run_review  # noqa: E402
from valuation.prices import PriceMeta  # noqa: E402
from valuation.ratios import DAILY_RATIOS, RATIO_BY_KEY  # noqa: E402
from valuation.redflags import InputBDocument  # noqa: E402
from valuation.report import ANNUAL_ROWS, LAUNCH, build_report, fmt_value  # noqa: E402
from valuation.stats import SERIES  # noqa: E402
from valuation.workbook import build_workbook  # noqa: E402

st.set_page_config(page_title="Valuation review", layout="wide")

FAMILY_LABELS = {
    "revenue": "Revenue", "gross_profit": "Gross profit", "earnings": "Earnings / EPS", "equity": "Parent equity / BVPS", "ocf": "Operating cash flow",
    "fcf": "Free cash flow", "da": "D&A", "ebitda": "EBITDA", "operating_income": "Operating income", "dividends": "Dividends",
    "capex": "Capex", "shares": "Diluted shares",
}
PALETTE = {
    "light": {"ratio": "#2a78d6", "held": "#eb6834", "rule": "#8a8984", "ink": "#52514e"},
    "dark": {"ratio": "#3987e5", "held": "#d95926", "rule": "#8a8984", "ink": "#c3c2b7"},
}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


# ------------------------------------------------------------------ cached work (memory only)

@st.cache_resource(max_entries=6, show_spinner=False)
def review_from_sources(identifier: str, as_of: date, current_price: float | None, docs: tuple[tuple[str, str], ...]):
    inputs = fetch_inputs(identifier, as_of, current_price, [InputBDocument(n, t) for n, t in docs])
    return run_review(inputs)


@st.cache_resource(max_entries=4, show_spinner=False)
def review_from_files(facts: bytes, prices: bytes | None, dividends: bytes | None, split_adjusted, dividend_adjusted, currency: str, ticker: str,
                      as_of: date, current_price: float | None, docs: tuple[tuple[str, str], ...]):
    data = json.loads(gzip.decompress(facts) if facts[:2] == b"\x1f\x8b" else facts)
    meta = PriceMeta(ticker=ticker, split_adjusted=split_adjusted, dividend_adjusted=dividend_adjusted, currency=currency)
    import io

    inputs = ReviewInputs(
        facts=data, as_of=as_of, prices=io.BytesIO(prices) if prices else None, price_meta=meta, dividends=io.BytesIO(dividends) if dividends else None,
        ticker=ticker, current_price=current_price, filings_text=[InputBDocument(n, t) for n, t in docs], source_notes=["Inputs A/C: files uploaded in the GUI."],
    )
    return run_review(inputs)


@st.cache_data(max_entries=6, show_spinner=False)
def workbook_bytes(_review, key: str) -> bytes:
    return build_workbook(_review)


# ------------------------------------------------------------------ sidebar

def sidebar() -> tuple:
    st.sidebar.header("Inputs")
    if "tickers" not in st.session_state:
        st.session_state.tickers = sources.default_tickers()
        st.session_state.list_name = "default list"
    with st.sidebar.expander(f"Ticker list: {st.session_state.list_name} ({len(st.session_state.tickers)})"):
        upload = st.file_uploader("Load another list (txt or csv)", type=["txt", "csv"], key="list_upload")
        pasted = st.text_area("…or paste tickers", placeholder="'AAPL','MSFT'  or one per line")
        c1, c2 = st.columns(2)
        if c1.button("Use this list", use_container_width=True):
            text = upload.getvalue().decode("utf-8", errors="replace") if upload is not None else pasted
            parsed = sources.parse_ticker_list(text)
            if parsed:
                st.session_state.tickers = parsed
                st.session_state.list_name = upload.name if upload is not None else "pasted list"
                st.rerun()
            st.warning("No tickers found in that list.")
        if c2.button("Reset to default", use_container_width=True):
            st.session_state.tickers = sources.default_tickers()
            st.session_state.list_name = "default list"
            st.rerun()

    mode = st.sidebar.radio("Data source", ["Fetch (SEC + Yahoo)", "Upload files"], horizontal=True, help="Fetched data stays in memory; nothing is saved to disk.")
    picked = st.sidebar.selectbox("Company from list", st.session_state.tickers, index=0)
    typed = st.sidebar.text_input("…or ticker / CIK", value="", placeholder="e.g. MSFT or 0000789019")
    identifier = (typed or picked or "").strip()
    as_of = st.sidebar.date_input("As-of date (Input D)", value=date.today(), max_value=date.today())
    price_text = st.sidebar.text_input("Current price (Input D, optional)", value="", placeholder="blank = last Input C close on or before the as-of date")
    current_price = None
    if price_text.strip():
        try:
            current_price = float(price_text)
        except ValueError:
            st.sidebar.error("Current price must be a number.")
    docs_files = st.sidebar.file_uploader("Input B excerpts (MD&A, auditor's report…)", type=["txt", "htm", "html", "md"], accept_multiple_files=True)
    docs = tuple((f.name, f.getvalue().decode("utf-8", errors="replace")) for f in docs_files or [])

    files = None
    if mode == "Upload files":
        st.sidebar.markdown("**Input A** companyfacts JSON · **Input C** prices CSV (`date, close[, dividend, split_ratio]`)")
        facts = st.sidebar.file_uploader("Input A (.json or .json.gz)", type=["json", "gz"])
        prices = st.sidebar.file_uploader("Input C prices (.csv)", type=["csv"])
        dividends = st.sidebar.file_uploader("Dividends (.csv, ex_date, amount; optional)", type=["csv"])
        split = st.sidebar.radio("Input C split-adjusted?", ["yes", "no", "not stated"], horizontal=True)
        div = st.sidebar.radio("Input C dividend-adjusted?", ["no", "yes", "not stated"], horizontal=True)
        currency = st.sidebar.text_input("Input C price currency", value="USD")
        to_bool = {"yes": True, "no": False, "not stated": None}
        if facts is not None:
            files = (facts.getvalue(), prices.getvalue() if prices else None, dividends.getvalue() if dividends else None, to_bool[split], to_bool[div], currency.upper().strip(), typed.upper().strip())
    st.sidebar.caption("Nothing is written to disk: inputs, workbooks and CSVs live in memory and download through the buttons.")
    return mode, identifier, as_of, current_price, docs, files


# ------------------------------------------------------------------ chart

def theme_type() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def held_series(family: str, basis: str):
    for s in SERIES:
        if s.family == family and s.basis == basis:
            return s
    return None


def charts(review, ratio_key: str, series) -> tuple[alt.LayerChart, alt.LayerChart] | None:
    """Two aligned panels (one y-scale each): the daily ratio, and the held TTM value as a step line."""
    daily = review.daily
    if daily is None:
        return None
    window = daily[daily["in_display_window"]].reset_index()
    ratio = RATIO_BY_KEY[ratio_key]
    scale = 1e6 if series.unit in ("money", "shares") else 1.0
    unit_label = {"money": f" ({review.header['currency'].split(',')[0]} millions)", "shares": " (millions)", "per_share": " (per share)"}[series.unit]
    df = pd.DataFrame(
        {
            "date": window["date"],
            "price": window["price"],
            "ratio": window[ratio_key] * (100 if ratio.kind == "yield" else 1),
            "held": pd.to_numeric(window[series.key], errors="coerce") / scale,
            "valuation_date": window["valuation_date"],
            "quarter_end": window["quarter_end"],
            "filing": window["form"].astype(str) + " " + window["accn"].astype(str),
        }
    )
    nm = window.get(f"{ratio_key}_nm", pd.Series(False, index=window.index)).astype(bool)
    df["ratio_text"] = ["NM" if flag else ("" if pd.isna(v) else (f"{v:,.2f}%" if ratio.kind == "yield" else f"{v:,.2f}")) for v, flag in zip(df["ratio"], nm)]
    df["held_text"] = [("" if pd.isna(v) else f"{v:,.2f}") for v in df["held"]]
    lo = review.display_start if review.display_start is not None else df["date"].min()
    domain = [pd.Timestamp(lo).isoformat(), pd.Timestamp(review.header["as_of"]).isoformat()]
    vdates = pd.DataFrame({"valuation_date": sorted(d for d in set(df["valuation_date"].dropna()) if d >= pd.Timestamp(lo))})
    colors = PALETTE["dark" if theme_type() == "dark" else "light"]
    tooltip = [
        alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
        alt.Tooltip("price:Q", title="Price", format=",.2f"),
        alt.Tooltip("ratio_text:N", title=ratio.label),
        alt.Tooltip("held_text:N", title=f"Held {series.label}{unit_label}"),
        alt.Tooltip("quarter_end:T", title="TTM period end", format="%Y-%m-%d"),
        alt.Tooltip("valuation_date:T", title="Valuation date", format="%Y-%m-%d"),
        alt.Tooltip("filing:N", title="Filing (form, accession)"),
    ]
    x = alt.X("date:T", title=None, scale=alt.Scale(domain=domain), axis=alt.Axis(format="%Y", tickCount="year", grid=False, labelColor=colors["ink"], tickColor=colors["rule"], domainColor=colors["rule"]))
    base = alt.Chart(df).encode(x=x)
    rules = alt.Chart(vdates).mark_rule(color=colors["rule"], strokeWidth=1, strokeDash=[2, 3], opacity=0.5).encode(x=alt.X("valuation_date:T", scale=alt.Scale(domain=domain)))
    # Equal y-axis extents keep the two panels' plot areas aligned on the shared date axis.
    yaxis = dict(grid=True, gridOpacity=0.25, labelColor=colors["ink"], titleColor=colors["ink"], domain=False, ticks=False, minExtent=64, maxExtent=64)
    ratio_title = ratio.label + (" (%)" if ratio.kind == "yield" else "")

    def panel(field, color, title, height, step):
        hover = alt.selection_point(nearest=True, on="pointermove", fields=["date"], empty=False, clear="pointerout")
        y = alt.Y(f"{field}:Q", title=ratio_title if field == "ratio" else f"{series.label}{unit_label}", scale=alt.Scale(zero=False), axis=alt.Axis(**yaxis))
        line = base.mark_line(color=color, strokeWidth=2, invalid=None, **({"interpolate": "step-after"} if step else {})).encode(y=y)
        hits = base.mark_rule(strokeWidth=10, opacity=0).encode(tooltip=tooltip).add_params(hover)
        cross = base.mark_rule(color=colors["ink"], strokeWidth=1).encode(opacity=alt.condition(hover, alt.value(0.6), alt.value(0)))
        dot = base.mark_point(filled=True, size=70, color=color, stroke="white", strokeWidth=2).encode(y=f"{field}:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)))
        return (
            alt.layer(rules, line, cross, dot, hits)
            .properties(height=height, width="container", title=alt.TitleParams(title, anchor="start", color=colors["ink"], fontSize=13))
            .configure_view(strokeWidth=0)
        )

    top = panel("ratio", colors["ratio"], f"Daily {ratio_title} = {ratio.formula}", 300, False)
    bottom = panel("held", colors["held"], f"Held {series.label} [{series.basis}]: flat between valuation dates (dashed lines)", 220, True)
    return top, bottom


def stats_panel(review, family: str, version: str, ratio_key: str):
    st.markdown(f"**Statistics — {FAMILY_LABELS[family]} — {version}** (quarter-end valuation dates, full history)")
    stats = review.stats.set_index("series")
    cols = st.columns(2)
    for col, basis in zip(cols, ("TTM", "Per share")):
        s = held_series(family, basis)
        with col:
            if s is None or s.key not in stats.index:
                st.caption(f"No {basis.lower()} series for this metric.")
                continue
            r = stats.loc[s.key]
            p = "raw" if version == "RAW" else "adj"
            st.markdown(f"*{s.label} [{basis}]*")
            items = [
                ("N", r[f"{p}_n"]), ("Comparisons", r[f"{p}_comparisons"]), ("Increases", r[f"{p}_increases"]),
                ("Increase rate", fmt_value(r[f"{p}_increase_rate"], "pct") or "n/a"), ("Record highs", r[f"{p}_record_highs"]),
                ("Latest is record high", r[f"{p}_latest_record"]),
            ]
            if p == "raw":
                items.append(("Latest rank (1 = highest)", int(r["raw_latest_rank"]) if _num(r["raw_latest_rank"]) else ""))
            else:
                items += [("Excluded observations (≤ 0)", r["adj_excluded"]), ("Pairs skipped", r["adj_pairs_skipped"]), ("Sign changes", r["adj_sign_changes"])]
            items += [("Latest value", fmt_value(r["latest"], s.unit) + (" m" if s.unit in ("money", "shares") and _num(r["latest"]) else "")), ("% change from prior", fmt_value(r["pct_change"], "pct") if _num(r["pct_change"]) else str(r["pct_change"] or ""))]
            st.dataframe(pd.DataFrame(items, columns=["Statistic", "Value"]).astype(str), hide_index=True, use_container_width=True)
    daily_n = review.daily_counts.get(ratio_key)
    st.caption(f"Daily observations with a valid {RATIO_BY_KEY[ratio_key].label} in the display window: {daily_n if daily_n is not None else 'n/a'} (separate from the quarter-end N). Money in millions.")


def daily_csv(review) -> bytes:
    daily = review.daily
    window = daily[daily["in_display_window"]].copy()
    out = pd.DataFrame({"date": window.index.date, "price": window["price"], "raw_close": window["raw_close"], "valuation_date": window["valuation_date"].dt.date,
                        "ttm_period_end": window["quarter_end"].dt.date, "form": window["form"], "accession": window["accn"], "market_cap": window["mcap"], "ev": window["ev"]})
    for r in DAILY_RATIOS:
        values = window[r.key].astype(object)
        values[window.get(f"{r.key}_nm", pd.Series(False, index=window.index)).astype(bool)] = "NM"
        out[r.label] = values.to_numpy()
    for s in SERIES:
        out[f"held {s.label} [{s.basis}]"] = window[s.key].to_numpy() if s.key in window else None
    return out.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ page

def show(review):
    h = review.header
    st.title(f"{h['company']} ({h['ticker'] or 'no ticker'})")
    st.caption(
        f"CIK {h['cik']} · {h['taxonomy']} / {h['currency']} · latest filing {h['latest_filing']} · as of {h['as_of'].date()} · display window {h['display_window']} · "
        f"statistics from {h['full_history_start'].date() if h['full_history_start'] is not None else 'n/a'} · prices {h['price_range']} ({h['price_basis']}) · splits: {h['splits']}"
    )
    if review.thin:
        st.error(f"Coverage too thin for a trend review: {review.thin_reason}")
        st.dataframe(review.coverage, use_container_width=True)
        return
    tabs = st.tabs(["Chart & statistics", "Snapshot", "Annual", "Coverage", "Data quality", "Red flags", "Downloads"])
    with tabs[0]:
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1, 1.1])
        labels = {r.key: r.label for r in DAILY_RATIOS}
        ratio_key = c1.selectbox("Ratio", list(labels), format_func=labels.get, index=0)
        families = [f for f in FAMILY_LABELS if held_series(f, "TTM") is not None]
        family = c2.selectbox("TTM metric", families, format_func=FAMILY_LABELS.get, index=families.index("earnings"))
        basis = c3.radio("Step-line basis", ["TTM", "Per share"], horizontal=True, index=1)
        version = c4.radio("Statistics", ["RAW", "ADJUSTED"], horizontal=True)
        series = held_series(family, basis) or held_series(family, "TTM")
        if review.daily is None:
            st.warning(f"No daily series: {review.price_block_reason or 'no price data (Input C)'}.")
        else:
            top, bottom = charts(review, ratio_key, series)
            st.altair_chart(top, use_container_width=True)
            st.altair_chart(bottom, use_container_width=True)
            st.download_button("Download daily table (CSV)", daily_csv(review), file_name=f"{h['ticker'] or h['cik']}_daily_{h['as_of'].date()}.csv", mime="text/csv")
        stats_panel(review, family, version, ratio_key)
    with tabs[1]:
        s = review.snapshot.copy()
        s["value"] = [fmt_value(v, u if u != "value" else "per_share") if u != "money" else fmt_value(v, "money") + " m" if _num(v) else (v if isinstance(v, str) else "") for v, u in zip(s["value"], s["unit"])]
        s["percentile"] = [(f"{p:.1f}" if p >= 1 else f"{p:.2f}") if _num(p) else "" for p in s["percentile"]]
        s["valuation_date"] = [str(pd.Timestamp(d).date()) if d is not None and not pd.isna(d) else "" for d in s["valuation_date"]]
        order = {"Price ratios": 0, "Balance sheet / market": 1, "TTM / per-share TTM": 2}
        s = s.assign(_o=s["group"].map(order)).sort_values("_o", kind="stable").drop(columns="_o")
        st.dataframe(s[["group", "item", "value", "percentile", "valuation_date", "formula", "inputs", "shares", "reason"]].astype(str), hide_index=True, use_container_width=True, height=36 * (len(s) + 1) + 4)
        st.caption("Percentile = share of trading days in the full daily history with a ratio at or below the current value. NM = zero or negative denominator; blank = missing input (reason shown).")
    with tabs[2]:
        annual = review.annual_all
        window = annual[annual["fy_start"] >= review.display_start] if review.display_start is not None else annual
        table = pd.DataFrame({str(e.date()): [fmt_value(row[k], u) for k, _, u in ANNUAL_ROWS] for e, (_, row) in zip(window["fy_end"], window.iterrows())}, index=[label for _, label, _ in ANNUAL_ROWS])
        st.dataframe(table, use_container_width=True)
        st.caption(f"Millions of {h['currency']} except per-share values. Growth (fiscal-year CAGRs):")
        st.dataframe(review.growth.astype(str), hide_index=True, use_container_width=True)
    with tabs[3]:
        st.dataframe(review.coverage, use_container_width=True)
        st.dataframe(review.coverage_summary, hide_index=True, use_container_width=True)
    with tabs[4]:
        dq = review.dq.frame()
        cats = st.multiselect("Categories", sorted(dq["category"].unique()), default=[])
        st.dataframe(dq[dq["category"].isin(cats)] if cats else dq, hide_index=True, use_container_width=True)
    with tabs[5]:
        st.dataframe(review.red_flags, hide_index=True, use_container_width=True)
    with tabs[6]:
        stem = f"{h['ticker'] or h['cik']}_{h['as_of'].date()}"
        key = f"{h['cik']}-{h['as_of'].date()}-{id(review)}"
        st.download_button("Excel workbook (.xlsx, live formulas)", workbook_bytes(review, key), file_name=f"{stem}_valuation.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Report (.md, sections 1–10)", build_report(review).encode("utf-8"), file_name=f"{stem}_report.md", mime="text/markdown")
        if review.daily is not None:
            st.download_button("Daily table (.csv)", daily_csv(review), file_name=f"{stem}_daily.csv", mime="text/csv")
        st.caption(f"Files are generated in memory when you click; your browser decides where they are saved. Launch command: `{LAUNCH}`")


def main():
    mode, identifier, as_of, current_price, docs, files = sidebar()
    try:
        if mode == "Upload files":
            if files is None:
                st.info("Upload Input A (companyfacts JSON) in the sidebar to start. Input C (prices) is optional: without it, price-based ratios are marked 'no price data'.")
                return
            with st.spinner("Running the review…"):
                review = review_from_files(*files, as_of, current_price, docs)
        else:
            if not identifier:
                st.info("Pick a company from the list or type a ticker / CIK.")
                return
            with st.spinner(f"Fetching SEC companyfacts and Yahoo prices for {identifier} (in memory) and running the review…"):
                review = review_from_sources(identifier, as_of, current_price, docs)
    except KeyError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # show the failure instead of a stack trace
        st.error(f"Review failed for {identifier}: {type(exc).__name__}: {exc}")
        return
    show(review)


main()

"""Markdown report in the order the review specifies (sections 1-10)."""

from __future__ import annotations

import math

import pandas as pd

from .ratios import RATIO_BY_KEY
from .stats import SERIES

LAUNCH = "python -B -m streamlit run valuation_app.py --server.headless true --browser.gatherUsageStats false"


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def fmt_value(v, unit: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, str):
        return v
    if unit == "money":
        return f"{v / 1e6:,.1f}"
    if unit == "shares":
        return f"{v / 1e6:,.1f}"
    if unit in ("per_share", "value"):
        return f"{v:,.2f}"
    if unit == "multiple":
        return f"{v:,.1f}x"
    if unit in ("yield", "ratio", "pct"):
        return f"{v * 100:,.2f}%"
    if unit == "count":
        return f"{int(v)}"
    return f"{v:,.4g}"


def _d(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return str(pd.Timestamp(x).date())
    except (ValueError, TypeError):
        return str(x)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    esc = lambda s: str(s).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


ANNUAL_ROWS = [
    ("revenue", "Revenue", "money"), ("gross_profit", "Gross profit", "money"), ("operating_income", "Operating income", "money"),
    ("net_income", "Net income (parent)", "money"), ("eps_diluted", "Diluted EPS (reported, split-normalized)", "per_share"),
    ("diluted_shares", "Diluted shares (millions)", "shares"), ("ocf", "Operating cash flow", "money"), ("capex", "Capex", "money"),
    ("fcf", "FCF (OCF - capex)", "money"), ("da", "D&A", "money"), ("dps_inputc", "Dividends per share (Input C ex-dates)", "per_share"),
    ("cash", "Cash", "money"), ("total_assets", "Total assets", "money"), ("total_liabilities", "Total liabilities", "money"),
    ("equity", "Parent equity", "money"), ("long_term_debt", "Long-term debt", "money"),
]
TTM_FOR_ANNUAL = {"revenue": "ttm_revenue", "gross_profit": "ttm_gross_profit", "operating_income": "ttm_operating_income", "net_income": "ttm_net_income",
                  "eps_diluted": "eps_ttm", "diluted_shares": "avg_diluted_shares", "ocf": "ttm_ocf", "capex": "ttm_capex", "fcf": "ttm_fcf", "da": "ttm_da",
                  "dps_inputc": "dps_ttm", "cash": "cash", "total_assets": "total_assets", "total_liabilities": "total_liabilities", "equity": "equity", "long_term_debt": "long_term_debt"}


def build_report(review, workbook_path: str | None = None, full_dq: bool = True) -> str:
    h = review.header
    snap = review.snapshot_inputs
    parts: list[str] = []
    title = f"{h['company']} ({h['ticker'] or 'no ticker'})"
    parts.append(f"# Valuation and fundamentals review: {title}\n")

    # 1. Header
    parts.append("## 1. Header\n")
    parts.append(_table(["Field", "Value"], [
        ["Company", h["company"]], ["Ticker", h["ticker"] or ""], ["CIK", h["cik"]],
        ["Taxonomy and currency", f"{h['taxonomy']}; {h['currency']}" + (f" (regimes: {', '.join(h['regimes'])})" if len(h["regimes"]) > 1 else "")],
        ["Latest filing", h["latest_filing"]], ["As-of date", _d(h["as_of"])], ["Display window", h["display_window"]],
        ["Full-history start (statistics)", _d(h["full_history_start"])], ["Price data range", h["price_range"]], ["Price adjustment basis", h["price_basis"]],
        ["Splits applied", h["splits"]],
    ]))
    if review.thin:
        parts.append(f"\n**Coverage too thin for a trend review.** {review.thin_reason} The review stops after the coverage check.\n")

    # 2. Coverage
    parts.append("\n## 2. Coverage matrix\n")
    parts.append("Fiscal years are labeled by the calendar year of the fiscal year end. A blank cell means no usable value.\n")
    cov = review.coverage
    if not cov.empty:
        parts.append(_table(["Item"] + list(cov.columns), [[idx] + list(row) for idx, row in cov.iterrows()]))
        cs = review.coverage_summary
        parts.append("\n" + _table(["Item", "First", "Last", "Years"], [[r["item"], r["first"], r["last"], str(r["years"])] for _, r in cs.iterrows()]))
    if review.thin:
        parts.append(_dq_section(review, full_dq))
        return "\n".join(parts) + "\n"

    # 3. Annual table
    parts.append("\n## 3. Annual table (display window)\n")
    parts.append(f"Amounts in millions of {h['currency']} except per-share values. Columns are fiscal year ends; the last column is the latest TTM.\n")
    annual = review.annual_all
    if not annual.empty and review.display_start is not None:
        window = annual[annual["fy_start"] >= review.display_start]
        latest = review.ttm.iloc[-1] if not review.ttm.empty else None
        headers = ["Item"] + [f"{_d(e)}{' (53w)' if w == '53-week' else ''}" for e, w in zip(window["fy_end"], window["weeks"])]
        add_ttm = latest is not None and (window.empty or latest["quarter_end"] > window["fy_end"].max())
        if add_ttm:
            headers.append(f"TTM {_d(latest['quarter_end'])}")
        rows = []
        for key, label, unit in ANNUAL_ROWS:
            row = [label] + [fmt_value(v, unit) for v in window[key]]
            if add_ttm:
                row.append(fmt_value(latest.get(TTM_FOR_ANNUAL[key]), unit))
            rows.append(row)
        parts.append(_table(headers, rows))

    # 4. Snapshot
    parts.append("\n## 4. Current snapshot\n")
    price = snap.get("price")
    parts.append(
        f"Price {fmt_value(price, 'per_share')} on {_d(snap.get('price_date'))} ({snap.get('price_source') or snap.get('price_reason', '')}); "
        f"shares outstanding {fmt_value(snap.get('shares_out'), 'shares')} million (cover page {_d(snap.get('cover_date'))}, {snap.get('cover_accn', '')}); "
        f"market cap {fmt_value(snap.get('mcap'), 'money')} million; EV {fmt_value(snap.get('ev'), 'money')} million "
        f"(borrowings {fmt_value(snap.get('total_borrowings'), 'money')}, NCI {fmt_value(snap.get('nci'), 'money') or 'not tagged'}, preferred {fmt_value(snap.get('preferred'), 'money') or 'not tagged'}, "
        f"cash {fmt_value(snap.get('cash'), 'money')}; balance sheet {_d(snap.get('bs_instant'))}, {snap.get('bs_accn', '')}). "
        "Lease liabilities and short-term investments are excluded from EV.\n"
    )
    s = review.snapshot
    if not s.empty:
        ttm_rows = s[s["group"] != "Price ratios"]
        parts.append(_table(["TTM / per-share item", "Value", "Valuation date", "Inputs"], [[r["item"], fmt_value(r["value"], r["unit"]) if r["unit"] != "money" else fmt_value(r["value"], "money") + " m", _d(r["valuation_date"]), r["inputs"] or r["reason"] or ""] for _, r in ttm_rows.iterrows()]))
        rows = []
        for _, r in s[s["group"] == "Price ratios"].iterrows():
            pct = review.percentiles.get(r["key"])
            pct_text = f"{pct['percentile']:.1f}" if pct and pct["percentile"] >= 1 else (f"{pct['percentile']:.2f}" if pct else "")
            pct_text += f" (n={pct['n']:,}, {_d(pct['from'])} to {_d(pct['to'])})" if pct else ""
            value = fmt_value(r["value"], r["unit"] if r["unit"] != "value" else "per_share")
            if _num(r["value"]) and r["unit"] == "multiple" and abs(r["value"]) > 100:
                value += " (>100x: check inputs)"
            rows.append([r["item"], value or "", r["formula"], r["shares"], pct_text, r["reason"] or ""])
        parts.append("\n" + _table(["Ratio", "Value", "Formula", "Share count used", "Percentile in full daily history", "Blank / NM reason"], rows))

    # 5. Statistics
    parts.append("\n## 5. TTM statistics (focus series)\n")
    parts.append(f"Observations are quarter-end valuation dates from {_d(review.ttm.iloc[review.stats_start_index]['valuation_date']) if not review.ttm.empty else ''} to {_d(h['as_of'])}.\n")
    st = review.stats
    if not st.empty:
        rows = []
        for _, r in st[st["focus"]].iterrows():
            rows.append([
                f"{r['label']} [{r['basis']}]", str(r["raw_n"]), str(r["raw_increases"]), fmt_value(r["raw_increase_rate"], "pct"), str(r["raw_record_highs"]), r["raw_latest_record"],
                str(r["adj_n"]), str(r["adj_increases"]), fmt_value(r["adj_increase_rate"], "pct"), str(r["adj_record_highs"]), r["adj_latest_record"],
                str(r["adj_excluded"]), str(r["adj_pairs_skipped"]), str(r["adj_sign_changes"]),
            ])
        parts.append(_table(["Series", "RAW N", "RAW incr.", "RAW incr. rate", "RAW records", "RAW latest record", "ADJ N", "ADJ incr.", "ADJ incr. rate", "ADJ records", "ADJ latest record", "Excluded (≤0)", "Pairs skipped", "Sign changes"], rows))

    # 6. Growth
    parts.append("\n## 6. Growth summary (fiscal-year CAGRs)\n")
    g = review.growth
    if not g.empty:
        rows = []
        for _, r in g.iterrows():
            cells = [r["metric"]]
            for n in (5, 10, 15):
                v = r.get(f"cagr_{n}")
                note = r.get(f"note_{n}") or ""
                cells.append((fmt_value(v, "pct") if _num(v) else "") + (f" ({note})" if note else ""))
            rows.append(cells)
        parts.append(_table(["Metric", "5-year", "10-year", "15-year"], rows))
        parts.append(f"\nEnd year: fiscal year ending {g.iloc[0]['to']}. Diluted shares and EPS are on the as-of split basis.")
        if len(h["regimes"]) > 1:
            parts.append(" **Comparability break:** the reporting regime changed during the history (see the data-quality log).")
        parts.append("\n")

    parts.append(_dq_section(review, full_dq))

    # 8. Red flags
    parts.append("\n## 8. Red flags\n")
    rf = review.red_flags
    if rf.empty:
        parts.append("None found by the screens run.\n")
    else:
        parts.append(_table(["Flag", "Severity", "Evidence", "Source"], [[r["flag"], r["severity"], r["evidence"], r["source"]] for _, r in rf.iterrows()]))

    # 9. Deliverables
    parts.append("\n## 9. Deliverables\n")
    parts.append(f"- Excel workbook: {workbook_path or 'built in memory (download from the GUI, or pass an output folder off the C: drive to the batch runner)'}\n")
    parts.append(f"- GUI: `{LAUNCH}` then open http://localhost:8501\n")

    # 10. Summary
    parts.append("\n## 10. Summary\n")
    parts.append(summary_text(review) + "\n")
    return "\n".join(parts)


def _dq_section(review, full: bool) -> str:
    dq = review.dq.frame()
    out = ["\n## 7. Data-quality log\n"]
    if dq.empty:
        out.append("No entries.\n")
        return "\n".join(out)
    counts = dq.groupby("category").size()
    out.append("Entries by category: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".\n")
    shown = dq if full else dq[dq["severity"].isin(["warning", "serious", "critical"]) & ~dq["category"].isin(["derived_quarter"])].head(60)
    out.append(_table(["Category", "Severity", "Item", "Period", "Detail", "Filing"], [[r["category"], r["severity"], r["item"], r["period"], r["detail"], r["filing"]] for _, r in shown.iterrows()]))
    return "\n".join(out)


def summary_text(review) -> str:
    """At most six factual sentences drawn from the tables above. No forecasts or recommendations."""
    sentences: list[str] = []
    st = review.stats.set_index("series") if not review.stats.empty else pd.DataFrame()

    def stat(key, field):
        return st.at[key, field] if key in st.index else None

    if "ttm_revenue" in st.index and stat("ttm_revenue", "raw_comparisons"):
        rec = stat("ttm_revenue", "raw_latest_record")
        sentences.append(
            f"TTM revenue rose in {stat('ttm_revenue', 'raw_increases')} of {stat('ttm_revenue', 'raw_comparisons')} consecutive valuation-date comparisons "
            f"({fmt_value(stat('ttm_revenue', 'raw_increase_rate'), 'pct')}), and the latest value {'is' if rec == 'yes' else 'is not'} a full-history record high."
        )
    if "eps_ttm" in st.index and stat("eps_ttm", "raw_n"):
        latest, pct = stat("eps_ttm", "latest"), stat("eps_ttm", "pct_change")
        sentences.append(
            f"TTM diluted EPS is {fmt_value(latest, 'per_share')}"
            + (f", {'up' if pct >= 0 else 'down'} {abs(pct) * 100:.1f}% from the prior valuation date" if _num(pct) else "")
            + f"; ADJUSTED EPS rose in {stat('eps_ttm', 'adj_increases')} of {stat('eps_ttm', 'adj_comparisons')} comparisons with {stat('eps_ttm', 'adj_excluded')} non-positive observation(s) excluded."
        )
    snap = review.snapshot.set_index("key") if not review.snapshot.empty else pd.DataFrame()
    for key in ("pe", "ev_ebitda"):
        if key in snap.index and _num(snap.at[key, "value"]):
            pct = review.percentiles.get(key)
            sentences.append(
                f"At the as-of price, {RATIO_BY_KEY[key].label} is {fmt_value(snap.at[key, 'value'], 'multiple')}"
                + (f", the {pct['percentile']:.0f}th percentile of its daily history since {_d(pct['from'])}." if pct else ".")
            )
        elif key in snap.index and snap.at[key, "value"] == "NM":
            sentences.append(f"{RATIO_BY_KEY[key].label} is NM because its denominator is zero or negative.")
    if "ttm_fcf" in st.index and stat("ttm_fcf", "raw_n"):
        sentences.append(f"TTM free cash flow is {fmt_value(stat('ttm_fcf', 'latest'), 'money')} million, with {stat('ttm_fcf', 'adj_excluded')} non-positive TTM observation(s) in the full history.")
    rf = review.red_flags
    serious = rf[rf["severity"].isin(["serious", "critical"])] if not rf.empty else rf
    dq = review.dq.frame()
    warn = int(dq["severity"].isin(["warning", "serious", "critical"]).sum()) if not dq.empty else 0
    if not serious.empty:
        sentences.append(f"Screens raised {len(serious)} serious or critical flag(s) ({', '.join(sorted(set(serious['flag'])))}) and the data-quality log holds {warn} warning(s).")
    else:
        sentences.append(f"No serious or critical red flags were raised by the data screens; the data-quality log holds {warn} warning(s).")
    return " ".join(sentences[:6])

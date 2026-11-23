"""Excel workbook with live formulas, built in memory.

Sheets (required): Snapshot, TTM_Quarterly, TTM_Stats, Ratios_Quarterly,
Valuation_Inputs, Data_Quality. Two supporting sheets keep every number
auditable: TTM_Build (the reported components behind each TTM value) and
Stats_Helpers (running counts and record-high flags used by TTM_Stats).
Daily values are never written here.

Colour convention: blue = value taken from Input A/C/D, black = formula,
green = link to another sheet.
"""

from __future__ import annotations

import io
import math

import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .ratios import RATIOS
from .stats import SERIES

FONT = "Arial"
BLUE = Font(name=FONT, size=10, color="0000FF")
BLACK = Font(name=FONT, size=10, color="000000")
GREEN = Font(name=FONT, size=10, color="008000")
BOLD = Font(name=FONT, size=10, bold=True)
TITLE = Font(name=FONT, size=12, bold=True)
HEAD_FILL = PatternFill("solid", fgColor="DDE4EE")
NOTE = Font(name=FONT, size=9, italic=True, color="555555")
EXTREME_FILL = PatternFill("solid", fgColor="FCE4D6")

FMT = {
    "money": "#,##0;(#,##0);0",
    "shares": "#,##0",
    "per_share": "0.00;(0.00);0.00",
    "multiple": '0.0"x";(0.0"x");0.0"x"',
    "yield": "0.00%;(0.00%);0.00%",
    "ratio": "0.0%;(0.0%);0.0%",
    "pct": "0.0%;(0.0%);0.0%",
    "date": "yyyy-mm-dd",
    "count": "0",
    "factor": "0.######",
    "value": "0.00",
}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _val(v):
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.to_pydatetime()
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _q(sheet: str) -> str:
    return f"'{sheet}'"


class Sheet:
    def __init__(self, ws):
        self.ws = ws

    def header(self, row: int, labels: list[str], widths: list[int] | None = None):
        for c, label in enumerate(labels, start=1):
            cell = self.ws.cell(row=row, column=c, value=label)
            cell.font = BOLD
            cell.fill = HEAD_FILL
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        if widths:
            for c, w in enumerate(widths, start=1):
                self.ws.column_dimensions[get_column_letter(c)].width = w

    def put(self, row: int, col: int, value, fmt: str | None = None, font: Font = BLUE):
        cell = self.ws.cell(row=row, column=col, value=_val(value))
        if isinstance(value, str) and value.startswith("="):
            cell.font = GREEN if "!" in value and value.count("!") == 1 and value[1] == "'" and "(" not in value else BLACK
        else:
            cell.font = font
        if fmt:
            cell.number_format = FMT.get(fmt, fmt)
        return cell


def build_workbook(review) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    names = ["Snapshot", "TTM_Quarterly", "TTM_Stats", "Ratios_Quarterly", "Valuation_Inputs", "Data_Quality", "TTM_Build", "Stats_Helpers"]
    sheets = {n: Sheet(wb.create_sheet(n)) for n in names}
    ttm = review.ttm.reset_index(drop=True)
    n = len(ttm)
    first, last = 2, n + 1  # data rows on the per-valuation-date sheets

    build_rows = _write_build(sheets["TTM_Build"], review)
    vi_cols, div_cols = _write_valuation_inputs(sheets["Valuation_Inputs"], review)
    tq_cols = _write_ttm_quarterly(sheets["TTM_Quarterly"], review, build_rows, vi_cols, div_cols)
    _write_ratios(sheets["Ratios_Quarterly"], review, tq_cols, vi_cols)
    helper_cols = _write_helpers(sheets["Stats_Helpers"], review, tq_cols)
    _write_stats(sheets["TTM_Stats"], review, tq_cols, helper_cols)
    _write_snapshot(sheets["Snapshot"], review, tq_cols, last)
    _write_dq(sheets["Data_Quality"], review)
    del first
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------- TTM_Build

BUILD_METRICS = ["revenue", "gross_profit", "operating_income", "net_income", "ocf", "capex", "da", "dividends_paid", "diluted_shares"]


def _write_build(sh: Sheet, review) -> dict[tuple[int, str], int]:
    ws = sh.ws
    ws["A1"] = "TTM_Build: reported components (Input A, latest restated, split-normalized) and the live formula for each TTM value"
    ws["A1"].font = TITLE
    sh.header(2, ["Quarter end", "Valuation date", "Metric", "Method", "Component A", "Component B", "Component C", "Component D", "TTM value",
                  "Alt: sum of 4 quarters", "Alt gap", "A is", "B is", "C is", "D is", "Source filings", "Reason if blank"],
              [12, 12, 16, 13, 18, 18, 18, 18, 18, 18, 9, 26, 26, 26, 26, 50, 40])
    ws["A3"] = "Methods: FY = annual value; FY+YTD-PYTD = A + B - C; 4Q = A + B + C + D; avg4Q = AVERAGE(A:D) (shares); blank = not computable (see reason)."
    ws["A3"].font = NOTE
    rows: dict[tuple[int, str], int] = {}
    build = review.build
    if build.empty:
        return rows
    ttm = review.ttm.reset_index(drop=True)
    index_of = {(ttm.at[i, "valuation_date"], ttm.at[i, "quarter_end"]): i for i in range(len(ttm))}
    r = 4
    for metric in BUILD_METRICS:
        sub = build[build["metric"] == metric]
        for _, b in sub.iterrows():
            i = index_of.get((b["valuation_date"], b["quarter_end"]))
            if i is None:
                continue
            rows[(i, metric)] = r
            unit = "shares" if metric == "diluted_shares" else "money"
            sh.put(r, 1, b["quarter_end"], "date")
            sh.put(r, 2, b["valuation_date"], "date")
            sh.put(r, 3, metric, font=BLACK)
            sh.put(r, 4, b["method"] or "", font=BLACK)
            for c, key in enumerate(["a", "b", "c", "d"], start=5):
                if _num(b[key]):
                    sh.put(r, c, float(b[key]), unit)
            method = b["method"]
            formula = {
                "FY": f"=E{r}",
                "FY+YTD-PYTD": f"=E{r}+F{r}-G{r}",
                "4Q": f"=E{r}+F{r}+G{r}+H{r}",
                "avg4Q": f"=AVERAGE(E{r}:H{r})",
            }.get(method)
            if formula:
                sh.put(r, 9, formula, unit)
            if _num(b["alt"]):
                sh.put(r, 10, float(b["alt"]), unit)
                sh.put(r, 11, f'=IF(AND(ISNUMBER(I{r}),ISNUMBER(J{r}),I{r}<>0),(I{r}-J{r})/ABS(I{r}),"")', "pct")
            for c, key in enumerate(["a_label", "b_label", "c_label", "d_label"], start=12):
                sh.put(r, c, b[key] or "", font=BLACK)
            sh.put(r, 16, b["sources"] or "", font=BLACK)
            sh.put(r, 17, b["reason"] or "", font=BLACK)
            r += 1
    ws.freeze_panes = "E4"
    return rows


# ---------------------------------------------------------------- Valuation_Inputs

VI_COLUMNS = [
    ("valuation_date", "Valuation date", "date"), ("quarter_end", "Quarter end", "date"), ("form", "Form", None), ("accn", "Accession", None),
    ("price_date", "Price date", "date"), ("raw_price", "Input C close (as supplied)", "per_share"), ("price_factor", "Split factor applied to close", "factor"),
    ("price", "Price (as-of split basis)", "per_share"), ("price_reason", "Price note", None),
    ("cover_shares_reported", "Cover-page shares (reported)", "shares"), ("cover_date", "Cover date", "date"), ("cover_accn", "Cover filing", None),
    ("share_factor", "Share split factor", "factor"), ("shares_out", "Shares outstanding (as-of basis)", "shares"),
    ("bs_instant", "Balance sheet date", "date"), ("bs_accn", "Balance sheet filing", None),
    ("cash", "Cash & equivalents", "money"), ("debt_current", "Borrowings, current", "money"), ("debt_noncurrent", "Borrowings, noncurrent", "money"),
    ("total_borrowings", "Total borrowings", "money"), ("nci", "Noncontrolling interest", "money"), ("preferred", "Preferred stock", "money"),
    ("equity", "Parent equity", "money"), ("goodwill", "Goodwill", "money"), ("intangibles", "Intangibles excl. goodwill", "money"), ("tbv", "Tangible book value", "money"),
    ("total_assets", "Total assets", "money"), ("total_liabilities", "Total liabilities", "money"), ("long_term_debt", "Long-term debt", "money"),
    ("mcap", "Market cap", "money"), ("ev", "Enterprise value", "money"), ("debt_tags", "Borrowing tags used", None), ("vi_notes", "Notes", None),
]


def _write_valuation_inputs(sh: Sheet, review) -> tuple[dict[str, str], tuple[str, str]]:
    ws = sh.ws
    cols = {key: get_column_letter(i) for i, (key, _, _) in enumerate(VI_COLUMNS, start=1)}
    sh.header(1, [label for _, label, _ in VI_COLUMNS], [12, 12, 8, 22, 12, 14, 12, 14, 30, 16, 12, 22, 10, 16, 12, 22] + [17] * 15 + [40, 50])
    ttm = review.ttm.reset_index(drop=True)
    c = cols
    for i, row in ttm.iterrows():
        r = i + 2
        for j, (key, _, fmt) in enumerate(VI_COLUMNS, start=1):
            if key == "price" and _num(row.get("raw_price")):
                sh.put(r, j, f'=IF(AND(ISNUMBER({c["raw_price"]}{r}),ISNUMBER({c["price_factor"]}{r})),{c["raw_price"]}{r}/{c["price_factor"]}{r},"")', fmt)
            elif key == "shares_out" and _num(row.get("cover_shares_reported")):
                sh.put(r, j, f'=IF(ISNUMBER({c["cover_shares_reported"]}{r}),{c["cover_shares_reported"]}{r}*{c["share_factor"]}{r},"")', fmt)
            elif key == "total_borrowings":
                sh.put(r, j, f'=IF(COUNT({c["debt_current"]}{r},{c["debt_noncurrent"]}{r})=0,"",SUM({c["debt_current"]}{r},{c["debt_noncurrent"]}{r}))', fmt)
            elif key == "tbv":
                sh.put(r, j, f'=IF(ISNUMBER({c["equity"]}{r}),IF(COUNT({c["goodwill"]}{r},{c["intangibles"]}{r})=0,"",{c["equity"]}{r}-SUM({c["goodwill"]}{r},{c["intangibles"]}{r})),"")', fmt)
            elif key == "mcap":
                sh.put(r, j, f'=IF(AND(ISNUMBER({c["price"]}{r}),ISNUMBER({c["shares_out"]}{r})),{c["price"]}{r}*{c["shares_out"]}{r},"")', fmt)
            elif key == "ev":
                sh.put(r, j, f'=IF(AND(ISNUMBER({c["mcap"]}{r}),ISNUMBER({c["cash"]}{r})),{c["mcap"]}{r}+SUM({c["total_borrowings"]}{r},{c["nci"]}{r},{c["preferred"]}{r})-{c["cash"]}{r},"")', fmt)
            elif key == "debt_tags":
                sh.put(r, j, "; ".join(t for t in (f"current: {row.get('debt_current_tag')}" if row.get("debt_current_tag") else "", f"noncurrent: {row.get('debt_noncurrent_tag')}" if row.get("debt_noncurrent_tag") else "", row.get("borrowings_note") or "") if t), font=BLACK)
            elif key == "vi_notes":
                notes = [x for x in (row.get("tbv_reason"), row.get("ev_reason"), row.get("shares_source") if "fallback" in str(row.get("shares_source")) else "") if x]
                sh.put(r, j, "; ".join(notes), font=BLACK)
            elif key in ("form", "accn", "price_reason", "cover_accn", "bs_accn"):
                sh.put(r, j, row.get(key) or "", font=BLACK)
            else:
                v = row.get(key)
                sh.put(r, j, v if (_num(v) or isinstance(v, pd.Timestamp)) else None, fmt)
    # Dividend events (Input C) for the SUMIFS behind TTM dividends per share.
    start_col = len(VI_COLUMNS) + 2
    ex_col, amt_col = get_column_letter(start_col), get_column_letter(start_col + 1)
    ws.cell(row=1, column=start_col, value="Dividend ex-date (Input C)").font = BOLD
    ws.cell(row=1, column=start_col + 1, value="Dividend per share (as-of split basis)").font = BOLD
    ws.cell(row=1, column=start_col).fill = HEAD_FILL
    ws.cell(row=1, column=start_col + 1).fill = HEAD_FILL
    ws.column_dimensions[ex_col].width = 14
    ws.column_dimensions[amt_col].width = 16
    prices = review.prices
    if prices is not None and prices.dividends is not None and not ttm.empty:
        lo = ttm["quarter_end"].min() - pd.DateOffset(years=1, days=7)
        d = prices.dividends[prices.dividends["ex_date"] >= lo]
        for k, (_, ev) in enumerate(d.iterrows(), start=2):
            sh.put(k, start_col, ev["ex_date"], "date")
            sh.put(k, start_col + 1, float(ev["amount"]), "0.0000")
    ws.freeze_panes = "E2"
    return cols, (ex_col, amt_col)


# ---------------------------------------------------------------- TTM_Quarterly

def _write_ttm_quarterly(sh: Sheet, review, build_rows, vi, div_cols) -> dict[str, str]:
    ws = sh.ws
    ttm = review.ttm.reset_index(drop=True)
    lead = [("quarter_end", "Quarter end (TTM period end)", "date"), ("fiscal_quarter", "Fiscal quarter", None), ("valuation_date", "Valuation date (first filing date)", "date"),
            ("form", "Form", None), ("accn", "Accession", None), ("regime", "Taxonomy / currency", None), ("in_stats", "In statistics", None)]
    labels = [label for _, label, _ in lead]
    cols: dict[str, str] = {}
    col = len(lead) + 1
    for s in SERIES:
        cols[s.key] = get_column_letter(col)
        cols[s.key + "__elig"] = get_column_letter(col + 1)
        unit = {"money": "", "shares": " (shares)", "per_share": " (per share)"}[s.unit]
        labels += [f"{s.label} [{s.basis}]{unit}", f"{s.label}: eligible for ADJUSTED (> 0)"]
        col += 2
    extra = [("eps_reported_ttm", "Reported EPS cross-check (FY+YTD-PYTD)", "per_share"), ("eps_check_gap", "EPS cross-check gap", "pct"),
             ("eps_cagr5", "5-year TTM EPS CAGR", "pct"), ("shares_method", "Diluted shares method", None), ("notes", "Missing-value reasons", None)]
    for key, label, _ in extra:
        cols[key] = get_column_letter(col)
        labels.append(label)
        col += 1
    sh.header(1, labels, [12, 8, 12, 8, 22, 14, 9] + [16, 9] * len(SERIES) + [14, 10, 10, 14, 60])
    c = cols
    ex, amt = div_cols
    for i, row in ttm.iterrows():
        r = i + 2
        sh.put(r, 1, row["quarter_end"], "date")
        sh.put(r, 2, row["fiscal_quarter"], font=BLACK)
        sh.put(r, 3, row["valuation_date"], "date")
        sh.put(r, 4, row["form"], font=BLACK)
        sh.put(r, 5, row["accn"], font=BLACK)
        sh.put(r, 6, row["regime"], font=BLACK)
        sh.put(r, 7, "Y" if i >= review.stats_start_index else "N (before regime break)", font=BLACK)
        shares = f"{c['avg_diluted_shares']}{r}"
        for s in SERIES:
            cell = f"{c[s.key]}{r}"
            key = s.key
            f = None
            metric = {"ttm_revenue": "revenue", "ttm_gross_profit": "gross_profit", "ttm_operating_income": "operating_income", "ttm_net_income": "net_income",
                      "ttm_ocf": "ocf", "ttm_capex": "capex", "ttm_da": "da", "ttm_dividends_paid": "dividends_paid", "avg_diluted_shares": "diluted_shares"}.get(key)
            if metric is not None:
                br = build_rows.get((i, metric))
                if br is not None and _num(row.get(key)):
                    f = f"={_q('TTM_Build')}!I{br}"
            elif key == "ttm_fcf":
                f = f'=IF(AND(ISNUMBER({c["ttm_ocf"]}{r}),ISNUMBER({c["ttm_capex"]}{r})),{c["ttm_ocf"]}{r}-{c["ttm_capex"]}{r},"")'
            elif key == "ttm_ebitda":
                f = f'=IF(AND(ISNUMBER({c["ttm_operating_income"]}{r}),ISNUMBER({c["ttm_da"]}{r})),{c["ttm_operating_income"]}{r}+{c["ttm_da"]}{r},"")'
            elif key == "equity":
                f = f'=IF(ISNUMBER({_q("Valuation_Inputs")}!{vi["equity"]}{r}),{_q("Valuation_Inputs")}!{vi["equity"]}{r},"")'
            elif key == "bvps":
                so = f"{_q('Valuation_Inputs')}!{vi['shares_out']}{r}"
                f = f'=IF(AND(ISNUMBER({c["equity"]}{r}),ISNUMBER({so})),IF({so}>0,{c["equity"]}{r}/{so},""),"")'
            elif key == "dps_ttm":
                if _num(row.get("dps_ttm")):
                    rng_d = f"{_q('Valuation_Inputs')}!${ex}:${ex}"
                    rng_a = f"{_q('Valuation_Inputs')}!${amt}:${amt}"
                    f = f'=SUMIFS({rng_a},{rng_d},">"&EDATE(A{r},-12),{rng_d},"<="&A{r})'
            else:
                src = {"revenue_ps": "ttm_revenue", "gp_ps": "ttm_gross_profit", "eps_ttm": "ttm_net_income", "ocf_ps": "ttm_ocf", "fcf_ps": "ttm_fcf",
                       "da_ps": "ttm_da", "ebitda_ps": "ttm_ebitda", "oi_ps": "ttm_operating_income"}[key]
                f = f'=IF(AND(ISNUMBER({c[src]}{r}),ISNUMBER({shares})),IF({shares}>0,{c[src]}{r}/{shares},""),"")'
            if f is not None:
                sh.put(r, ws[cell].column, f, s.unit)
            sh.put(r, ws[cell].column + 1, f'=IF(ISNUMBER({cell}),IF({cell}>0,"Y","N"),"")', font=BLACK)
        if _num(row.get("eps_reported_ttm")):
            sh.put(r, ws[f"{c['eps_reported_ttm']}1"].column, float(row["eps_reported_ttm"]), "per_share")
            sh.put(r, ws[f"{c['eps_check_gap']}1"].column, f'=IF(AND(ISNUMBER({c["eps_ttm"]}{r}),ISNUMBER({c["eps_reported_ttm"]}{r})),IF({c["eps_reported_ttm"]}{r}=0,"",ABS({c["eps_ttm"]}{r}-{c["eps_reported_ttm"]}{r})/ABS({c["eps_reported_ttm"]}{r})),"")', "pct")
        base = row.get("eps_cagr5_base_row")
        if _num(base):
            b = int(base) + 2
            e = c["eps_ttm"]
            sh.put(r, ws[f"{c['eps_cagr5']}1"].column, f'=IF(AND(ISNUMBER({e}{r}),ISNUMBER({e}{b})),IF(AND({e}{r}>0,{e}{b}>0),({e}{r}/{e}{b})^(1/5)-1,""),"")', "pct")
        sh.put(r, ws[f"{c['shares_method']}1"].column, row.get("shares_method") or "", font=BLACK)
        sh.put(r, ws[f"{c['notes']}1"].column, row.get("notes") or "", font=BLACK)
    ws.freeze_panes = "H2"
    return cols


# ---------------------------------------------------------------- Ratios_Quarterly

def ratio_formula(ratio, refs: dict[str, str]) -> str:
    """Excel formula for a ratio. `refs` maps input names to cell references."""
    if ratio.key == "graham":
        e, b = refs["eps"], refs["bvps"]
        return f'=IF(AND(ISNUMBER({e}),ISNUMBER({b})),IF(AND({e}>0,{b}>0),SQRT(22.5*{e}*{b}),"NM"),"")'
    if ratio.key == "peg":
        pe, g = refs["pe"], refs["cagr"]
        return f'=IF(AND(ISNUMBER({g}),{g}>0),IF(ISNUMBER({pe}),{pe}/({g}*100),IF({pe}="NM","NM","")),"")'
    n, d = refs[ratio.numerator], refs[ratio.denominator]
    return f'=IF(AND(ISNUMBER({n}),ISNUMBER({d})),IF({d}<=0,"NM",{n}/{d}),"")'


def _write_ratios(sh: Sheet, review, tq, vi):
    ws = sh.ws
    labels = ["Valuation date", "Quarter end", "Price", "Market cap", "Enterprise value"] + [f"{r.label}" for r in RATIOS] + ["Missing-input reasons"]
    sh.header(1, labels, [12, 12, 12, 18, 18] + [12] * len(RATIOS) + [70])
    ws.cell(row=1, column=1).comment = Comment("Formulas: " + " | ".join(f"{r.label} = {r.formula}" for r in RATIOS), "review")
    ttm = review.ttm.reset_index(drop=True)
    ratio_col = {r.key: get_column_letter(6 + k) for k, r in enumerate(RATIOS)}
    V, T = _q("Valuation_Inputs"), _q("TTM_Quarterly")
    from .pipeline import ratio_inputs
    from .ratios import compute_all

    for i, row in ttm.iterrows():
        r = i + 2
        sh.put(r, 1, f"={T}!C{r}", "date")
        sh.put(r, 2, f"={T}!A{r}", "date")
        sh.put(r, 3, f'=IF(ISNUMBER({V}!{vi["price"]}{r}),{V}!{vi["price"]}{r},"")', "per_share")
        sh.put(r, 4, f'=IF(ISNUMBER({V}!{vi["mcap"]}{r}),{V}!{vi["mcap"]}{r},"")', "money")
        sh.put(r, 5, f'=IF(ISNUMBER({V}!{vi["ev"]}{r}),{V}!{vi["ev"]}{r},"")', "money")
        refs = {
            "price": f"C{r}", "mcap": f"D{r}", "ev": f"E{r}",
            "eps": f"{T}!{tq['eps_ttm']}{r}", "revenue": f"{T}!{tq['ttm_revenue']}{r}", "gross_profit": f"{T}!{tq['ttm_gross_profit']}{r}",
            "equity": f"{T}!{tq['equity']}{r}", "tbv": f"{V}!{vi['tbv']}{r}", "ocf": f"{T}!{tq['ttm_ocf']}{r}", "fcf": f"{T}!{tq['ttm_fcf']}{r}",
            "dps": f"{T}!{tq['dps_ttm']}{r}", "operating_income": f"{T}!{tq['ttm_operating_income']}{r}", "ebitda": f"{T}!{tq['ttm_ebitda']}{r}",
            "bvps": f"{T}!{tq['bvps']}{r}", "pe": f"{ratio_col['pe']}{r}", "cagr": f"{T}!{tq['eps_cagr5']}{r}",
        }
        for k, ratio in enumerate(RATIOS):
            fmt = {"multiple": "multiple", "yield": "yield", "ratio": "ratio", "value": "per_share"}[ratio.kind]
            sh.put(r, 6 + k, ratio_formula(ratio, refs), fmt)
        values = compute_all(ratio_inputs(row))
        reasons = [f"{rd.label}: {values[rd.key][1]}" for rd in RATIOS if values[rd.key][0] is None and values[rd.key][1]]
        sh.put(r, 6 + len(RATIOS), "; ".join(reasons), font=BLACK)
    last = len(ttm) + 1
    for ratio in RATIOS:
        if ratio.kind == "multiple" and last >= 2:
            rng = f"{ratio_col[ratio.key]}2:{ratio_col[ratio.key]}{last}"
            ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["100"], fill=EXTREME_FILL))
            ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["-100"], fill=EXTREME_FILL))
    note_row = last + 2
    ws.cell(row=note_row, column=1, value="Shaded multiples exceed 100x in absolute value: check inputs. NM = zero or negative denominator. Blank = input missing (reason in the last column).").font = NOTE
    ws.freeze_panes = "F2"


# ---------------------------------------------------------------- Stats_Helpers

def _write_helpers(sh: Sheet, review, tq) -> dict[str, tuple[str, str, str]]:
    ws = sh.ws
    ttm = review.ttm.reset_index(drop=True)
    s0 = review.stats_start_index + 2
    last = len(ttm) + 1
    T = _q("TTM_Quarterly")
    ws.cell(row=1, column=1, value="Valuation date").font = BOLD
    cols: dict[str, tuple[str, str, str]] = {}
    col = 2
    for s in SERIES:
        names = (get_column_letter(col), get_column_letter(col + 1), get_column_letter(col + 2))
        cols[s.key] = names
        for k, label in enumerate(["running count", "RAW record high", "ADJ record high"]):
            cell = ws.cell(row=1, column=col + k, value=f"{s.label} [{s.basis}]: {label}")
            cell.font = BOLD
            cell.fill = HEAD_FILL
            cell.alignment = Alignment(wrap_text=True)
        col += 3
    for r in range(s0, last + 1):
        sh.put(r, 1, f"={T}!C{r}", "date")
        for s in SERIES:
            x = tq[s.key]
            cnt, raw, adj = cols[s.key]
            sh.put(r, ws[f"{cnt}1"].column, f"=COUNT({T}!{x}${s0}:{x}{r})", "count")
            if r == s0:
                sh.put(r, ws[f"{raw}1"].column, "=0", "count")
                sh.put(r, ws[f"{adj}1"].column, "=0", "count")
            else:
                prev = f"{T}!{x}${s0}:{x}{r - 1}"
                cur = f"{T}!{x}{r}"
                sh.put(r, ws[f"{raw}1"].column, f"=IF(AND(ISNUMBER({cur}),COUNT({prev})>0),IF({cur}>MAX({prev}),1,0),0)", "count")
                sh.put(r, ws[f"{adj}1"].column, f'=IF(ISNUMBER({cur}),IF(AND({cur}>0,COUNTIF({prev},">0")>0),IF({cur}>MAX({prev}),1,0),0),0)', "count")
    ws.freeze_panes = "B2"
    return cols


# ---------------------------------------------------------------- TTM_Stats

STAT_COLUMNS = [
    ("series", "Series"), ("basis", "Basis"), ("focus", "Focus"), ("start", "Statistics start (valuation date)"),
    ("raw_n", "RAW N"), ("raw_comparisons", "RAW comparisons"), ("raw_increases", "RAW increases"), ("raw_decreases", "RAW decreases"), ("raw_unchanged", "RAW unchanged"),
    ("raw_increase_rate", "RAW increase rate"), ("raw_record_highs", "RAW record highs"), ("raw_latest_record", "RAW latest is record high"), ("raw_latest_rank", "RAW latest rank (1 = highest)"),
    ("adj_n", "ADJ N"), ("adj_comparisons", "ADJ comparisons"), ("adj_pairs_skipped", "ADJ pairs skipped"), ("adj_increases", "ADJ increases"), ("adj_decreases", "ADJ decreases"),
    ("adj_unchanged", "ADJ unchanged"), ("adj_increase_rate", "ADJ increase rate"), ("adj_record_highs", "ADJ record highs"), ("adj_latest_record", "ADJ latest is record high"),
    ("adj_excluded", "ADJ excluded observations (≤ 0)"), ("adj_sign_changes", "Sign changes"),
    ("latest", "Latest value"), ("latest_date", "Latest valuation date"), ("prior", "Prior value"), ("prior_date", "Prior valuation date"), ("pct_change", "% change from prior"),
    ("min", "Full-history minimum"), ("min_date", "Minimum date"), ("max", "Full-history maximum"), ("max_date", "Maximum date"),
    ("daily_obs", "Daily observations in display window (linked ratio)"), ("daily_ratio", "Linked daily ratio"),
]


def _write_stats(sh: Sheet, review, tq, helpers):
    ws = sh.ws
    ttm = review.ttm.reset_index(drop=True)
    s0 = review.stats_start_index + 2
    last = len(ttm) + 1
    T, H = _q("TTM_Quarterly"), _q("Stats_Helpers")
    ws["A1"] = "TTM statistics over quarter-end valuation dates (full history from the statistics start). RAW and ADJUSTED side by side."
    ws["A1"].font = TITLE
    ws["A2"] = ("RAW counts every numeric value (a move from -2 to -1 is an increase). ADJUSTED keeps values > 0 only; pairs with an excluded side are 'pairs skipped'. "
                "Record highs exclude the first (reference) observation. Daily observation counts are computed in Python from Input C (daily values are not stored in this workbook).")
    ws["A2"].font = NOTE
    sh.header(4, [label for _, label in STAT_COLUMNS], [34, 9, 7, 13] + [10] * 20 + [16, 12, 16, 12, 10, 16, 12, 16, 12, 12, 12])
    col = {key: get_column_letter(i) for i, (key, _) in enumerate(STAT_COLUMNS, start=1)}
    stats = review.stats.set_index("series") if not review.stats.empty else pd.DataFrame()
    for k, s in enumerate(SERIES):
        r = 5 + k
        x = tq[s.key]
        rng, rp, rn = f"{T}!${x}${s0}:${x}${last}", f"{T}!${x}${s0}:${x}${last - 1}", f"{T}!${x}${s0 + 1}:${x}${last}"
        dates = f"{T}!$C${s0}:$C${last}"
        cnt, raw_h, adj_h = (f"{H}!${c}${s0}:${c}${last}" for c in helpers[s.key])
        C = lambda key: f"{col[key]}{r}"  # noqa: E731
        single = last <= s0
        pair = "=0" if single else None
        both = f"ISNUMBER({rp})*ISNUMBER({rn})"
        adjp = f"{both}*({rp}>0)*({rn}>0)"
        fmt = s.unit
        sh.put(r, 1, f"{s.label}", font=BOLD if s.focus else BLACK)
        sh.put(r, 2, s.basis, font=BLACK)
        sh.put(r, 3, "yes" if s.focus else "", font=BLACK)
        sh.put(r, 4, f"={T}!C{s0}", "date")
        formulas = {
            "raw_n": f"=COUNT({rng})",
            "raw_comparisons": pair or f"=SUMPRODUCT({both})",
            "raw_increases": pair or f"=SUMPRODUCT({both}*({rn}>{rp}))",
            "raw_decreases": pair or f"=SUMPRODUCT({both}*({rn}<{rp}))",
            "raw_unchanged": pair or f"=SUMPRODUCT({both}*({rn}={rp}))",
            "raw_increase_rate": f'=IF({C("raw_comparisons")}>0,{C("raw_increases")}/{C("raw_comparisons")},"")',
            "raw_record_highs": f"=SUM({raw_h})",
            "raw_latest_record": f'=IF({C("raw_n")}<1,"n/a (no values)",IF({C("raw_n")}=1,"n/a (single value)",IF(INDEX({raw_h},MATCH({C("raw_n")},{cnt},0))=1,"yes","no")))',
            "raw_latest_rank": f'=IF({C("raw_n")}>0,1+COUNTIF({rng},">"&{C("latest")}),"")',
            "adj_n": f'=COUNTIF({rng},">0")',
            "adj_comparisons": pair or f"=SUMPRODUCT({adjp})",
            "adj_pairs_skipped": f'={C("raw_comparisons")}-{C("adj_comparisons")}',
            "adj_increases": pair or f"=SUMPRODUCT({adjp}*({rn}>{rp}))",
            "adj_decreases": pair or f"=SUMPRODUCT({adjp}*({rn}<{rp}))",
            "adj_unchanged": pair or f"=SUMPRODUCT({adjp}*({rn}={rp}))",
            "adj_increase_rate": f'=IF({C("adj_comparisons")}>0,{C("adj_increases")}/{C("adj_comparisons")},"")',
            "adj_record_highs": f"=SUM({adj_h})",
            "adj_latest_record": f'=IF({C("raw_n")}<1,"n/a (no values)",IF({C("latest")}<=0,"excluded (≤ 0)",IF({C("adj_n")}<=1,"n/a (single value)",IF(INDEX({adj_h},MATCH({C("raw_n")},{cnt},0))=1,"yes","no"))))',
            "adj_excluded": f'=COUNTIF({rng},"<=0")',
            "adj_sign_changes": pair or f"=SUMPRODUCT({both}*((({rp}>0)*({rn}<0))+(({rp}<0)*({rn}>0))))",
            "latest": f'=IF({C("raw_n")}>0,INDEX({rng},MATCH({C("raw_n")},{cnt},0)),"")',
            "latest_date": f'=IF({C("raw_n")}>0,INDEX({dates},MATCH({C("raw_n")},{cnt},0)),"")',
            "prior": f'=IF({C("raw_n")}>1,INDEX({rng},MATCH({C("raw_n")}-1,{cnt},0)),"")',
            "prior_date": f'=IF({C("raw_n")}>1,INDEX({dates},MATCH({C("raw_n")}-1,{cnt},0)),"")',
            "pct_change": f'=IF(AND(ISNUMBER({C("latest")}),ISNUMBER({C("prior")})),IF({C("prior")}=0,"NM",({C("latest")}-{C("prior")})/ABS({C("prior")})),"")',
            "min": f'=IF({C("raw_n")}>0,MIN({rng}),"")',
            "min_date": f'=IF({C("raw_n")}>0,INDEX({dates},MATCH({C("min")},{rng},0)),"")',
            "max": f'=IF({C("raw_n")}>0,MAX({rng}),"")',
            "max_date": f'=IF({C("raw_n")}>0,INDEX({dates},MATCH({C("max")},{rng},0)),"")',
        }
        fmts = {"raw_increase_rate": "pct", "adj_increase_rate": "pct", "pct_change": "pct", "latest": fmt, "prior": fmt, "min": fmt, "max": fmt,
                "latest_date": "date", "prior_date": "date", "min_date": "date", "max_date": "date"}
        for key, formula in formulas.items():
            sh.put(r, ws[f"{col[key]}1"].column, formula, fmts.get(key, "count"))
        daily = stats.at[s.key, "daily_obs"] if s.key in stats.index else None
        sh.put(r, ws[f"{col['daily_obs']}1"].column, int(daily) if _num(daily) else None, "count")
        sh.put(r, ws[f"{col['daily_ratio']}1"].column, s.daily_ratio or "", font=BLACK)
    ws.freeze_panes = "B5"


# ---------------------------------------------------------------- Snapshot

def _write_snapshot(sh: Sheet, review, tq, last_row: int):
    ws = sh.ws
    h, snap = review.header, review.snapshot_inputs
    ws["A1"] = f"{h['company']} ({h['ticker'] or 'no ticker'}) — CIK {h['cik']} — valuation snapshot as of {snap['as_of'].date()}"
    ws["A1"].font = TITLE
    ws["A2"] = f"Latest filing: {h['latest_filing']}. Taxonomy {h['taxonomy']}; currency {h['currency']}. Prices: {h['price_range']} ({h['price_basis']}). Splits: {h['splits']}."
    ws["A2"].font = NOTE
    ws.column_dimensions["A"].width = 40
    for c, w in zip("BCDEFGH", [20, 14, 60, 40, 36, 14, 50]):
        ws.column_dimensions[c].width = w
    T = _q("TTM_Quarterly")
    # ---- inputs block
    sh.header(4, ["Input", "Value", "Date", "Source / note"])
    inputs = [
        ("price", "Price (as-of split basis)", snap.get("price"), "per_share", snap.get("price_date"), snap.get("price_source") or snap.get("price_reason", "")),
        ("shares", "Shares outstanding (as-of split basis)", snap.get("shares_out"), "shares", snap.get("cover_date"), f"{snap.get('shares_source', '')}; {snap.get('cover_form', '')} {snap.get('cover_accn', '')}; split factor {snap.get('share_factor', '')}"),
        ("cash", "Cash & equivalents", snap.get("cash"), "money", snap.get("bs_instant"), f"{snap.get('bs_form', '')} {snap.get('bs_accn', '')}"),
        ("debt", "Total borrowings (current + noncurrent, as tagged)", snap.get("total_borrowings"), "money", snap.get("bs_instant"), snap.get("borrowings_note") or f"current: {snap.get('debt_current_tag') or '-'}; noncurrent: {snap.get('debt_noncurrent_tag') or '-'}"),
        ("nci", "Noncontrolling interest", snap.get("nci"), "money", snap.get("bs_instant"), "" if snap.get("nci") is not None else "not tagged (excluded from EV)"),
        ("pref", "Preferred stock", snap.get("preferred"), "money", snap.get("bs_instant"), "" if snap.get("preferred") is not None else "not tagged (excluded from EV)"),
        ("equity", "Parent equity", snap.get("equity"), "money", snap.get("bs_instant"), ""),
        ("goodwill", "Goodwill", snap.get("goodwill"), "money", snap.get("bs_instant"), "" if snap.get("goodwill") is not None else "not tagged"),
        ("intang", "Intangibles excl. goodwill", snap.get("intangibles"), "money", snap.get("bs_instant"), "" if snap.get("intangibles") is not None else "not tagged"),
    ]
    ref: dict[str, str] = {}
    r = 5
    for key, label, value, fmt, when, note in inputs:
        sh.put(r, 1, label, font=BLACK)
        sh.put(r, 2, value if _num(value) else None, fmt)
        sh.put(r, 3, when, "date")
        sh.put(r, 4, note or "", font=BLACK)
        ref[key] = f"B{r}"
        r += 1
    derived = [
        ("mcap", "Market cap", f'=IF(AND(ISNUMBER({ref["price"]}),ISNUMBER({ref["shares"]})),{ref["price"]}*{ref["shares"]},"")', "money", "Price x shares outstanding (cover page)"),
        ("ev", "Enterprise value", f'=IF(AND(ISNUMBER(B{r}),ISNUMBER({ref["cash"]})),B{r}+SUM({ref["debt"]},{ref["nci"]},{ref["pref"]})-{ref["cash"]},"")', "money", "Market cap + borrowings + NCI + preferred - cash (untagged components excluded; lease liabilities and short-term investments excluded)"),
        ("tbv", "Tangible book value", f'=IF(ISNUMBER({ref["equity"]}),IF(COUNT({ref["goodwill"]},{ref["intang"]})=0,"",{ref["equity"]}-SUM({ref["goodwill"]},{ref["intang"]})),"")', "money", "Parent equity - goodwill - intangibles (if tagged)"),
        ("bvps", "Book value per share", f'=IF(AND(ISNUMBER({ref["equity"]}),ISNUMBER({ref["shares"]})),IF({ref["shares"]}>0,{ref["equity"]}/{ref["shares"]},""),"")', "per_share", "Parent equity / cover-page shares outstanding"),
    ]
    for key, label, formula, fmt, note in derived:
        sh.put(r, 1, label, font=BLACK)
        sh.put(r, 2, formula, fmt)
        sh.put(r, 4, note, font=BLACK)
        ref[key] = f"B{r}"
        r += 1
    # ---- TTM block (links to the latest TTM_Quarterly row)
    r += 1
    sh.header(r, ["TTM / per-share TTM value", "Value", "Valuation date", "Inputs"])
    r += 1
    ttm = review.ttm
    latest = ttm.iloc[-1] if not ttm.empty else None
    for s in SERIES:
        sh.put(r, 1, f"{s.label} [{s.basis}]", font=BLACK)
        if latest is not None:
            sh.put(r, 2, f'=IF(ISNUMBER({T}!{tq[s.key]}{last_row}),{T}!{tq[s.key]}{last_row},"")', s.unit)
            sh.put(r, 3, f"={T}!C{last_row}", "date")
            sh.put(r, 4, f"TTM ending {latest['quarter_end'].date()} ({latest['form']} {latest['accn']})", font=BLACK)
        ref["ttm_" + s.key] = f"B{r}"
        r += 1
    sh.put(r, 1, "5-year TTM EPS CAGR", font=BLACK)
    sh.put(r, 2, f'=IF(ISNUMBER({T}!{tq["eps_cagr5"]}{last_row}),{T}!{tq["eps_cagr5"]}{last_row},"")', "pct")
    sh.put(r, 4, (latest.get("eps_cagr5_reason") or "") if latest is not None else "", font=BLACK)
    ref["cagr"] = f"B{r}"
    r += 2
    # ---- ratios
    sh.header(r, ["Ratio", "Value", "Valuation date", "Formula", "Inputs", "Share count used", "Percentile (full daily history)", "Reason if blank / NM"])
    ws.cell(row=r, column=7).comment = Comment("Computed in Python from the daily series (Input C price / held fundamentals). Daily values are not stored in this workbook.", "review")
    r += 1
    refs = {
        "price": ref["price"], "mcap": ref["mcap"], "ev": ref["ev"], "eps": ref["ttm_eps_ttm"], "revenue": ref["ttm_ttm_revenue"], "gross_profit": ref["ttm_ttm_gross_profit"],
        "equity": ref["equity"], "tbv": ref["tbv"], "ocf": ref["ttm_ttm_ocf"], "fcf": ref["ttm_ttm_fcf"], "dps": ref["ttm_dps_ttm"], "operating_income": ref["ttm_ttm_operating_income"],
        "ebitda": ref["ttm_ttm_ebitda"], "bvps": ref["bvps"], "cagr": ref["cagr"],
    }
    snapshot = review.snapshot.set_index("key") if not review.snapshot.empty else pd.DataFrame()
    pe_row = r + [x.key for x in RATIOS].index("pe")
    refs["pe"] = f"B{pe_row}"
    for ratio in RATIOS:
        fmt = {"multiple": "multiple", "yield": "yield", "ratio": "ratio", "value": "per_share"}[ratio.kind]
        sh.put(r, 1, ratio.label, font=BLACK)
        sh.put(r, 2, ratio_formula(ratio, refs), fmt)
        if latest is not None:
            sh.put(r, 3, latest["valuation_date"], "date")
        sh.put(r, 4, ratio.formula, font=BLACK)
        info = snapshot.loc[ratio.key] if ratio.key in snapshot.index else None
        sh.put(r, 5, info["inputs"] if info is not None else "", font=BLACK)
        sh.put(r, 6, ratio.shares, font=BLACK)
        pct = info["percentile"] if info is not None else None
        sh.put(r, 7, pct / 100 if _num(pct) else None, "pct")
        sh.put(r, 8, (info["reason"] or "") if info is not None else "", font=BLACK)
        r += 1
    rng = f"B{r - len(RATIOS)}:B{r - 1}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThan", formula=["100"], fill=EXTREME_FILL))
    r += 1
    ws.cell(row=r, column=1, value="Blue = value from Inputs A/C/D; black = formula; green = link. Shaded multiples exceed 100x: check inputs. This sheet holds the only current-state values in the workbook.").font = NOTE


# ---------------------------------------------------------------- Data_Quality

def _write_dq(sh: Sheet, review):
    ws = sh.ws
    for c, w in zip("ABCDEF", [24, 10, 34, 12, 110, 28]):
        ws.column_dimensions[c].width = w
    r = 1
    ws.cell(row=r, column=1, value="Coverage (first and last fiscal year with usable data)").font = TITLE
    r += 1
    sh.header(r, ["Item", "First", "Last", "Years"])
    for c, w in zip("ABCDEF", [24, 10, 34, 12, 110, 28]):
        ws.column_dimensions[c].width = w
    r += 1
    for _, row in review.coverage_summary.iterrows():
        for c, key in enumerate(["item", "first", "last", "years"], start=1):
            sh.put(r, c, row[key], font=BLACK)
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="Red flags (each cited to its filing)").font = TITLE
    r += 1
    sh.header(r, ["Flag", "Severity", "Source", "", "Evidence"])
    r += 1
    for _, row in review.red_flags.iterrows():
        sh.put(r, 1, row["flag"], font=BLACK)
        sh.put(r, 2, row["severity"], font=BLACK)
        sh.put(r, 3, row["source"], font=BLACK)
        sh.put(r, 5, row["evidence"], font=BLACK)
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="Data-quality log").font = TITLE
    r += 1
    sh.header(r, ["Category", "Severity", "Item", "Period", "Detail", "Filing"])
    r += 1
    for _, row in review.dq.frame().iterrows():
        for c, key in enumerate(["category", "severity", "item", "period", "detail", "filing"], start=1):
            sh.put(r, c, row[key], font=BLACK)
        r += 1

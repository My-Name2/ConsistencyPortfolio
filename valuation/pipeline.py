"""Run the full valuation and fundamentals review for one company."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .dq import DQLog
from .facts import CompanyFacts, load_companyfacts
from .periods import Calendar, Quarter, quarterly_values, reported_ttm_eps, ttm_flow, ttm_shares, build_calendar
from .prices import PriceData, PriceMeta, apply_basis, expected_trading_days, load_prices, price_on_or_before, ttm_dividends
from .ratios import DAILY_RATIOS, NM, RATIOS, compute_all, is_extreme
from .redflags import InputBDocument, data_flags, text_flags
from .splits import SplitBasis, detect_splits
from .stats import SERIES, stats_table
from .tags import BALANCE_ANCHOR_METRICS, CALENDAR_METRICS, COVER_SHARES_TAG, DEBT_TAGS, DEI, IFRS, METRICS, US, MetricDef
from .values import MetricTable, build_metric_table, derive_gross_profit, lessor_revenue, parent_equity, parent_share_of_profit_loss

FLOW_KEYS = ["revenue", "gross_profit", "operating_income", "net_income", "ocf", "capex", "da", "dividends_paid", "impairment", "disc_ops", "disposal_gains"]
TTM_COLUMNS = {
    "revenue": "ttm_revenue",
    "gross_profit": "ttm_gross_profit",
    "operating_income": "ttm_operating_income",
    "net_income": "ttm_net_income",
    "ocf": "ttm_ocf",
    "capex": "ttm_capex",
    "da": "ttm_da",
    "dividends_paid": "ttm_dividends_paid",
}
INSTANT_KEYS = ["cash", "total_assets", "total_liabilities", "equity", "long_term_debt", "goodwill", "intangibles", "nci", "preferred", "bs_shares"]
DISPLAY_YEARS = 15
EPS_CHECK_TOL = 0.05
THIN_MIN_YEARS = 3
THIN_MIN_TTM = 8
COVER_CLASS_RATIO = 0.85  # cover-page shares below this share of weighted basic shares cover one class only


@dataclass
class ReviewInputs:
    facts: object  # Input A: dict, path or file
    as_of: date  # Input D
    prices: object | None = None  # Input C: DataFrame, path or file (date, close[, dividend, split_ratio])
    price_meta: PriceMeta = field(default_factory=PriceMeta)
    dividends: object | None = None
    filings_text: list[InputBDocument] = field(default_factory=list)  # Input B
    current_price: float | None = None  # Input D price, if given
    ticker: str = ""
    source_notes: list[str] = field(default_factory=list)  # provenance of Inputs A and C


@dataclass
class Review:
    inputs: ReviewInputs
    cf: CompanyFacts
    dq: DQLog
    calendar: Calendar
    tables: dict[str, MetricTable]
    prices: PriceData | None
    basis: SplitBasis
    ttm: pd.DataFrame
    build: pd.DataFrame
    annual_all: pd.DataFrame
    coverage: pd.DataFrame
    coverage_summary: pd.DataFrame
    stats: pd.DataFrame
    stats_start_index: int
    snapshot: pd.DataFrame
    snapshot_inputs: dict
    growth: pd.DataFrame
    red_flags: pd.DataFrame
    header: dict
    thin: bool
    thin_reason: str
    display_start: pd.Timestamp | None
    price_block_reason: str
    filings: pd.DataFrame
    amendments: pd.DataFrame
    latest_one_time: dict
    daily: pd.DataFrame | None = None
    percentiles: dict = field(default_factory=dict)
    daily_counts: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

def _num(v) -> bool:
    return isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


def _slot_metric(key: str, taxonomy_tags: dict[str, tuple[str, ...]]) -> MetricDef:
    return MetricDef(key, key.replace("_", " "), "instant", "money", False, taxonomy_tags, {})


class BalanceSheet:
    """Balance sheet and share count from the most recent filing on or before a date."""

    def __init__(self, cf: CompanyFacts, tables: dict[str, MetricTable], debt: dict[str, MetricTable], basis: SplitBasis, dq: DQLog):
        self.cf, self.tables, self.debt, self.basis, self.dq = cf, tables, debt, basis, dq
        anchor_tags = set()
        for key in BALANCE_ANCHOR_METRICS:
            for taxonomy, tags in METRICS[key].candidates.items():
                anchor_tags |= {(taxonomy, t) for t in tags}
        f = cf.facts
        keys = f["taxonomy"] + ":" + f["tag"]
        mask = (f["kind"] == "instant") & keys.isin({f"{t}:{g}" for t, g in anchor_tags})
        anchors = f[mask & (f["end"] <= f["filed"])]
        self.anchor = anchors.groupby("accn").agg(filed=("filed", "min"), instant=("end", "max"), form=("form", "first")).reset_index().sort_values(["filed", "accn"])
        cover = f[(f["taxonomy"] == DEI) & (f["tag"] == COVER_SHARES_TAG) & (f["unit"] == "shares")]
        self.cover = cover.sort_values(["filed", "accn"])
        self._logged_multi: set[str] = set()
        self._logged_partial: set[str] = set()
        self._logged_fallback = False

    def at(self, when: pd.Timestamp) -> dict:
        out: dict = {}
        cand = self.anchor[self.anchor["filed"] <= when]
        if cand.empty:
            out["bs_reason"] = "no balance sheet filed on or before this date"
            return out
        last = cand.iloc[-1]
        instant = last["instant"]
        out.update(bs_instant=instant, bs_accn=last["accn"], bs_form=last["form"], bs_filed=last["filed"])
        for key in INSTANT_KEYS:
            row = self.tables[key].get(None, instant)
            out[key] = None if row is None else float(row["value"])
            out[f"{key}_tag"] = "" if row is None else row["tag"]
            out[f"{key}_accn"] = "" if row is None else row["accn"]
        out.update(self._borrowings(instant))
        gw, ia, eq = out.get("goodwill"), out.get("intangibles"), out.get("equity")
        if eq is None:
            out["tbv"], out["tbv_reason"] = None, "parent equity not tagged"
        elif gw is None and ia is None:
            out["tbv"], out["tbv_reason"] = None, "goodwill and intangibles not tagged"
        else:
            out["tbv"] = eq - (gw or 0.0) - (ia or 0.0)
            out["tbv_reason"] = "" if (gw is not None and ia is not None) else ("intangibles not tagged" if ia is None else "goodwill not tagged") + " (treated as not present)"
        out.update(self._cover_shares(when, instant))
        return out

    def _borrowings(self, instant) -> dict:
        def get(slot):
            row = self.debt[slot].get(None, instant)
            return (None, "") if row is None else (float(row["value"]), row["tag"])

        dct, dct_tag = get("debt_current_total")
        ltdc, ltdc_tag = get("ltd_current")
        st, st_tag = get("short_term")
        ltdn, ltdn_tag = get("ltd_noncurrent")
        ltdt, ltdt_tag = get("ltd_total")
        notes = []
        if ltdn is not None:
            noncurrent, nc_src = ltdn, ltdn_tag
            if dct is not None:
                current, c_src = dct, dct_tag
            elif ltdc is not None or st is not None:
                current = (ltdc or 0.0) + (st or 0.0)
                c_src = " + ".join(t for t in (ltdc_tag, st_tag) if t)
            else:
                current, c_src = None, ""
        elif ltdt is not None:
            if ltdc is not None:
                noncurrent, nc_src = ltdt - ltdc, f"{ltdt_tag} - {ltdc_tag}"
                current = dct if dct is not None else ltdc + (st or 0.0)
                c_src = dct_tag if dct is not None else " + ".join(t for t in (ltdc_tag, st_tag) if t)
            else:
                noncurrent, nc_src = ltdt, f"{ltdt_tag} (includes current maturities)"
                current, c_src = st, st_tag
                if dct is not None:
                    notes.append("DebtCurrent not added: LongTermDebt already includes current maturities and the split is not tagged")
        else:
            noncurrent, nc_src = None, ""
            if dct is not None:
                current, c_src = dct, dct_tag
            elif ltdc is not None or st is not None:
                current = (ltdc or 0.0) + (st or 0.0)
                c_src = " + ".join(t for t in (ltdc_tag, st_tag) if t)
            else:
                current, c_src = None, ""
        total = None if current is None and noncurrent is None else (current or 0.0) + (noncurrent or 0.0)
        return {
            "debt_current": current,
            "debt_current_tag": c_src,
            "debt_noncurrent": noncurrent,
            "debt_noncurrent_tag": nc_src,
            "total_borrowings": total,
            "borrowings_note": "; ".join(notes) if notes else ("no borrowings tagged at this instant" if total is None else ""),
        }

    def _cover_shares(self, when, instant) -> dict:
        cand = self.cover[self.cover["filed"] <= when]
        if not cand.empty and cand["end"].max() < instant - pd.Timedelta(days=400):
            if "stale" not in self._logged_partial:
                self._logged_partial.add("stale")
                self.dq.add(
                    "fallback_tag",
                    "Shares outstanding",
                    f"The latest cover-page share count ({cand['end'].max().date()}) is more than a year older than the balance sheet "
                    f"({instant.date()}); it is not used and market cap falls back to other share counts.",
                    period=instant,
                    severity="warning",
                )
            cand = cand.iloc[0:0]
        if not cand.empty:
            accn = cand.iloc[-1]["accn"]
            rows = cand[cand["accn"] == accn]
            rows = rows[rows["end"] == rows["end"].max()].drop_duplicates("val")
            reported = float(rows["val"].sum())
            filed = rows.iloc[0]["filed"]
            if len(rows) > 1 and accn not in self._logged_multi:
                self._logged_multi.add(accn)
                self.dq.add("assumption", "cover-page shares", f"{len(rows)} distinct cover-page share counts in one filing (likely share classes); summed to {reported:,.0f}.", period=rows.iloc[0]["end"], filing=accn, severity="warning")
            factor = self.basis.share_factor(filed)
            weighted = self._weighted_basic(instant)
            if weighted and reported * factor < COVER_CLASS_RATIO * weighted:
                if accn not in self._logged_partial:
                    self._logged_partial.add(accn)
                    self.dq.add(
                        "fallback_tag",
                        "Shares outstanding",
                        f"Cover-page shares {reported * factor:,.0f} are {reported * factor / weighted:.0%} of weighted basic shares {weighted:,.0f} for the period; "
                        "they appear to cover one share class only (or are a placeholder), so market cap uses the weighted basic count.",
                        period=rows.iloc[0]["end"],
                        filing=accn,
                        severity="warning",
                    )
                return self._fallback_shares(instant, skip_balance_sheet=True)
            return {
                "cover_shares_reported": reported,
                "cover_date": rows.iloc[0]["end"],
                "cover_accn": accn,
                "cover_form": rows.iloc[0]["form"],
                "cover_filed": filed,
                "share_factor": factor,
                "shares_out": reported * factor,
                "shares_source": "dei:EntityCommonStockSharesOutstanding",
            }
        return self._fallback_shares(instant)

    def _weighted_basic(self, instant) -> float | None:
        basic = self.tables["basic_shares"].table
        if basic.empty:
            return None
        row = self._latest_basic(instant)
        return None if row is None else float(row["value"])

    def _latest_basic(self, instant) -> dict | None:
        """Weighted basic shares for the period ending at the instant, else the latest within 200 days before it."""
        basic = self.tables["basic_shares"].table
        if basic.empty:
            return None
        near = basic[(basic["end"] <= instant + pd.Timedelta(days=3)) & (basic["end"] >= instant - pd.Timedelta(days=200))]
        if near.empty:
            return None
        latest_end = near["end"].max()
        return near[near["end"] == latest_end].sort_values("days").iloc[0].to_dict()

    def _fallback_shares(self, instant, skip_balance_sheet: bool = False) -> dict:
        row = None if skip_balance_sheet else self.tables["bs_shares"].get(None, instant)
        source = None
        if row is not None:
            source = f"{row['tag']} (balance sheet, fallback)"
        else:
            row = self._latest_basic(instant)
            if row is not None:
                source = f"{row['tag']} weighted basic, {int(row['days'])}-day period ending {row['end'].date()} (fallback)"
        if row is not None:
            if not self._logged_fallback:
                self._logged_fallback = True
                self.dq.add(
                    "fallback_tag",
                    "Shares outstanding",
                    "dei:EntityCommonStockSharesOutstanding is not available without share-class dimensions (companyfacts excludes them); "
                    f"market cap uses {source} instead. Check the share count in the workbook (Valuation_Inputs).",
                    severity="warning",
                )
            return {
                "cover_shares_reported": float(row["value"]),
                "cover_date": instant,
                "cover_accn": row["accn"],
                "cover_form": row["form"],
                "cover_filed": row["filed"],
                "share_factor": 1.0,
                "shares_out": float(row["value"]),
                "shares_source": source,
            }
        return {"shares_out": None, "shares_reason": "no shares outstanding filed on or before this date"}


# ---------------------------------------------------------------- main

def run_review(inp: ReviewInputs) -> Review:
    dq = DQLog()
    as_of = pd.Timestamp(inp.as_of).normalize()
    cf = load_companyfacts(inp.facts, inp.as_of)
    for note in inp.source_notes:
        dq.add("input", "provenance", note)
    if cf.excluded_after_asof:
        dq.add("input", "Input A", f"{cf.excluded_after_asof:,} facts filed after the as-of date {as_of.date()} were excluded (no look-ahead).")
    dq.add("input", "Input B", f"{len(inp.filings_text)} excerpt(s) supplied." if inp.filings_text else "Not supplied: going-concern, auditor-opinion, material-weakness and auditor-change screens are not assessable.")

    prices = None
    meta = inp.price_meta or PriceMeta()
    if inp.prices is not None:
        prices = load_prices(inp.prices, meta, inp.dividends, dq)
        post = prices.close_raw.index > as_of
        if post.any():
            dq.add("input", "Input C", f"{int(post.sum())} price rows after the as-of date ignored.")
        prices.close_raw = prices.close_raw[~post]
        prices.invalid_days = [d for d in prices.invalid_days if d <= as_of]
        prices.dividends_raw = prices.dividends_raw[prices.dividends_raw["ex_date"] <= as_of].reset_index(drop=True)
        if prices.close_raw.empty:
            dq.add("input", "Input C", "No prices on or before the as-of date.", severity="warning")
            prices = None
    else:
        dq.add("input", "Input C", "Not supplied: every price-based ratio is 'no price data'.", severity="warning")

    explicit = prices.splits if prices is not None else None
    events = detect_splits(cf.facts, dq, prices.close_raw if prices is not None else None, meta.split_adjusted, explicit, as_of)
    basis = SplitBasis(events, as_of)

    price_block = ""
    if prices is not None:
        apply_basis(prices, basis, dq, check_from=cf.filings["filed"].min() - pd.DateOffset(years=1))
        _restore_asof_basis(prices, as_of, dq)
        if meta.split_adjusted is None:
            if basis.events:
                price_block = "Input C does not state whether prices are split-adjusted and a split was detected"
                dq.add("price_basis", "split basis not stated", price_block + "; per-share price ratios left blank.", severity="critical")
            else:
                dq.add("price_basis", "split basis not stated", "Input C does not state its split basis; no split detected in Input A, so the basis does not change any ratio.", severity="warning")
        if meta.dividend_adjusted is None:
            dq.add("price_basis", "dividend basis not stated", "Input C does not state whether closes are dividend-adjusted; treated as traded closes.", severity="warning")

    tables = {key: build_metric_table(cf, METRICS[key], basis, dq) for key in METRICS}
    tables["revenue"] = lessor_revenue(tables["revenue"], tables["lease_income"], dq)
    tables["gross_profit"] = derive_gross_profit(tables["gross_profit"], tables["revenue"], tables["cost_of_revenue"], dq)
    tables["net_income"] = parent_share_of_profit_loss(tables["net_income"], tables["nci_income"], dq)
    tables["equity"] = parent_equity(tables["equity"], tables["nci"], dq)
    debt = {slot: build_metric_table(cf, _slot_metric(slot, {US: DEBT_TAGS[US][slot], IFRS: DEBT_TAGS[IFRS][slot]}), basis, DQLog()) for slot in DEBT_TAGS[US]}

    currencies = sorted(set(cf.filings["dominant_currency"].dropna()))
    reporting_currency = _latest_regime(cf)[1]
    if prices is not None and not price_block:
        if meta.currency and reporting_currency and meta.currency != reporting_currency:
            price_block = f"price currency {meta.currency} differs from reporting currency {reporting_currency}; no FX input supplied"
            dq.add("price_basis", "currency mismatch", price_block + ". Price-based ratios left blank.", severity="critical")
        elif not meta.currency and reporting_currency and reporting_currency != "USD":
            price_block = f"price currency not stated and reporting currency is {reporting_currency}"
            dq.add("price_basis", "currency not stated", price_block + ". Price-based ratios left blank.", severity="critical")
        elif not meta.currency:
            dq.add("price_basis", "currency not stated", "Input C currency not stated; assumed USD, the reporting currency.", severity="warning")

    calendar = build_calendar([tables[k] for k in CALENDAR_METRICS], dq, as_of)
    qv = {key: quarterly_values(tables[key], calendar, dq) for key in FLOW_KEYS}
    qv_shares = quarterly_values(tables["diluted_shares"], calendar, DQLog())
    bs = BalanceSheet(cf, tables, debt, basis, dq)

    rows, build_rows = _valuation_rows(cf, calendar, tables, qv, qv_shares, bs, prices, price_block, dq)
    ttm = pd.DataFrame(rows)
    build = pd.DataFrame(build_rows)
    if not ttm.empty:
        _add_cagr(ttm)
    stats_start = _regime_breaks(ttm, dq)
    annual_all = _annual_table(calendar, tables, prices)
    coverage, coverage_summary = _coverage(calendar, tables, cf, prices, as_of)
    filings = cf.filings
    amendments = filings[filings["is_amendment"]]
    for row in amendments.itertuples():
        dq.add("amendment", row.form, f"Amended filing {row.form} filed {row.filed.date()}; its values supersede earlier ones for the periods it reports.", period=row.filed, filing=row.accn)
    _missing_periods(calendar, ttm, dq)

    display_start = None
    fy_done = [y for y in calendar.years if y.end is not None and y.end <= as_of]
    if fy_done:
        display_start = fy_done[-DISPLAY_YEARS:][0].start

    n_years = int(annual_all["revenue"].notna().sum()) if not annual_all.empty else 0
    n_years = max(n_years, int(annual_all["net_income"].notna().sum()) if not annual_all.empty else 0)
    n_ttm = int(ttm["ttm_net_income"].notna().sum()) if not ttm.empty else 0
    thin = n_years < THIN_MIN_YEARS or n_ttm < THIN_MIN_TTM
    thin_reason = f"{n_years} fiscal year(s) with revenue or net income and {n_ttm} TTM earnings observation(s); at least {THIN_MIN_YEARS} and {THIN_MIN_TTM} are needed for a trend review." if thin else ""
    if thin:
        dq.add("coverage", "coverage too thin", thin_reason, severity="critical")

    snapshot_inputs = _snapshot_inputs(inp, ttm, bs, prices, price_block, as_of, dq)
    review = Review(
        inputs=inp, cf=cf, dq=dq, calendar=calendar, tables=tables, prices=prices, basis=basis, ttm=ttm, build=build,
        annual_all=annual_all, coverage=coverage, coverage_summary=coverage_summary, stats=pd.DataFrame(), stats_start_index=stats_start,
        snapshot=pd.DataFrame(), snapshot_inputs=snapshot_inputs, growth=_growth(annual_all, ttm), red_flags=pd.DataFrame(),
        header={}, thin=thin, thin_reason=thin_reason, display_start=display_start, price_block_reason=price_block,
        filings=filings, amendments=amendments, latest_one_time=_latest_one_time(ttm),
    )
    review.daily = daily_table(review)
    review.daily_counts = _daily_counts(review)
    review.stats = stats_table(ttm, stats_start, review.daily_counts) if not ttm.empty else pd.DataFrame()
    review.snapshot, review.percentiles = _snapshot_table(review)
    _flag_extremes(review)
    flags = data_flags(review) + text_flags(inp.filings_text)
    if not inp.filings_text:
        flags.append({"flag": "text screens not assessable", "severity": "info", "evidence": "Input B not supplied: going-concern language, auditor opinion, material weakness and auditor change could not be checked.", "source": "-"})
    review.red_flags = pd.DataFrame(flags, columns=["flag", "severity", "evidence", "source"])
    review.header = _header(review, currencies)
    return review


def _restore_asof_basis(prices: PriceData, as_of: pd.Timestamp, dq: DQLog) -> None:
    """Split-adjusted feeds are adjusted through their download date; undo splits after the as-of date."""
    if prices.meta.split_adjusted is not True or prices.splits.empty:
        return
    later = prices.splits[prices.splits["date"] > as_of]
    if later.empty:
        return
    factor = float(np.prod(later["ratio"]))
    prices.close = prices.close * factor
    if not prices.dividends.empty:
        prices.dividends["amount"] = prices.dividends["amount"] * factor
    dq.add("price_basis", "splits after as-of date", f"Input C is adjusted for split(s) after the as-of date ({', '.join(str(d.date()) for d in later['date'])}); closes multiplied by {factor:g} to restore the as-of basis.", severity="warning")


def _latest_regime(cf: CompanyFacts) -> tuple[str, str]:
    periodic = cf.filings.dropna(subset=["report_period"])
    if periodic.empty:
        return US, ""
    last = periodic.iloc[-1]
    return last["regime_taxonomy"], last["dominant_currency"] or ""


def _valuation_rows(cf, cal: Calendar, tables, qv, qv_shares, bs: BalanceSheet, prices, price_block, dq):
    rows, build = [], []
    running = None
    triggers = []
    for f in cf.filings.itertuples():
        rp = f.report_period
        if pd.isna(rp):
            continue
        if running is not None and rp <= running + pd.Timedelta(days=3):
            continue
        running = rp
        quarter = next((q for q in cal.quarters if abs(q.end - rp) <= pd.Timedelta(days=3)), None)
        if quarter is None:
            dq.add("period_label", "filing period outside fiscal calendar", f"{f.form} {f.accn} first reports a period ending {rp.date()} that is not a fiscal quarter end; no valuation date created.", period=rp, filing=f.accn)
            continue
        if triggers and triggers[-1][1] == f.filed:
            dq.add("assumption", "same-day filings", f"{triggers[-1][2]} and {f.accn} were both filed {f.filed.date()}; the later period ({quarter.label}) is the valuation row.", period=f.filed)
            triggers.pop()
        triggers.append((quarter, f.filed, f.accn, f.form, f.regime_taxonomy, f.dominant_currency))

    fy_fallback: list[pd.Timestamp] = []
    fx_rows: list[pd.Timestamp] = []
    for quarter, filed, accn, form, taxonomy, currency in triggers:
        row = {
            "quarter_end": quarter.end,
            "fiscal_quarter": f"Q{quarter.k}",
            "fy_end": cal.years[quarter.fy].end,
            "valuation_date": filed,
            "form": form,
            "accn": accn,
            "regime": f"{taxonomy} {currency}",
        }
        reasons = []
        for key in FLOW_KEYS:
            t = ttm_flow(tables[key], qv[key], cal, quarter, dq if key in TTM_COLUMNS else None)
            col = TTM_COLUMNS.get(key, f"ttm_{key}")
            row[col] = t.value
            row[f"{col}_method"] = t.method
            if t.value is None and key in TTM_COLUMNS:
                reasons.append(t.reason)
            if key in TTM_COLUMNS:
                build.append(_build_row(row, key, t))
        sh = ttm_shares(tables["diluted_shares"], qv_shares, cal, quarter)
        row["avg_diluted_shares"] = sh.value
        row["shares_method"] = sh.method
        row["shares_note"] = sh.sources or sh.reason
        build.append(_build_row(row, "diluted_shares", sh))
        if sh.value is None:
            reasons.append(f"Diluted shares: {sh.reason}")
        elif sh.method == "FY" and quarter.k != 4:
            fy_fallback.append(quarter.end)

        ni, shares = row["ttm_net_income"], row["avg_diluted_shares"]
        row["eps_ttm"] = ni / shares if _num(ni) and _num(shares) and shares > 0 else None
        reported = reported_ttm_eps(tables["eps_diluted"], cal, quarter)
        row["eps_reported_ttm"] = reported
        if _num(row["eps_ttm"]) and _num(reported) and reported != 0:
            gap = abs(row["eps_ttm"] - reported) / abs(reported)
            row["eps_check_gap"] = gap
            if gap > EPS_CHECK_TOL:
                dq.add("eps_crosscheck", "TTM diluted EPS", f"TTM ending {quarter.end.date()}: NI/avg diluted shares = {row['eps_ttm']:.4f} vs reported FY+YTD-PYTD EPS = {reported:.4f} ({gap:.1%} apart).", period=quarter.end, severity="warning")
        else:
            row["eps_check_gap"] = None

        ocf, capex = row["ttm_ocf"], row["ttm_capex"]
        row["ttm_fcf"] = ocf - capex if _num(ocf) and _num(capex) else None
        oi, da = row["ttm_operating_income"], row["ttm_da"]
        row["ttm_ebitda"] = oi + da if _num(oi) and _num(da) else None
        missing_oi = "operating income (OperatingIncomeLoss) not available for this TTM period"
        missing = [n for n, v in (("operating income", oi), ("D&A", da)) if not _num(v)]
        row["ebitda_reason"] = f"TTM EBITDA unavailable: {' and '.join(missing)} missing" if missing else ""
        row["operating_income_reason"] = missing_oi if not _num(oi) else ""
        missing = [n for n, v in (("operating cash flow", ocf), ("capex", capex)) if not _num(v)]
        row["fcf_reason"] = f"TTM FCF unavailable: {' and '.join(missing)} missing" if missing else ""

        bal = bs.at(filed)
        row.update(bal)
        if row.get("equity") is None:
            reasons.append("Parent equity: " + ("not tagged at " + str(bal["bs_instant"].date()) if "bs_instant" in bal else bal.get("bs_reason", "")))

        dps, n_div, dps_reason = ttm_dividends(prices, quarter.end)
        row["dps_ttm"], row["dps_count"], row["dps_reason"] = dps, n_div, dps_reason

        for col, src in (("revenue_ps", "ttm_revenue"), ("gp_ps", "ttm_gross_profit"), ("oi_ps", "ttm_operating_income"), ("ocf_ps", "ttm_ocf"), ("fcf_ps", "ttm_fcf"), ("da_ps", "ttm_da"), ("ebitda_ps", "ttm_ebitda")):
            v = row[src]
            row[col] = v / shares if _num(v) and _num(shares) and shares > 0 else None
        so = row.get("shares_out")
        row["bvps"] = row["equity"] / so if _num(row.get("equity")) and _num(so) and so > 0 else None

        price_currency = prices.meta.currency if prices is not None else ""
        row["currency_ok"] = not (price_currency and currency and price_currency != currency)
        if prices is None:
            lookup_reason = "no price data"
            row.update(price=None, price_date=None, raw_price=None, price_factor=None)
        elif not row["currency_ok"]:
            lookup_reason = f"fundamentals reported in {currency}, price in {price_currency}; no FX input"
            row.update(price=None, price_date=None, raw_price=None, price_factor=None)
            fx_rows.append(quarter.end)
        elif price_block:
            lookup_reason = price_block
            row.update(price=None, price_date=None, raw_price=None, price_factor=None)
        else:
            look = price_on_or_before(prices, filed, allow_stale=False)
            lookup_reason = look.reason
            row.update(price=look.close, price_date=look.date, raw_price=look.raw_close, price_factor=look.factor if look.close is not None else None)
            if look.close is None and look.reason:
                dq.add("missing_period", "price at valuation date", f"Valuation date {filed.date()} ({quarter.label}): {look.reason}; price ratios left blank.", period=filed)
        row["price_reason"] = lookup_reason
        _market_values(row)
        row["notes"] = "; ".join(r for r in reasons if r)
        rows.append(row)
    if fx_rows:
        dq.add(
            "price_basis",
            "currency mismatch by period",
            f"{len(fx_rows)} valuation date(s) (TTM ends {fx_rows[0].date()} to {fx_rows[-1].date()}) report fundamentals in a currency other than Input C's "
            f"{prices.meta.currency} prices. No FX input was supplied, so price ratios for those dates (and the daily days they cover) are blank.",
            severity="warning",
        )
    if fy_fallback:
        dq.add(
            "shares_method",
            "TTM average diluted shares",
            f"For {len(fy_fallback)} TTM period(s) ending Q1-Q3 ({fy_fallback[0].date()} to {fy_fallback[-1].date()}) the four quarterly weighted diluted counts "
            "were not all reported (a 3-month Q4 count is rarely filed and is never derived), so the latest fiscal-year count is used. "
            "TTM EPS is cross-checked against reported EPS; gaps above 5% are logged under eps_crosscheck.",
        )
    return rows, build


def _build_row(row: dict, key: str, t) -> dict:
    comps = list(t.components) + [None] * (4 - len(t.components))
    labels = list(t.component_labels) + [""] * (4 - len(t.component_labels))
    return {
        "quarter_end": row["quarter_end"],
        "valuation_date": row["valuation_date"],
        "metric": key,
        "method": t.method,
        "a": comps[0], "b": comps[1], "c": comps[2], "d": comps[3],
        "a_label": labels[0], "b_label": labels[1], "c_label": labels[2], "d_label": labels[3],
        "ttm": t.value,
        "alt": t.alt_value,
        "alt_method": t.alt_method,
        "reason": t.reason,
        "sources": t.sources,
    }


def _market_values(row: dict) -> None:
    price, so = row.get("price"), row.get("shares_out")
    row["mcap"] = price * so if _num(price) and _num(so) else None
    cash = row.get("cash")
    if _num(row["mcap"]) and _num(cash):
        row["ev"] = row["mcap"] + sum(row.get(k) or 0.0 for k in ("total_borrowings", "nci", "preferred")) - cash
        row["ev_reason"] = ""
    else:
        row["ev"] = None
        row["ev_reason"] = "market cap unavailable" if not _num(row["mcap"]) else "cash not tagged"


def _add_cagr(ttm: pd.DataFrame) -> None:
    cagr, reasons, base_idx = [], [], []
    ends = list(ttm["quarter_end"])
    for i, end in enumerate(ends):
        target = end - pd.DateOffset(years=5)
        j = next((j for j in range(i) if abs(ends[j] - target) <= pd.Timedelta(days=10)), None)
        base_idx.append(j)
        if j is None:
            cagr.append(None)
            reasons.append("no TTM EPS five years earlier")
            continue
        now, then = ttm.at[i, "eps_ttm"], ttm.at[j, "eps_ttm"]
        if not (_num(now) and _num(then)):
            cagr.append(None)
            reasons.append("TTM EPS missing at an endpoint")
        elif now <= 0 or then <= 0:
            cagr.append(None)
            reasons.append("TTM EPS not positive at an endpoint; CAGR undefined")
        else:
            cagr.append((now / then) ** (1 / 5) - 1)
            reasons.append("")
    ttm["eps_cagr5"] = cagr
    ttm["eps_cagr5_reason"] = reasons
    ttm["eps_cagr5_base_row"] = base_idx


def _regime_breaks(ttm: pd.DataFrame, dq: DQLog) -> int:
    if ttm.empty:
        return 0
    start = 0
    regimes = list(ttm["regime"])
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            prev_tax, prev_cur = regimes[i - 1].split(" ")
            tax, cur = regimes[i].split(" ")
            category = "taxonomy_change" if prev_tax != tax else "currency_change"
            dq.add(
                category,
                f"{regimes[i - 1]} -> {regimes[i]}",
                f"Reporting regime changed at the valuation date {ttm.at[i, 'valuation_date'].date()} ({ttm.at[i, 'form']} {ttm.at[i, 'accn']}). "
                "Series are not spliced across the break for statistics: statistics restart here. TTM values near the break can combine periods from both regimes.",
                period=ttm.at[i, "quarter_end"],
                filing=ttm.at[i, "accn"],
                severity="critical",
            )
            start = i
    return start


def _annual_table(cal: Calendar, tables, prices) -> pd.DataFrame:
    rows = []
    for fy in cal.years:
        if fy.end is None:
            continue
        row = {"fy_end": fy.end, "fy_start": fy.start, "fy_label": fy.label, "weeks": fy.week_note, "days": fy.days}
        for key in ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted", "diluted_shares", "ocf", "capex", "da", "dividends_paid", "dps_declared", "impairment"]:
            r = tables[key].get(fy.start, fy.end)
            row[key] = None if r is None else float(r["value"])
            row[f"accn_{key}"] = None if r is None else r["accn"]
        for key in ["cash", "total_assets", "total_liabilities", "equity", "long_term_debt"]:
            r = tables[key].get(None, fy.end)
            row[key] = None if r is None else float(r["value"])
            row[f"accn_{key}"] = None if r is None else r["accn"]
        row["fcf"] = row["ocf"] - row["capex"] if _num(row["ocf"]) and _num(row["capex"]) else None
        if prices is not None and prices.close is not None and fy.start >= prices.first_date - pd.Timedelta(days=7) and fy.end <= prices.last_date + pd.Timedelta(days=7):
            d = prices.dividends
            row["dps_inputc"] = float(d.loc[(d["ex_date"] >= fy.start) & (d["ex_date"] <= fy.end), "amount"].sum())
        else:
            row["dps_inputc"] = None
        rows.append(row)
    return pd.DataFrame(rows)


COVERAGE_ITEMS = [
    ("revenue", "Revenue"), ("gross_profit", "Gross profit"), ("operating_income", "Operating income"), ("net_income", "Net income (parent)"),
    ("eps_diluted", "Diluted EPS"), ("diluted_shares", "Diluted shares"), ("ocf", "Operating cash flow"), ("capex", "Capex"), ("da", "D&A"),
    ("dividends_paid", "Dividends paid"), ("cash", "Cash"), ("total_assets", "Total assets"), ("total_liabilities", "Total liabilities"),
    ("equity", "Parent equity"), ("long_term_debt", "Long-term debt"),
]


def _coverage(cal: Calendar, tables, cf: CompanyFacts, prices, as_of) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = [fy for fy in cal.years if fy.end is not None and fy.end <= as_of]
    cols = [str(fy.end.year) if sum(1 for y in years if y.end.year == fy.end.year) == 1 else str(fy.end.date()) for fy in years]
    matrix, summary = {}, []
    for key, label in COVERAGE_ITEMS:
        marks = []
        for fy in years:
            r = tables[key].get(fy.start, fy.end) if METRICS[key].kind == "duration" else tables[key].get(None, fy.end)
            marks.append("✓" if r is not None else "")
        matrix[label] = marks
        have = [fy for fy, m in zip(years, marks) if m]
        summary.append({"item": label, "first": have[0].label if have else "", "last": have[-1].label if have else "", "years": len(have)})
    cover = cf.facts[(cf.facts["tag"] == COVER_SHARES_TAG)]
    matrix["Cover-page shares"] = ["✓" if ((cover["filed"] >= fy.start) & (cover["filed"] <= fy.end + pd.Timedelta(days=120))).any() else "" for fy in years]
    if prices is not None and prices.close is not None:
        marks = []
        for fy in years:
            expected = expected_trading_days(fy.start, fy.end)
            have = prices.close[(prices.close.index >= fy.start) & (prices.close.index <= fy.end)]
            share = len(have) / len(expected) if len(expected) else 0
            marks.append("✓" if share >= 0.95 else ("partial" if share > 0 else ""))
        matrix["Prices (Input C)"] = marks
        summary.append({"item": "Prices (Input C)", "first": str(prices.first_date.date()), "last": str(prices.last_date.date()), "years": sum(1 for m in marks if m)})
    else:
        matrix["Prices (Input C)"] = [""] * len(years)
        summary.append({"item": "Prices (Input C)", "first": "", "last": "", "years": 0})
    return pd.DataFrame(matrix, index=cols).T, pd.DataFrame(summary)


def _missing_periods(cal: Calendar, ttm: pd.DataFrame, dq: DQLog) -> None:
    annual_only = []
    for fy in cal.years:
        missing = [k for k in (1, 2, 3, 4) if k not in fy.quarter_ends]
        if fy.end is not None and missing == [1, 2, 3]:
            annual_only.append(fy)
        elif fy.end is not None and missing and len(missing) < 4:
            dq.add("missing_period", "fiscal quarter end not identified", f"{fy.label}: no period data ends Q{', Q'.join(map(str, missing))}; those quarters and the TTM values that need them are blank.", period=fy.end)
    if annual_only:
        dq.add(
            "missing_period",
            "annual-only reporting",
            f"{len(annual_only)} fiscal year(s) ({annual_only[0].label} to {annual_only[-1].label}) have no interim XBRL periods (e.g. 20-F/40-F filers). "
            "Those years contribute one fiscal-year-end valuation date each (TTM = FY); no quarters are derived.",
        )


def _growth(annual: pd.DataFrame, ttm: pd.DataFrame) -> pd.DataFrame:
    if annual.empty:
        return pd.DataFrame()
    items = [("revenue", "Revenue"), ("operating_income", "Operating income"), ("net_income", "Net income"), ("fcf", "Free cash flow"), ("diluted_shares", "Diluted shares"), ("eps_diluted", "Diluted EPS")]
    rows = []
    a = annual.reset_index(drop=True)
    last = len(a) - 1
    for key, label in items:
        row = {"metric": label}
        for n in (5, 10, 15):
            j = last - n
            if j < 0:
                row[f"cagr_{n}"], row[f"note_{n}"] = None, f"fewer than {n + 1} fiscal years"
                continue
            v1, v0 = a.at[last, key], a.at[j, key]
            if not (_num(v1) and _num(v0)):
                row[f"cagr_{n}"], row[f"note_{n}"] = None, "endpoint missing"
            elif v1 <= 0 or v0 <= 0:
                row[f"cagr_{n}"], row[f"note_{n}"] = None, "non-positive endpoint"
            else:
                years = (a.at[last, "fy_end"] - a.at[j, "fy_end"]).days / 365.25
                row[f"cagr_{n}"] = (v1 / v0) ** (1 / n) - 1
                notes = []
                if abs(years - n) > 0.1:
                    notes.append(f"span {years:.2f} years (fiscal calendar change)")
                weeks = {a.at[last, "weeks"], a.at[j, "weeks"]}
                if "53-week" in weeks and len(weeks) > 1:
                    notes.append("one endpoint is a 53-week year")
                row[f"note_{n}"] = "; ".join(notes)
            row[f"from_{n}"] = str(a.at[j, "fy_end"].date()) if j >= 0 else ""
        row["to"] = str(a.at[last, "fy_end"].date())
        rows.append(row)
    return pd.DataFrame(rows)


def _latest_one_time(ttm: pd.DataFrame) -> dict:
    if ttm.empty:
        return {}
    last = ttm.iloc[-1]
    return {
        "net_income": last.get("ttm_net_income"),
        "impairment": last.get("ttm_impairment"),
        "discontinued operations": last.get("ttm_disc_ops"),
        "disposal gains": last.get("ttm_disposal_gains"),
        "_source": f"{last['form']} {last['accn']}",
    }


def _snapshot_inputs(inp: ReviewInputs, ttm: pd.DataFrame, bs: BalanceSheet, prices, price_block, as_of, dq) -> dict:
    out: dict = {"as_of": as_of}
    if inp.current_price is not None:
        out.update(price=float(inp.current_price), price_date=as_of, price_source="Input D current price", raw_price=float(inp.current_price), price_factor=1.0, price_reason="")
    elif prices is None:
        out.update(price=None, price_date=None, price_source="", price_reason="no price data")
    elif price_block:
        out.update(price=None, price_date=None, price_source="", price_reason=price_block)
    else:
        look = price_on_or_before(prices, as_of, allow_stale=True)
        out.update(price=look.close, price_date=look.date, price_source="last Input C close on or before the as-of date", raw_price=look.raw_close, price_factor=look.factor, price_reason=look.reason)
        if look.stale_days > 5:
            dq.add("stale_price", "current price", f"Last close {look.date.date()} is {look.stale_days} trading days before the as-of date {as_of.date()}.", period=look.date, severity="warning")
    out.update(bs.at(as_of))
    out["ttm_row"] = len(ttm) - 1 if not ttm.empty else None
    _market_values(out)
    return out


SNAPSHOT_FIELDS = [
    ("ttm_revenue", "TTM revenue", "money"), ("ttm_gross_profit", "TTM gross profit", "money"), ("ttm_operating_income", "TTM operating income (EBIT)", "money"),
    ("ttm_net_income", "TTM net income attributable to parent", "money"), ("ttm_ocf", "TTM operating cash flow", "money"), ("ttm_capex", "TTM capex", "money"),
    ("ttm_fcf", "TTM free cash flow (OCF - capex)", "money"), ("ttm_da", "TTM D&A (cash flow statement)", "money"), ("ttm_ebitda", "TTM EBITDA (operating income + D&A)", "money"),
    ("ttm_dividends_paid", "TTM dividends paid", "money"), ("avg_diluted_shares", "TTM average diluted shares", "shares"), ("eps_ttm", "TTM diluted EPS", "per_share"),
    ("revenue_ps", "Revenue per share", "per_share"), ("gp_ps", "Gross profit per share", "per_share"), ("oi_ps", "Operating income per share", "per_share"),
    ("ocf_ps", "Operating cash flow per share", "per_share"), ("fcf_ps", "Free cash flow per share", "per_share"), ("da_ps", "D&A per share", "per_share"),
    ("ebitda_ps", "EBITDA per share", "per_share"), ("dps_ttm", "TTM dividends per share (Input C)", "per_share"), ("eps_cagr5", "5-year TTM EPS CAGR", "pct"),
]


def _text(src: dict, key: str) -> str:
    """A reason or note field as text; NaN and None become ''."""
    v = src.get(key)
    return v if isinstance(v, str) else ""


def ratio_inputs(row: dict | pd.Series, snap: dict | None = None) -> dict:
    """Map a TTM row (and optional snapshot market inputs) to ratio numerator/denominator names."""
    src = dict(row)
    if snap is not None:
        for key in ("price", "mcap", "ev", "equity", "tbv", "shares_out", "cash", "total_borrowings", "nci", "preferred", "price_reason", "ev_reason", "tbv_reason"):
            src[key] = snap.get(key)
        so, eq = snap.get("shares_out"), snap.get("equity")
        src["bvps"] = eq / so if _num(eq) and _num(so) and so > 0 else None
    clean = {k: (float(v) if _num(v) else (None if not isinstance(v, str) else v)) for k, v in src.items()}
    return {
        "price": clean.get("price"), "mcap": clean.get("mcap"), "ev": clean.get("ev"),
        "eps": clean.get("eps_ttm"), "revenue": clean.get("ttm_revenue"), "gross_profit": clean.get("ttm_gross_profit"),
        "equity": clean.get("equity"), "tbv": clean.get("tbv"), "ocf": clean.get("ttm_ocf"), "fcf": clean.get("ttm_fcf"),
        "dps": clean.get("dps_ttm"), "operating_income": clean.get("ttm_operating_income"), "ebitda": clean.get("ttm_ebitda"),
        "bvps": clean.get("bvps"), "cagr": clean.get("eps_cagr5"), "cagr_reason": _text(src, "eps_cagr5_reason"),
        "price_reason": _text(src, "price_reason") or "price unavailable",
        "mcap_reason": _text(src, "price_reason") or ("no shares outstanding available for market cap" if not _num(src.get("shares_out")) else "market cap unavailable"),
        "ev_reason": _text(src, "ev_reason") or "EV unavailable", "tbv_reason": _text(src, "tbv_reason") or "TBV unavailable",
        "dps_reason": _text(src, "dps_reason") or "TTM dividends per share unavailable",
        "eps_reason": "TTM EPS unavailable" + (f" ({_text(src, 'shares_note') or 'no diluted share count'})" if not _num(src.get("avg_diluted_shares")) else " (net income missing)"),
        "revenue_reason": "TTM revenue unavailable", "gross_profit_reason": "TTM gross profit unavailable",
        "equity_reason": "parent equity unavailable", "ocf_reason": "TTM operating cash flow unavailable", "fcf_reason": _text(src, "fcf_reason") or "TTM FCF unavailable",
        "operating_income_reason": _text(src, "operating_income_reason") or "TTM operating income unavailable",
        "ebitda_reason": _text(src, "ebitda_reason") or "TTM EBITDA unavailable",
        "bvps_reason": "BVPS unavailable",
    }


def _snapshot_table(review: Review) -> tuple[pd.DataFrame, dict]:
    ttm, snap = review.ttm, review.snapshot_inputs
    if ttm.empty:
        return pd.DataFrame(), {}
    last = ttm.iloc[-1]
    rows = []
    for key, label, unit in SNAPSHOT_FIELDS:
        rows.append({"group": "TTM / per-share TTM", "key": key, "item": label, "value": last.get(key), "unit": unit, "valuation_date": last["valuation_date"], "inputs": f"TTM ending {last['quarter_end'].date()} ({last['form']} {last['accn']})", "formula": "", "shares": "", "percentile": None, "reason": last.get("eps_cagr5_reason") if key == "eps_cagr5" else ""})
    so, eq = snap.get("shares_out"), snap.get("equity")
    bvps = eq / so if _num(eq) and _num(so) and so > 0 else None
    rows.append({"group": "Balance sheet / market", "key": "bvps", "item": "Book value per share", "value": bvps, "unit": "per_share", "valuation_date": snap.get("bs_filed"), "inputs": f"equity at {snap.get('bs_instant').date() if snap.get('bs_instant') is not None else '-'} / cover-page shares", "formula": "Parent equity / shares outstanding (cover page)", "shares": "cover-page shares outstanding", "percentile": None, "reason": ""})
    for key, label in (("price", "Price"), ("mcap", "Market cap"), ("ev", "Enterprise value")):
        rows.append({"group": "Balance sheet / market", "key": key, "item": label, "value": snap.get(key), "unit": "per_share" if key == "price" else "money", "valuation_date": snap.get("price_date"), "inputs": snap.get("price_source", ""), "formula": {"price": "", "mcap": "Price x shares outstanding (cover page)", "ev": "Market cap + borrowings + NCI + preferred - cash"}[key], "shares": "cover-page shares outstanding" if key != "price" else "", "percentile": None, "reason": snap.get("price_reason") if key == "price" else snap.get("ev_reason", "") if key == "ev" else ""})
    inputs = ratio_inputs(last, snap)
    values = compute_all(inputs)
    percentiles = {}
    daily = review.daily
    for ratio in RATIOS:
        value, reason = values[ratio.key]
        pct = None
        if ratio.daily and daily is not None and ratio.key in daily and _num(value):
            hist = daily[ratio.key].dropna()
            if len(hist):
                pct = 100.0 * float((hist <= value).sum()) / len(hist)
                percentiles[ratio.key] = {"percentile": pct, "n": int(len(hist)), "from": hist.index.min(), "to": hist.index.max()}
        shares_note = ratio.shares
        source = snap.get("shares_source") or ""
        if source and not source.startswith("dei:"):
            shares_note = shares_note.replace("cover-page shares outstanding", source)
        rows.append({"group": "Price ratios", "key": ratio.key, "item": ratio.label, "value": value, "unit": ratio.kind, "valuation_date": last["valuation_date"], "inputs": f"price {snap.get('price_date').date() if snap.get('price_date') is not None else '-'}; TTM ending {last['quarter_end'].date()}; balance sheet {snap.get('bs_instant').date() if snap.get('bs_instant') is not None else '-'}", "formula": ratio.formula, "shares": shares_note, "percentile": pct, "reason": reason})
    return pd.DataFrame(rows), percentiles


def _flag_extremes(review: Review) -> None:
    for _, row in review.snapshot.iterrows():
        if row["group"] == "Price ratios" and row["unit"] == "multiple" and _num(row["value"]) and abs(row["value"]) > 100:
            review.dq.add("extreme_multiple", row["item"], f"Current {row['item']} = {row['value']:,.1f}x (> 100x); check inputs: {row['inputs']}.", severity="warning")
    if review.ttm.empty:
        return
    for i, last in review.ttm.iterrows():
        values = compute_all(ratio_inputs(last))
        for ratio in RATIOS:
            v = values[ratio.key][0]
            if is_extreme(ratio, v):
                review.dq.add("extreme_multiple", ratio.label, f"{ratio.label} = {v:,.1f}x at valuation date {last['valuation_date'].date()} (TTM ending {last['quarter_end'].date()}); check inputs.", period=last["valuation_date"], filing=last["accn"])


def _header(review: Review, currencies: list[str]) -> dict:
    cf, inp = review.cf, review.inputs
    periodic = cf.filings.dropna(subset=["report_period"])
    latest = periodic.iloc[-1] if not periodic.empty else cf.filings.iloc[-1]
    regimes = sorted(set(review.ttm["regime"])) if not review.ttm.empty else []
    full_start = None
    if not review.ttm.empty:
        sub = review.ttm.iloc[review.stats_start_index:]
        have = sub[[s.key for s in SERIES if s.key in sub]].notna().any(axis=1)
        if have.any():
            full_start = sub.loc[have.idxmax(), "valuation_date"]
    p = review.prices
    meta = inp.price_meta
    return {
        "company": cf.entity_name,
        "ticker": inp.ticker or meta.ticker,
        "cik": cf.cik,
        "taxonomy": ", ".join(cf.taxonomies),
        "currency": ", ".join(currencies),
        "regimes": regimes,
        "latest_filing": f"{latest['form']} {latest['accn']} filed {latest['filed'].date()}",
        "as_of": review.snapshot_inputs["as_of"],
        "display_window": f"{review.display_start.date()} to {review.snapshot_inputs['as_of'].date()}" if review.display_start is not None else "n/a",
        "full_history_start": full_start,
        "price_range": f"{p.first_date.date()} to {p.last_date.date()}" if p is not None else "no price data",
        "price_basis": (
            f"split-adjusted: {_yn(meta.split_adjusted)}; dividend-adjusted: {_yn(meta.dividend_adjusted)}; currency: {meta.currency or 'not stated'}"
            if p is not None
            else "n/a"
        ),
        "splits": review.basis.describe_within(review.cf.facts["filed"].min()),
    }


def _yn(v):
    return "not stated" if v is None else ("yes" if v else "no")


# ---------------------------------------------------------------- daily

def daily_table(review: Review) -> pd.DataFrame | None:
    prices, ttm = review.prices, review.ttm
    if prices is None or prices.close is None or ttm.empty or review.price_block_reason:
        return None
    rows = ttm.dropna(subset=["valuation_date"]).sort_values("valuation_date").reset_index(drop=True)
    rows = rows.drop_duplicates("valuation_date", keep="last")
    rows["held_row"] = rows.index
    close = prices.close
    days = pd.DataFrame({"date": close.index, "price": close.to_numpy(), "raw_close": prices.close_raw.reindex(close.index).to_numpy()})
    days = days[days["date"] >= rows["valuation_date"].min()]
    keep = [
        "valuation_date", "quarter_end", "form", "accn", "held_row", "currency_ok", "shares_out", "cash", "total_borrowings", "nci", "preferred", "equity", "tbv",
        "eps_ttm", "ttm_revenue", "ttm_gross_profit", "ttm_ocf", "ttm_fcf", "dps_ttm", "ttm_operating_income", "ttm_ebitda", "eps_cagr5",
    ] + [s.key for s in SERIES if s.key not in ("eps_ttm", "ttm_revenue", "ttm_gross_profit", "ttm_ocf", "ttm_fcf", "dps_ttm", "ttm_operating_income", "ttm_ebitda", "equity")]
    keep = list(dict.fromkeys(k for k in keep if k in rows))
    days["date"] = days["date"].astype("datetime64[ns]")
    right = rows[keep].copy()
    right["valuation_date"] = right["valuation_date"].astype("datetime64[ns]")
    merged = pd.merge_asof(days.sort_values("date"), right.sort_values("valuation_date"), left_on="date", right_on="valuation_date", direction="backward")
    num = lambda c: pd.to_numeric(merged[c], errors="coerce")  # noqa: E731
    merged["mcap"] = merged["price"] * num("shares_out")
    merged["ev"] = merged["mcap"] + num("total_borrowings").fillna(0) + num("nci").fillna(0) + num("preferred").fillna(0) - num("cash")
    uncertain = merged["date"].map(lambda d: bool(prices.uncertain_days.get(d, False)))
    if "currency_ok" in merged:
        uncertain = uncertain | ~merged["currency_ok"].fillna(True).astype(bool)  # no FX input: leave blank
    src = {
        "price": merged["price"], "mcap": merged["mcap"], "ev": merged["ev"], "eps": num("eps_ttm"), "revenue": num("ttm_revenue"),
        "gross_profit": num("ttm_gross_profit"), "equity": num("equity"), "tbv": num("tbv"), "ocf": num("ttm_ocf"), "fcf": num("ttm_fcf"),
        "dps": num("dps_ttm"), "operating_income": num("ttm_operating_income"), "ebitda": num("ttm_ebitda"),
    }
    for ratio in DAILY_RATIOS:
        if ratio.key == "peg":
            continue
        n, d = src[ratio.numerator], src[ratio.denominator]
        value = n / d
        nm = d <= 0
        value = value.where(~nm & n.notna() & d.notna())
        merged[ratio.key] = value.where(~uncertain)
        merged[f"{ratio.key}_nm"] = nm & n.notna() & ~uncertain
    cagr = num("eps_cagr5")
    pe = merged["pe"]
    merged["peg"] = (pe / (cagr * 100)).where((cagr > 0) & pe.notna())
    merged["peg_nm"] = merged["pe_nm"] & (cagr > 0)
    merged["basis_uncertain"] = uncertain
    merged["in_display_window"] = merged["date"] >= review.display_start if review.display_start is not None else True
    return merged.set_index("date")


def _daily_counts(review: Review) -> dict:
    daily = review.daily
    if daily is None:
        return {}
    window = daily[daily["in_display_window"]]
    return {r.key: int(window[r.key].notna().sum()) for r in DAILY_RATIOS}

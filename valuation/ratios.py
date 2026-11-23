"""Ratio definitions shared by the snapshot, the per-filing table and the daily series."""

from __future__ import annotations

import math
from dataclasses import dataclass

NM = "NM"
EXTREME = 100.0


@dataclass(frozen=True)
class RatioDef:
    key: str
    label: str
    numerator: str
    denominator: str
    kind: str  # "multiple", "yield", "ratio", "value"
    formula: str
    shares: str
    daily: bool = True


RATIOS: list[RatioDef] = [
    RatioDef("pe", "P/E", "price", "eps", "multiple", "Price / TTM diluted EPS", "TTM average diluted shares (in EPS)"),
    RatioDef("earnings_yield", "Earnings yield", "eps", "price", "yield", "TTM diluted EPS / Price", "TTM average diluted shares (in EPS)"),
    RatioDef("ps", "P/S", "mcap", "revenue", "multiple", "Market cap / TTM revenue", "cover-page shares outstanding (market cap)"),
    RatioDef("pgp", "P/GP", "mcap", "gross_profit", "multiple", "Market cap / TTM gross profit", "cover-page shares outstanding (market cap)"),
    RatioDef("pb", "P/B", "mcap", "equity", "multiple", "Market cap / parent equity", "cover-page shares outstanding (market cap)"),
    RatioDef("ptbv", "P/TBV", "mcap", "tbv", "multiple", "Market cap / (parent equity - goodwill - intangibles)", "cover-page shares outstanding (market cap)"),
    RatioDef("pcfo", "P/CFO", "mcap", "ocf", "multiple", "Market cap / TTM operating cash flow", "cover-page shares outstanding (market cap)"),
    RatioDef("pfcf", "P/FCF", "mcap", "fcf", "multiple", "Market cap / TTM free cash flow", "cover-page shares outstanding (market cap)"),
    RatioDef("fcf_yield", "FCF yield", "fcf", "mcap", "yield", "TTM free cash flow / Market cap", "cover-page shares outstanding (market cap)"),
    RatioDef("div_yield", "Dividend yield", "dps", "price", "yield", "TTM dividends per share (Input C ex-dates) / Price", "per share as supplied in Input C"),
    RatioDef("payout", "Payout ratio", "dps", "eps", "ratio", "TTM dividends per share / TTM diluted EPS", "Input C DPS; TTM average diluted shares (in EPS)", daily=False),
    RatioDef("ev_revenue", "EV/Revenue", "ev", "revenue", "multiple", "EV / TTM revenue", "cover-page shares outstanding (in EV)"),
    RatioDef("ev_gp", "EV/Gross profit", "ev", "gross_profit", "multiple", "EV / TTM gross profit", "cover-page shares outstanding (in EV)"),
    RatioDef("ev_ebit", "EV/EBIT", "ev", "operating_income", "multiple", "EV / TTM operating income", "cover-page shares outstanding (in EV)"),
    RatioDef("ev_ebitda", "EV/EBITDA", "ev", "ebitda", "multiple", "EV / (TTM operating income + TTM D&A)", "cover-page shares outstanding (in EV)"),
    RatioDef("ev_cfo", "EV/CFO", "ev", "ocf", "multiple", "EV / TTM operating cash flow", "cover-page shares outstanding (in EV)"),
    RatioDef("ev_fcf", "EV/FCF", "ev", "fcf", "multiple", "EV / TTM free cash flow", "cover-page shares outstanding (in EV)"),
    RatioDef("peg", "PEG", "pe", "cagr_pct", "multiple", "P/E / (5-year TTM EPS CAGR x 100), only when CAGR > 0", "TTM average diluted shares (in EPS)"),
    RatioDef("graham", "Graham number", "eps", "bvps", "value", "sqrt(22.5 x TTM EPS x BVPS), only when both > 0", "EPS: TTM average diluted; BVPS: cover-page shares", daily=False),
]
RATIO_BY_KEY = {r.key: r for r in RATIOS}
DAILY_RATIOS = [r for r in RATIOS if r.daily]


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def compute(ratio: RatioDef, inputs: dict) -> tuple[object, str]:
    """Return (value | 'NM' | None, reason). None means missing."""
    num = inputs.get(ratio.numerator)
    den = inputs.get(ratio.denominator)
    if ratio.key == "graham":
        if not (_num(num) and _num(den)):
            return None, "missing " + " and ".join(n for n, v in (("TTM EPS", num), ("BVPS", den)) if not _num(v))
        if num <= 0 or den <= 0:
            return NM, "EPS or BVPS not positive"
        return math.sqrt(22.5 * num * den), ""
    if ratio.key == "peg":
        cagr = inputs.get("cagr")
        if not _num(cagr):
            return None, inputs.get("cagr_reason") or "5-year EPS CAGR unavailable"
        if cagr <= 0:
            return None, "5-year EPS CAGR not positive; PEG not computed"
        if num == NM:
            return NM, "P/E is NM"
        if not _num(num):
            return None, "P/E unavailable"
        return num / (cagr * 100.0), ""
    if not _num(num) or not _num(den):
        missing = [name for name, value in ((ratio.numerator, num), (ratio.denominator, den)) if not _num(value)]
        reasons = [inputs.get(f"{m}_reason") or f"{m} missing" for m in missing]
        return None, "; ".join(reasons)
    if den <= 0:
        return NM, f"{ratio.denominator} is zero or negative"
    return num / den, ""


def compute_all(inputs: dict, keys: list[str] | None = None) -> dict[str, tuple[object, str]]:
    out: dict[str, tuple[object, str]] = {}
    for ratio in RATIOS:
        if keys is not None and ratio.key not in keys:
            continue
        if ratio.key == "peg":
            continue
        out[ratio.key] = compute(ratio, inputs)
    if keys is None or "peg" in keys:
        pe_value = out.get("pe", compute(RATIO_BY_KEY["pe"], inputs))[0]
        peg_inputs = dict(inputs, pe=pe_value, cagr_pct=None)
        out["peg"] = compute(RATIO_BY_KEY["peg"], peg_inputs)
    return out


def is_extreme(ratio: RatioDef, value) -> bool:
    return ratio.kind == "multiple" and _num(value) and abs(value) > EXTREME

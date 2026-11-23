"""Red-flag screens. Every flag cites the filing it comes from.

Data screens use Input A. Text screens (going concern, auditor opinion,
material weakness, auditor change) need Input B; without it they are reported
as not assessable rather than clean.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd

ACCN_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")

DEADLINE_DAYS = {"10-K": 90, "10-Q": 45, "20-F": 120, "10-KT": 90, "10-QT": 45}


@dataclass
class InputBDocument:
    name: str
    text: str

    @property
    def citation(self) -> str:
        match = ACCN_RE.search(self.name) or ACCN_RE.search(self.text[:5000])
        return f"{self.name}" + (f" (accession {match.group(0)})" if match else "")


def clean_text(raw: str) -> str:
    if "<" in raw and ">" in raw:
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        raw = html.unescape(raw)
    return re.sub(r"\s+", " ", raw)


def _sentences_with(text: str, pattern: re.Pattern, limit: int = 3) -> list[str]:
    out = []
    for match in pattern.finditer(text):
        lo = text.rfind(".", 0, match.start()) + 1
        hi = text.find(".", match.end())
        sentence = text[lo : hi + 1 if hi != -1 else len(text)].strip()
        if sentence and sentence not in out:
            out.append(sentence[:400])
        if len(out) >= limit:
            break
    return out


TEXT_SCREENS = [
    ("going concern", re.compile(r"substantial doubt[^.]{0,200}going concern|going concern[^.]{0,200}substantial doubt", re.I), "critical"),
    ("non-clean auditor opinion", re.compile(r"adverse opinion|qualified opinion|disclaimer of opinion|do not express an opinion|except for the (?:effects|possible effects)", re.I), "critical"),
    ("material weakness", re.compile(r"material weakness(?:es)?[^.]{0,160}(?:identified|exist|not effective|ineffective)|(?:not effective|ineffective)[^.]{0,160}material weakness", re.I), "serious"),
    ("auditor change", re.compile(r"(?:dismiss(?:ed|al)|resign(?:ed|ation)|engaged)[^.]{0,200}independent registered public accounting firm|change(?:s)? in (?:the )?(?:company'?s )?(?:independent )?(?:registered public )?account(?:ant|ing firm)", re.I), "serious"),
    ("impairment (text)", re.compile(r"impairment (?:charge|loss)[^.]{0,120}(?:\$|million|billion)", re.I), "warning"),
]


def text_flags(docs: list[InputBDocument]) -> list[dict]:
    flags = []
    for doc in docs:
        text = clean_text(doc.text)
        clean_opinion = re.search(r"present fairly, in all material respects", text, re.I)
        for name, pattern, severity in TEXT_SCREENS:
            hits = _sentences_with(text, pattern)
            if name == "non-clean auditor opinion" and hits and clean_opinion:
                severity = "warning"  # wording found alongside standard clean-opinion language; read in context
            for sentence in hits:
                flags.append({"flag": name, "severity": severity, "evidence": f"“{sentence}”", "source": doc.citation})
    return flags


def data_flags(review) -> list[dict]:
    flags: list[dict] = []
    ttm: pd.DataFrame = review.ttm
    annual: pd.DataFrame = review.annual_all

    # Negative equity at any valuation date.
    neg = ttm[pd.to_numeric(ttm["equity"], errors="coerce") < 0] if "equity" in ttm else ttm.iloc[0:0]
    if not neg.empty:
        last = neg.iloc[-1]
        flags.append(
            {
                "flag": "negative equity",
                "severity": "serious",
                "evidence": f"Parent equity negative at {len(neg)} valuation date(s); latest {last['equity']:,.0f} at {last['bs_instant'].date()}.",
                "source": f"{last['bs_form']} {last['bs_accn']}",
            }
        )

    # Persistent negative FCF: three or more consecutive negative fiscal years, or 3 of the last 5.
    fcf = annual[["fy_end", "fcf", "accn_ocf"]].dropna(subset=["fcf"]) if not annual.empty and "fcf" in annual else pd.DataFrame()
    if not fcf.empty:
        negative = list(fcf["fcf"] < 0)
        run, best = 0, 0
        for is_neg in negative:
            run = run + 1 if is_neg else 0
            best = max(best, run)
        last5 = negative[-5:]
        if best >= 3 or sum(last5) >= 3:
            years = fcf[fcf["fcf"] < 0]
            flags.append(
                {
                    "flag": "persistent negative FCF",
                    "severity": "serious",
                    "evidence": f"Annual FCF negative in {len(years)} fiscal year(s), longest run {best}; years ending "
                    + ", ".join(str(d.date()) for d in years["fy_end"].tail(6)),
                    "source": "; ".join(str(a) for a in years["accn_ocf"].tail(3)),
                }
            )

    # Dilution not explained by a detected split (shares are already split-normalized).
    if not annual.empty and "diluted_shares" in annual:
        sh = annual[["fy_end", "diluted_shares", "accn_diluted_shares"]].dropna()
        for (_, a), (_, b) in zip(sh.iloc[:-1].iterrows(), sh.iloc[1:].iterrows()):
            change = b["diluted_shares"] / a["diluted_shares"] - 1 if a["diluted_shares"] else 0
            if change > 0.05:
                flags.append(
                    {
                        "flag": "dilution",
                        "severity": "warning",
                        "evidence": f"Diluted weighted shares (split-normalized) rose {change:.1%} in the fiscal year ending {b['fy_end'].date()}; not explained by a detected split.",
                        "source": str(b["accn_diluted_shares"]),
                    }
                )

    # Late periodic filings (beyond the longest standard deadline for the form).
    filings = review.filings
    for row in filings.itertuples():
        base_form = row.form.replace("/A", "")
        if row.form.endswith("/A") or base_form not in DEADLINE_DAYS or pd.isna(row.report_period):
            continue
        limit = DEADLINE_DAYS[base_form]
        if base_form == "20-F" and row.report_period < pd.Timestamp("2011-12-15"):
            limit = 183
        days = (row.filed - row.report_period).days
        if days > limit:
            flags.append(
                {
                    "flag": "late filing",
                    "severity": "serious",
                    "evidence": f"{row.form} for the period ending {row.report_period.date()} filed {row.filed.date()} ({days} days; longest standard deadline {limit}).",
                    "source": f"{row.form} {row.accn}",
                }
            )

    # Impairments by fiscal year.
    if not annual.empty and "impairment" in annual:
        imp = annual[["fy_end", "impairment", "accn_impairment"]].dropna()
        for _, row in imp[imp["impairment"] > 0].iterrows():
            flags.append(
                {
                    "flag": "impairment",
                    "severity": "warning",
                    "evidence": f"Impairment charges {row['impairment']:,.0f} in the fiscal year ending {row['fy_end'].date()}.",
                    "source": str(row["accn_impairment"]),
                }
            )

    # Multiples driven by one-time items (latest TTM).
    latest = review.latest_one_time
    if latest:
        ni = latest.get("net_income")
        items = {
            k: float(v)
            for k, v in latest.items()
            if k != "net_income" and not k.startswith("_") and isinstance(v, (int, float)) and v == v and v != 0
        }
        if isinstance(ni, (int, float)) and ni == ni and ni and items:
            total = sum(abs(v) for v in items.values())
            if total > 0.2 * abs(ni):
                flags.append(
                    {
                        "flag": "multiples driven by one-time items",
                        "severity": "serious",
                        "evidence": "Latest TTM one-time items "
                        + ", ".join(f"{k} {v:,.0f}" for k, v in items.items())
                        + f" equal {total / abs(ni):.0%} of TTM net income {ni:,.0f}; earnings-based multiples are affected.",
                        "source": latest.get("_source", ""),
                    }
                )

    for row in review.amendments.itertuples():
        flags.append({"flag": "amendment", "severity": "info", "evidence": f"{row.form} filed {row.filed.date()}.", "source": f"{row.form} {row.accn}"})
    return flags

"""Fiscal calendar, quarterly values and TTM construction.

Periods are identified by start/end dates and duration only (never by the
fy/fp fields). Durations: annual 350-380 days, quarter 80-100 days, 6M YTD
170-200, 9M YTD 260-290. Anything else is a stub or transition period and is
kept out of trend math.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dq import DQLog
from .values import MetricTable

TOL = pd.Timedelta(days=3)
TTM_DISAGREE_TOL = 0.001
QUARTER_MISMATCH_TOL = 0.005


def classify_days(days: float) -> str:
    if pd.isna(days):
        return "instant"
    if 350 <= days <= 380:
        return "annual"
    if 80 <= days <= 100:
        return "quarter"
    if 170 <= days <= 200:
        return "ytd6"
    if 260 <= days <= 290:
        return "ytd9"
    if 101 <= days <= 125:
        return "long_quarter"  # e.g. a 16-week first quarter; valid only as Q1 starting at the fiscal-year start
    return "other"


@dataclass
class FiscalYear:
    index: int
    start: pd.Timestamp
    end: pd.Timestamp | None
    quarter_ends: dict[int, pd.Timestamp] = field(default_factory=dict)
    consecutive_with_prior: bool = False

    @property
    def days(self) -> int | None:
        return None if self.end is None else int((self.end - self.start).days + 1)

    @property
    def week_note(self) -> str:
        if self.days == 364:
            return "52-week"
        if self.days == 371:
            return "53-week"
        return ""

    @property
    def label(self) -> str:
        return f"FY{self.end.year}" if self.end is not None else f"FY in progress (from {self.start.date()})"


@dataclass(frozen=True)
class Quarter:
    fy: int
    k: int
    start: pd.Timestamp | None  # None when the preceding quarter end is unknown (e.g. annual-only 20-F reporting)
    end: pd.Timestamp

    @property
    def label(self) -> str:
        return f"Q{self.k} ending {self.end.date()}"


@dataclass
class Calendar:
    years: list[FiscalYear]
    quarters: list[Quarter]

    def quarter(self, fy: int, k: int) -> Quarter | None:
        for q in self.quarters:
            if q.fy == fy and q.k == k:
                return q
        return None

    def previous_quarters(self, q: Quarter, n: int) -> list[Quarter] | None:
        """The n consecutive quarters ending at q (inclusive), or None if any is missing."""
        out = [q]
        fy, k = q.fy, q.k
        while len(out) < n:
            k -= 1
            if k == 0:
                if not self.years[fy].consecutive_with_prior:
                    return None
                fy, k = fy - 1, 4
            prior = self.quarter(fy, k)
            if prior is None:
                return None
            out.insert(0, prior)
        return out


def _cluster_dates(dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Merge dates within 3 days; the most frequent member represents the cluster."""
    if not dates:
        return []
    ordered = sorted(dates)
    clusters: list[list[pd.Timestamp]] = [[ordered[0]]]
    for d in ordered[1:]:
        if d - clusters[-1][-1] <= TOL:
            clusters[-1].append(d)
        else:
            clusters.append([d])
    return [Counter(c).most_common(1)[0][0] for c in clusters]


def _mode_end(periods: pd.DataFrame, start: pd.Timestamp, kinds: tuple[str, ...]) -> pd.Timestamp | None:
    hits = periods[(periods["cls"].isin(kinds)) & ((periods["start"] - start).abs() <= TOL)]
    if hits.empty:
        return None
    return Counter(hits["end"]).most_common(1)[0][0]


def _following_quarter_end(periods: pd.DataFrame, prior_end: pd.Timestamp) -> pd.Timestamp | None:
    nxt = prior_end + pd.Timedelta(days=1)
    hits = periods[(periods["cls"] == "quarter") & ((periods["start"] - nxt).abs() <= TOL)]
    if hits.empty:
        return None
    return Counter(hits["end"]).most_common(1)[0][0]


def build_calendar(tables: list[MetricTable], dq: DQLog, as_of: pd.Timestamp) -> Calendar:
    frames = [t.table[["start", "end", "days"]] for t in tables if not t.table.empty]
    if not frames:
        return Calendar([], [])
    periods = pd.concat(frames, ignore_index=True).drop_duplicates()
    periods = periods[periods["end"] <= as_of].copy()
    periods["cls"] = periods["days"].map(classify_days)

    starts = list(periods.loc[periods["cls"].isin(["annual", "ytd6", "ytd9"]), "start"])
    starts += [e + pd.Timedelta(days=1) for e in periods.loc[periods["cls"] == "annual", "end"]]
    fy_starts = _cluster_dates(starts)

    years: list[FiscalYear] = []
    for start in fy_starts:
        end = _mode_end(periods, start, ("annual",))
        q1 = _mode_end(periods, start, ("quarter",)) or _mode_end(periods, start, ("long_quarter",))
        q2 = _mode_end(periods, start, ("ytd6",)) or (q1 and _following_quarter_end(periods, q1))
        q3 = _mode_end(periods, start, ("ytd9",)) or (q2 and _following_quarter_end(periods, q2))
        # Q4 is only ever the annual period's end; a trailing 3M period alone does not define a year end.
        ends ={k: v for k, v in {1: q1, 2: q2, 3: q3, 4: end}.items() if v is not None}
        if not ends:
            continue
        # Discard quarter ends that are out of order.
        valid, last = {}, start
        for k in sorted(ends):
            if ends[k] > last:
                valid[k] = ends[k]
                last = ends[k]
        years.append(FiscalYear(len(years), start, end, valid))

    # Drop provisional years that overlap a real fiscal year (e.g. E+1 of a changed fiscal year).
    cleaned: list[FiscalYear] = []
    for fy in years:
        if cleaned and cleaned[-1].end is not None and fy.start <= cleaned[-1].end:
            continue
        cleaned.append(fy)
    years = cleaned
    for i, fy in enumerate(years):
        fy.index = i
        if i > 0 and years[i - 1].end is not None:
            fy.consecutive_with_prior = abs(fy.start - (years[i - 1].end + pd.Timedelta(days=1))) <= TOL

    quarters: list[Quarter] = []
    for fy in years:
        for k in (1, 2, 3, 4):
            if k not in fy.quarter_ends:
                continue  # a missing quarter end breaks quarter chains; the fiscal-year end still stands on its own
            if k == 1:
                q_start = fy.start
            elif (k - 1) in fy.quarter_ends:
                q_start = fy.quarter_ends[k - 1] + pd.Timedelta(days=1)
            else:
                q_start = None
            quarters.append(Quarter(fy.index, k, q_start, fy.quarter_ends[k]))

    _log_calendar(years, periods, dq)
    return Calendar(years, quarters)


def _log_calendar(years: list[FiscalYear], periods: pd.DataFrame, dq: DQLog) -> None:
    for fy in years:
        if fy.week_note == "53-week":
            dq.add("period_label", "53-week fiscal year", f"{fy.label} runs {fy.start.date()} to {fy.end.date()} (371 days). Comparisons with 52-week years are not adjusted.", period=fy.end)
    for i in range(1, len(years)):
        prev, cur = years[i - 1], years[i]
        if prev.end is None:
            continue
        if not cur.consecutive_with_prior:
            dq.add(
                "fiscal_calendar",
                "fiscal year gap or change",
                f"Fiscal year starting {cur.start.date()} does not follow the year ending {prev.end.date()}; TTM chains break here.",
                period=cur.start,
                severity="warning",
            )
        if cur.end is not None and abs((cur.end.month * 31 + cur.end.day) - (prev.end.month * 31 + prev.end.day)) > 7 and abs(
            (cur.end.month * 31 + cur.end.day) - (prev.end.month * 31 + prev.end.day)
        ) < 365 - 7:
            dq.add("fiscal_calendar", "fiscal year end change", f"Fiscal year end moved from {prev.end.strftime('%m-%d')} to {cur.end.strftime('%m-%d')}.", period=cur.end, severity="warning")
    long_q1 = [fy for fy in years if 1 in fy.quarter_ends and (fy.quarter_ends[1] - fy.start).days + 1 > 100]
    if long_q1:
        days = sorted({(fy.quarter_ends[1] - fy.start).days + 1 for fy in long_q1})
        dq.add(
            "period_label",
            "non-standard first quarter",
            f"{len(long_q1)} fiscal year(s) have a first quarter of {', '.join(map(str, days))} days (e.g. a 16-week Q1). It is used only inside YTD "
            "differencing and TTM sums, which still cover full 52/53-week years; it is never trended on its own.",
        )
    starts_of_years = {fy.start for fy in years}
    other = periods[(periods["cls"] == "other") | ((periods["cls"] == "long_quarter") & ~periods["start"].isin(starts_of_years))].drop_duplicates(["start", "end"])
    for _, row in other.iterrows():
        dq.add(
            "period_label",
            "stub/transition period",
            f"Period {row['start'].date()} to {row['end'].date()} ({int(row['days'])} days) is neither annual nor quarterly; excluded from trend math.",
            period=row["end"],
        )


# ---------------------------------------------------------------- quarterly values

def quarterly_values(metric: MetricTable, cal: Calendar, dq: DQLog) -> pd.DataFrame:
    rows = []
    additive = metric.metric.additive
    for q in cal.quarters:
        fy = cal.years[q.fy]
        reported = metric.get(q.start, q.end) if q.start is not None else None
        if reported is not None and not (classify_days(reported["days"]) == "quarter" or (q.k == 1 and classify_days(reported["days"]) == "long_quarter")):
            reported = None
        ytd = metric.get(fy.start, q.end)
        prev_ytd = None
        if q.k >= 2:
            prev_q = cal.quarter(q.fy, q.k - 1)
            prev_ytd = metric.get(fy.start, prev_q.end) if prev_q is not None else None
        derived = None
        if additive and q.k >= 2 and ytd is not None and prev_ytd is not None:
            derived = float(ytd["value"]) - float(prev_ytd["value"])
        value, source = None, ""
        if reported is not None:
            value, source = float(reported["value"]), "reported"
            if derived is not None and _rel_diff(value, derived) > QUARTER_MISMATCH_TOL:
                dq.add(
                    "quarter_mismatch",
                    metric.metric.label,
                    f"{q.label}: reported 3M {value:,.4g} vs YTD difference {derived:,.4g}; reported value used.",
                    period=q.end,
                    filing=reported["accn"],
                )
        elif derived is not None:
            value, source = derived, "derived"
            label = {2: "Q2 = 6M - 3M", 3: "Q3 = 9M - 6M", 4: "Q4 = FY - 9M"}[q.k]
            dq.add(
                "derived_quarter",
                metric.metric.label,
                f"{q.label}: {label} = {float(ytd['value']):,.6g} ({ytd['tag']}, {ytd['accn']}) - {float(prev_ytd['value']):,.6g} "
                f"({prev_ytd['tag']}, {prev_ytd['accn']}) = {derived:,.6g}"
                + ("; components use different tags" if ytd["tag"] != prev_ytd["tag"] else ""),
                period=q.end,
                filing=ytd["accn"],
            )
        rows.append(
            {
                "fy": q.fy,
                "k": q.k,
                "q_start": q.start,
                "q_end": q.end,
                "value": value,
                "source": source,
                "ytd": None if ytd is None else float(ytd["value"]),
                "ytd_accn": None if ytd is None else ytd["accn"],
                "accn": None if reported is None else reported["accn"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["index"] = {(r["fy"], r["k"]): r for r in rows}
    return frame


def _qv_row(qv: pd.DataFrame, fy: int, k: int) -> dict | None:
    index = qv.attrs.get("index")
    if index is None:
        index = {(r["fy"], r["k"]): r for r in qv.to_dict("records")}
        qv.attrs["index"] = index
    return index.get((fy, k))


def _rel_diff(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


# ---------------------------------------------------------------- TTM

@dataclass
class TTMValue:
    value: float | None
    method: str  # "FY", "FY+YTD-PYTD", "4Q" or ""
    components: list[float | None]
    component_labels: list[str]
    alt_value: float | None
    alt_method: str
    reason: str  # why the value is blank
    sources: str


def ttm_flow(metric: MetricTable, qv: pd.DataFrame, cal: Calendar, q: Quarter, dq: DQLog | None = None) -> TTMValue:
    fy = cal.years[q.fy]
    label = metric.metric.label
    four = cal.previous_quarters(q, 4)
    quarter_sum, quarter_parts, quarter_labels = None, [], []
    if four is not None:
        for item in four:
            match = _qv_row(qv, item.fy, item.k)
            v = None if match is None or match["value"] is None or pd.isna(match["value"]) else float(match["value"])
            quarter_parts.append(v)
            quarter_labels.append(f"Q{item.k} {item.end.date()}")
        if all(v is not None for v in quarter_parts):
            quarter_sum = float(sum(quarter_parts))

    primary, method, parts, labels, sources, reason = None, "", [], [], "", ""
    if q.k == 4:
        row = metric.get(fy.start, fy.end)
        if row is not None:
            primary, method = float(row["value"]), "FY"
            parts, labels = [primary], [f"FY {fy.start.date()}..{fy.end.date()}"]
            sources = f"FY {row['accn']}"
        else:
            reason = f"{label}: no annual value for {fy.label}"
    else:
        prior = cal.years[q.fy - 1] if q.fy > 0 else None
        if prior is None or not fy.consecutive_with_prior or prior.end is None:
            reason = f"{label}: no consecutive prior fiscal year"
        else:
            fy_prev = metric.get(prior.start, prior.end)
            ytd = metric.get(fy.start, q.end)
            prior_q = cal.quarter(prior.index, q.k)
            pytd = metric.get(prior.start, prior_q.end) if prior_q is not None else None
            missing = [n for n, r in (("prior FY", fy_prev), ("current YTD", ytd), ("prior-year YTD", pytd)) if r is None]
            if not missing:
                parts = [float(fy_prev["value"]), float(ytd["value"]), float(pytd["value"])]
                primary, method = parts[0] + parts[1] - parts[2], "FY+YTD-PYTD"
                labels = [f"FY {prior.start.date()}..{prior.end.date()}", f"YTD {fy.start.date()}..{q.end.date()}", f"PYTD {prior.start.date()}..{prior_q.end.date()}"]
                sources = f"FY {fy_prev['accn']}; YTD {ytd['accn']}; PYTD {pytd['accn']}"
            else:
                reason = f"{label}: missing {', '.join(missing)}"

    alt, alt_method = None, ""
    if primary is not None:
        if method != "4Q" and quarter_sum is not None:
            alt, alt_method = quarter_sum, "4Q"
    elif quarter_sum is not None:
        primary, method, parts, labels, reason = quarter_sum, "4Q", quarter_parts, quarter_labels, ""
        sources = "sum of four quarters"
    if primary is not None and alt is not None and dq is not None and _rel_diff(primary, alt) > TTM_DISAGREE_TOL:
        dq.add(
            "ttm_method",
            label,
            f"TTM ending {q.end.date()}: {method} = {primary:,.6g} vs sum of four quarters = {alt:,.6g} ({_rel_diff(primary, alt):.2%} apart). {method} used.",
            period=q.end,
            severity="warning",
        )
    return TTMValue(primary, method, parts, labels, alt, alt_method, reason, sources)


def ttm_shares(shares: MetricTable, qv: pd.DataFrame, cal: Calendar, q: Quarter) -> TTMValue:
    four = cal.previous_quarters(q, 4)
    if four is not None:
        vals = []
        for item in four:
            match = _qv_row(qv, item.fy, item.k)
            vals.append(float(match["value"]) if match is not None and match["source"] == "reported" else None)
        if all(v is not None for v in vals):
            return TTMValue(float(np.mean(vals)), "avg4Q", vals, [f"Q{i.k} {i.end.date()}" for i in four], None, "", "", "four reported quarterly counts")
    fy_index = q.fy if q.k == 4 else q.fy - 1
    if fy_index < 0:
        return TTMValue(None, "", [], [], None, "", "no fiscal-year diluted share count", "")
    fy = cal.years[fy_index]
    if fy.end is None:
        return TTMValue(None, "", [], [], None, "", "no fiscal-year diluted share count", "")
    row = shares.get(fy.start, fy.end)
    if row is None:
        return TTMValue(None, "", [], [], None, "", f"no diluted share count for {fy.label}", "")
    return TTMValue(float(row["value"]), "FY", [float(row["value"])], [f"FY {fy.start.date()}..{fy.end.date()}"], None, "", "", f"FY count ({fy.label}, {row['accn']}): quarterly counts unavailable")


def reported_ttm_eps(eps: MetricTable, cal: Calendar, q: Quarter) -> float | None:
    """FY EPS + YTD EPS - prior-year YTD EPS, from reported values (cross-check only)."""
    fy = cal.years[q.fy]
    if q.k == 4:
        return eps.value(fy.start, fy.end)
    if q.fy == 0 or not fy.consecutive_with_prior:
        return None
    prior = cal.years[q.fy - 1]
    prior_q = cal.quarter(prior.index, q.k)
    if prior.end is None or prior_q is None:
        return None
    parts = [eps.value(prior.start, prior.end), eps.value(fy.start, q.end), eps.value(prior.start, prior_q.end)]
    if any(p is None for p in parts):
        return None
    return parts[0] + parts[1] - parts[2]

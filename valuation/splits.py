"""Stock-split detection and split-basis normalization.

Evidence comes from Input A: the same period's share count (or per-share
value) reported in two filings at a ratio close to a split factor. The split
ex-date is located in Input C when prices are not split-adjusted. Explicit
split events supplied with Input C are also honoured.

Basis convention: every share count and per-share value is restated to the
split basis in force on the as-of date ("current basis").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dq import DQLog
from .tags import METRICS, SPLIT_FACTORS, SPLIT_RATIO_TAGS

SHARE_TOL = 0.02  # share counts are reported precisely
PER_SHARE_TOL = 0.03  # EPS is rounded to cents
MIN_EPS_FOR_EVIDENCE = 0.50


@dataclass
class SplitEvent:
    factor: float  # shares multiply by this factor (2.0 for 2-for-1, 0.1 for 1-for-10)
    window_start: pd.Timestamp | None  # last filing observed on the old basis
    window_end: pd.Timestamp | None  # first filing observed on the new basis
    effective: pd.Timestamp  # date from which the new basis applies
    source: str  # how `effective` was determined
    evidence: int = 0
    examples: list[str] = field(default_factory=list)
    price_located: bool = False
    share_evidence: int = 0

    @property
    def label(self) -> str:
        if self.factor >= 1:
            return f"{_fmt_factor(self.factor)}-for-1"
        return f"1-for-{_fmt_factor(1 / self.factor)}"


def _fmt_factor(value: float) -> str:
    return f"{value:g}"


def match_factor(ratio: float, tolerance: float) -> float | None:
    if not np.isfinite(ratio) or ratio <= 0:
        return None
    if abs(ratio - 1) <= max(tolerance * 2, 0.05):
        return None
    for k in SPLIT_FACTORS:
        if abs(ratio / k - 1) <= tolerance:
            return float(k)
        if abs(ratio * k - 1) <= tolerance:
            return 1.0 / k
    return None


def _evidence_rows(facts: pd.DataFrame) -> list[tuple[float, pd.Timestamp, pd.Timestamp, str]]:
    share_tags = set(METRICS["diluted_shares"].candidates["us-gaap"]) | set(METRICS["diluted_shares"].candidates["ifrs-full"])
    share_tags |= {"WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageShares", "CommonStockSharesOutstanding"}
    per_share_tags = set(METRICS["eps_diluted"].candidates["us-gaap"]) | set(METRICS["eps_diluted"].candidates["ifrs-full"])
    per_share_tags |= {"EarningsPerShareBasic", "BasicEarningsLossPerShare"}

    events = []
    subset = facts[facts["tag"].isin(share_tags | per_share_tags) & (facts["taxonomy"] != "dei")]
    for (tag, unit, start, end), group in subset.groupby(["tag", "unit", "start", "end"], dropna=False):
        group = group.sort_values(["filed", "accn"]).drop_duplicates("accn", keep="last")
        if len(group) < 2:
            continue
        values = group["val"].to_numpy(dtype=float)
        filed = group["filed"].tolist()
        is_share = tag in share_tags
        for i in range(len(group) - 1):
            a, b = values[i], values[i + 1]
            if filed[i] == filed[i + 1] or a == 0 or b == 0 or np.sign(a) != np.sign(b):
                continue
            if is_share:
                ratio, tol = b / a, SHARE_TOL
            else:
                if min(abs(a), abs(b)) < MIN_EPS_FOR_EVIDENCE / 2:
                    continue  # cent rounding makes small per-share values unreliable evidence
                ratio, tol = a / b, PER_SHARE_TOL
            k = match_factor(ratio, tol)
            if k is None:
                continue
            period = f"{start.date() if pd.notna(start) else ''}..{end.date()}"
            events.append((k, filed[i], filed[i + 1], f"{tag} {period}: {a:g} -> {b:g}", is_share))
    return events


def _cluster(events) -> list[SplitEvent]:
    clusters: list[SplitEvent] = []
    for k, a, b, example, is_share in sorted(events, key=lambda e: (e[0], e[2], e[1])):
        placed = False
        for cluster in clusters:
            if abs(cluster.factor - k) > 1e-9:
                continue
            lo = max(cluster.window_start, a)
            hi = min(cluster.window_end, b)
            if lo < hi:
                cluster.window_start, cluster.window_end = lo, hi
                cluster.evidence += 1
                cluster.share_evidence += int(is_share)
                if len(cluster.examples) < 4:
                    cluster.examples.append(example)
                placed = True
                break
        if not placed:
            clusters.append(SplitEvent(k, a, b, b, "first filing on the new basis", 1, [example], False, int(is_share)))
    return clusters


def locate_in_prices(event: SplitEvent, raw_close: pd.Series) -> pd.Timestamp | None:
    """Find the trading day in the filing window whose close-to-close move matches 1/k."""
    if raw_close is None or raw_close.empty:
        return None
    series = raw_close.dropna()
    ratio = series / series.shift(1)
    lo = event.window_start if event.window_start is not None else series.index.min()
    hi = event.window_end if event.window_end is not None else series.index.max()
    window = ratio[(ratio.index > lo) & (ratio.index <= hi)].dropna()
    if window.empty:
        return None
    distance = (np.log(window) + math.log(event.factor)).abs()
    best = distance.idxmin()
    if distance.loc[best] < math.log(1.35):
        return best
    return None


def detect_splits(
    facts: pd.DataFrame,
    dq: DQLog,
    raw_close: pd.Series | None,
    prices_split_adjusted: bool | None,
    explicit: pd.DataFrame | None,
    as_of: pd.Timestamp,
) -> list[SplitEvent]:
    clusters = _cluster(_evidence_rows(facts))
    tagged = facts[facts["tag"].isin(SPLIT_RATIO_TAGS)].drop_duplicates(["val", "end"])
    for _, row in tagged.iterrows():
        dq.add("split", "tagged split ratio", f"{row['tag']} = {row['val']:g} (period ending {row['end'].date()}).", period=row["end"], filing=row["accn"])
    explicit_events = []
    if explicit is not None and not explicit.empty:
        for _, row in explicit.iterrows():
            explicit_events.append((pd.Timestamp(row["date"]), float(row["ratio"])))

    def corroborated(cluster: SplitEvent) -> bool:
        lo, hi = cluster.window_start, cluster.window_end + pd.Timedelta(days=5)
        if any(abs(r / cluster.factor - 1) < 0.02 and lo < d <= hi for d, r in explicit_events):
            return True
        return any(abs(v / cluster.factor - 1) < 0.02 for v in tagged["val"])

    confirmed: list[SplitEvent] = []
    for cluster in clusters:
        if corroborated(cluster) or (cluster.share_evidence >= 1 and cluster.evidence >= 2):
            confirmed.append(cluster)
            continue
        why = "per-share values only (often a restatement, not a split)" if cluster.share_evidence == 0 else "a single restated value"
        dq.add(
            "split",
            "possible split (not applied)",
            f"A {cluster.label} change is supported by {why}: {'; '.join(cluster.examples[:2])}. No share-count, tagged-ratio or Input C confirmation; not applied.",
            period=cluster.window_end,
            severity="warning",
        )
    first_filed = facts["filed"].min()

    for cluster in confirmed:
        matched = None
        for ex_date, ratio in explicit_events:
            if abs(ratio / cluster.factor - 1) < 0.02 and cluster.window_start < ex_date <= cluster.window_end + pd.Timedelta(days=5):
                matched = (ex_date, ratio)
                break
        if matched:
            cluster.effective, cluster.source, cluster.price_located = matched[0], "Input C split event", True
            explicit_events.remove(matched)
        elif prices_split_adjusted is False:
            located = locate_in_prices(cluster, raw_close)
            if located is not None:
                cluster.effective, cluster.source, cluster.price_located = located, "price move in Input C", True
        dq.add(
            "split",
            f"likely split {cluster.label}",
            (
                f"Factor {cluster.factor:g} (shares x{cluster.factor:g}, per-share /{cluster.factor:g}). "
                f"Old basis last seen in a filing on {cluster.window_start.date()}, new basis first seen {cluster.window_end.date()}. "
                f"Effective date used: {cluster.effective.date()} ({cluster.source}). Evidence: {cluster.evidence} restated values, e.g. "
                + "; ".join(cluster.examples[:3])
            ),
            period=cluster.effective,
            severity="warning",
        )

    for ex_date, ratio in explicit_events:
        if ex_date > as_of:
            continue
        before = ex_date < first_filed
        event = SplitEvent(ratio, None, None, ex_date, "Input C split event" + (" (before Input A history)" if before else " (not confirmed by filings)"), 0, [], True)
        confirmed.append(event)
        dq.add(
            "split",
            f"split from Input C {event.label}",
            (
                f"Split on {ex_date.date()} predates the first filing in Input A ({first_filed.date()}); it changes no filing value and only matters for unadjusted prices."
                if before
                else f"Split on {ex_date.date()} listed in Input C but not confirmed by restated filing values; applied as supplied."
            ),
            period=ex_date,
            severity="info" if before else "warning",
        )
    return sorted(confirmed, key=lambda e: e.effective)


class SplitBasis:
    """Cumulative split factors to restate values onto the as-of date's basis."""

    def __init__(self, events: list[SplitEvent], as_of: pd.Timestamp):
        self.events = [e for e in events if e.effective <= as_of]

    def share_factor(self, on_basis_of: pd.Timestamp) -> float:
        """Multiply a share count published on `on_basis_of` by this factor."""
        factor = 1.0
        for event in self.events:
            if event.effective > on_basis_of:
                factor *= event.factor
        return factor

    def price_factor(self, day: pd.Timestamp) -> float:
        """Divide an unadjusted price on `day` by this factor."""
        return self.share_factor(day)

    def uncertain(self, day: pd.Timestamp) -> SplitEvent | None:
        """A split whose ex-date was not located and whose filing window contains `day`."""
        for event in self.events:
            if not event.price_located and event.window_start is not None and event.window_start < day < event.window_end:
                return event
        return None

    def describe(self) -> str:
        if not self.events:
            return "no splits detected"
        return "; ".join(f"{e.label} effective {e.effective.date()} ({e.source})" for e in self.events)

    def describe_within(self, start: pd.Timestamp) -> str:
        events = [e for e in self.events if e.effective >= start]
        if not events:
            return "no splits during the filing history"
        return "; ".join(f"{e.label} effective {e.effective.date()} ({e.source})" for e in events)

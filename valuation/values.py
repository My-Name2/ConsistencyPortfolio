"""Per-metric value tables built from Input A.

For each period (start/end for durations, end for instants):
  * the value is the latest restated value filed on or before the as-of date;
  * the first filing that reported the period is kept for timing and audit;
  * share counts and per-share values are restated to the as-of split basis;
  * the first candidate tag with a value wins, other tags are logged fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dq import DQLog
from .facts import CompanyFacts
from .splits import SplitBasis
from .tags import CAPEX_COMPONENT_TAGS, CAPEX_TOTAL_TAGS, US, MetricDef

RESTATEMENT_TOL = 0.005
ANNUAL_FORMS = {"10-K", "10-K/A", "10-K405", "10-KT", "10-KT/A", "20-F", "20-F/A", "40-F", "40-F/A"}
TOL_DAYS = 3


def _unit_mask(frame: pd.DataFrame, unit_type: str) -> pd.Series:
    if unit_type == "money":
        return frame["unit"].eq(frame["dominant_currency"]) & frame["currency"].notna()
    if unit_type == "per_share":
        return frame["unit"].str.endswith("/shares") & frame["currency"].eq(frame["dominant_currency"])
    if unit_type == "shares":
        return frame["unit"].eq("shares")
    raise ValueError(unit_type)


def _normalize(values: pd.Series, filed: pd.Series, unit_type: str, basis: SplitBasis) -> pd.Series:
    if unit_type == "money":
        return values.astype(float)
    factors = filed.map(basis.share_factor).astype(float)
    if unit_type == "shares":
        return values.astype(float) * factors
    return values.astype(float) / factors


def raw_tag_table(cf: CompanyFacts, taxonomy: str, tag: str, kind: str, unit_type: str, basis: SplitBasis) -> pd.DataFrame:
    """Latest and first reported value per period for a single tag."""
    facts = cf.facts
    subset = facts[(facts["taxonomy"] == taxonomy) & (facts["tag"] == tag) & (facts["kind"] == kind)]
    if subset.empty:
        return pd.DataFrame()
    subset = subset[_unit_mask(subset, unit_type)]
    if subset.empty:
        return pd.DataFrame()
    subset = subset.sort_values(["filed", "accn"]).copy()
    subset["norm"] = _normalize(subset["val"], subset["filed"], unit_type, basis)
    keys = ["start", "end"] if kind == "duration" else ["end"]
    ignored = pd.Series(dtype=float)
    if kind == "duration":
        # A full-year period takes its value from annual reports when one reports it; a full-year
        # value that appears in a 10-Q (often a mis-dated comparative) is ignored and logged.
        days = (subset["end"] - subset["start"]).dt.days + 1
        annual = days.between(350, 380)
        from_annual_form = subset["form"].isin(ANNUAL_FORMS)
        period = subset["start"].astype(str) + "|" + subset["end"].astype(str)
        covered = set(period[annual & from_annual_form])
        drop = annual & ~from_annual_form & period.isin(covered)
        if drop.any():
            ignored = subset[drop].groupby(keys)["norm"].last()
            subset = subset[~drop]
    latest = subset.groupby(keys, dropna=False).tail(1).set_index(keys)
    first = subset.groupby(keys, dropna=False).head(1).set_index(keys)
    n_filings = subset.groupby(keys, dropna=False)["accn"].nunique()
    table = pd.DataFrame(
        {
            "value": latest["norm"],
            "value_reported": latest["val"],
            "unit": latest["unit"],
            "accn": latest["accn"],
            "filed": latest["filed"],
            "form": latest["form"],
            "first_value": first["norm"],
            "first_accn": first["accn"],
            "first_filed": first["filed"],
            "first_form": first["form"],
            "n_filings": n_filings,
            "regime_taxonomy": latest["regime_taxonomy"],
            "currency": latest["dominant_currency"],
        }
    ).reset_index()
    if kind == "duration":
        table["days"] = (table["end"] - table["start"]).dt.days + 1
    else:
        table["start"] = pd.NaT
        table["days"] = np.nan
    table["taxonomy"] = taxonomy
    table["tag"] = tag
    table["ignored_quarterly_report_value"] = np.nan
    if not ignored.empty and kind == "duration":
        lookup = {(a, b): v for (a, b), v in ignored.items()}
        table["ignored_quarterly_report_value"] = [lookup.get((a, b), np.nan) for a, b in zip(table["start"], table["end"])]
        same = (table["ignored_quarterly_report_value"] - table["value"]).abs() <= table["value"].abs() * RESTATEMENT_TOL
        table.loc[same, "ignored_quarterly_report_value"] = np.nan
    diff = (table["value"] - table["first_value"]).abs()
    scale = table[["value", "first_value"]].abs().max(axis=1).replace(0, np.nan)
    table["restated"] = (diff / scale > RESTATEMENT_TOL) & (diff > 0)
    return table


@dataclass
class MetricTable:
    metric: MetricDef
    table: pd.DataFrame  # one row per period

    def __post_init__(self) -> None:
        t = self.table
        if t.empty:
            self._start = self._end = np.array([], dtype="int64")
            self._records: list[dict] = []
            return
        self._end = pd.to_datetime(t["end"]).to_numpy(dtype="datetime64[ns]").astype("int64")
        starts = pd.to_datetime(t["start"]) if "start" in t else pd.Series(pd.NaT, index=t.index)
        self._start = starts.to_numpy(dtype="datetime64[ns]").astype("int64")
        self._records = t.to_dict("records")

    def _match(self, start, end, tol: int = TOL_DAYS) -> dict | None:
        if not self._records:
            return None
        tol_ns = tol * 86_400_000_000_000
        end_ns = pd.Timestamp(end).value
        gap = np.abs(self._end - end_ns)
        mask = gap <= tol_ns
        if self.metric.kind == "duration":
            if start is None:
                return None
            sgap = np.abs(self._start - pd.Timestamp(start).value)
            mask &= sgap <= tol_ns
            gap = gap + sgap
        hits = np.flatnonzero(mask)
        if hits.size == 0:
            return None
        best = hits[np.argmin(gap[hits])] if hits.size > 1 else hits[0]
        return self._records[best]

    def get(self, start, end, tol: int = TOL_DAYS) -> dict | None:
        return self._match(start, end, tol)

    def value(self, start, end, tol: int = TOL_DAYS) -> float | None:
        row = self._match(start, end, tol)
        return None if row is None else float(row["value"])

    def annual(self) -> pd.DataFrame:
        t = self.table
        return t[(t["days"] >= 350) & (t["days"] <= 380)] if not t.empty else t


def build_metric_table(cf: CompanyFacts, metric: MetricDef, basis: SplitBasis, dq: DQLog) -> MetricTable:
    pieces = []
    for taxonomy, tags in metric.candidates.items():
        for priority, tag in enumerate(tags):
            table = raw_tag_table(cf, taxonomy, tag, metric.kind, metric.unit, basis)
            if table.empty:
                continue
            table["priority"] = priority
            pieces.append(table)
    if not pieces:
        return MetricTable(metric, pd.DataFrame())
    combined = pd.concat(pieces, ignore_index=True)
    keys = ["start", "end"] if metric.kind == "duration" else ["end"]
    # The taxonomy of the most recent filing that reported the period wins; then tag priority.
    latest_tax = combined.sort_values("filed").groupby(keys, dropna=False)["taxonomy"].last().rename("period_taxonomy")
    combined = combined.merge(latest_tax.reset_index(), on=keys, how="left")
    combined = combined[combined["taxonomy"] == combined["period_taxonomy"]].drop(columns="period_taxonomy")
    chosen = combined.sort_values(["priority"]).groupby(keys, dropna=False).head(1)
    chosen = chosen.sort_values(keys).reset_index(drop=True)

    for (taxonomy, tag), group in chosen[chosen["priority"] > 0].groupby(["taxonomy", "tag"]):
        primary = metric.candidates[taxonomy][0]
        note = metric.notes.get(tag, "")
        dq.add(
            "fallback_tag",
            metric.label,
            f"{taxonomy}:{tag} used instead of {primary} for {len(group)} period(s) ending "
            f"{group['end'].min().date()} to {group['end'].max().date()}." + (f" Note: {note}." if note else ""),
            severity="warning" if note else "info",
        )
    if "ignored_quarterly_report_value" in chosen:
        for _, row in chosen[chosen["ignored_quarterly_report_value"].notna()].iterrows():
            dq.add(
                "restatement",
                metric.label,
                f"{row['tag']} {row['start'].date()}..{row['end'].date()}: a quarterly report shows {row['ignored_quarterly_report_value']:,.4g} for this full year; "
                f"the annual report's {row['value']:,.4g} ({row['form']} {row['accn']}) is used.",
                period=row["end"],
                filing=row["accn"],
                severity="warning",
            )
    for _, row in chosen[chosen["restated"]].iterrows():
        period = f"{row['start'].date()}..{row['end'].date()}" if pd.notna(row["start"]) else f"{row['end'].date()}"
        change = (row["value"] - row["first_value"]) / abs(row["first_value"]) if row["first_value"] else np.nan
        dq.add(
            "restatement",
            metric.label,
            f"{row['tag']} {period}: first reported {row['first_value']:,.4g} ({row['first_form']} {row['first_accn']}), "
            f"latest {row['value']:,.4g} ({row['form']} {row['accn']}); change {change:+.2%}. Latest value used.",
            period=row["end"],
            filing=row["accn"],
        )
    return MetricTable(metric, chosen)


def derive_gross_profit(gp: MetricTable, revenue: MetricTable, cost: MetricTable, dq: DQLog) -> MetricTable:
    """Fill gross profit as revenue minus cost of revenue where GrossProfit is not tagged."""
    if revenue.table.empty or cost.table.empty:
        return gp
    merged = revenue.table.merge(cost.table, on=["start", "end"], suffixes=("_rev", "_cor"))
    existing = set()
    if not gp.table.empty:
        existing = {(s, e) for s, e in zip(gp.table["start"], gp.table["end"])}
    rows = []
    for _, row in merged.iterrows():
        if (row["start"], row["end"]) in existing:
            continue
        rows.append(
            {
                "start": row["start"],
                "end": row["end"],
                "value": row["value_rev"] - row["value_cor"],
                "value_reported": np.nan,
                "unit": row["unit_rev"],
                "accn": row["accn_rev"],
                "filed": max(row["filed_rev"], row["filed_cor"]),
                "form": row["form_rev"],
                "first_value": row["first_value_rev"] - row["first_value_cor"],
                "first_accn": row["first_accn_rev"],
                "first_filed": max(row["first_filed_rev"], row["first_filed_cor"]),
                "first_form": row["first_form_rev"],
                "n_filings": row["n_filings_rev"],
                "regime_taxonomy": row["regime_taxonomy_rev"],
                "currency": row["currency_rev"],
                "days": row["days_rev"],
                "taxonomy": row["taxonomy_rev"],
                "tag": f"derived: {row['tag_rev']} - {row['tag_cor']}",
                "restated": bool(row["restated_rev"] or row["restated_cor"]),
                "priority": 99,
            }
        )
    if not rows:
        return gp
    derived = pd.DataFrame(rows)
    dq.add(
        "derived_value",
        "Gross profit",
        f"GrossProfit not tagged for {len(derived)} period(s) ending {derived['end'].min().date()} to "
        f"{derived['end'].max().date()}; computed as revenue minus cost of revenue from the same periods.",
        severity="warning",
    )
    table = pd.concat([gp.table, derived], ignore_index=True) if not gp.table.empty else derived
    return MetricTable(gp.metric, table.sort_values(["start", "end"]).reset_index(drop=True))


def parent_share_of_profit_loss(ni: MetricTable, nci: MetricTable, dq: DQLog) -> MetricTable:
    """Where ProfitLoss (which includes NCI) was the only net income tag, subtract tagged NCI income."""
    t = ni.table
    if t.empty or not (t["tag"] == "ProfitLoss").any():
        return ni
    t = t.copy()
    adjusted = 0
    for i in t.index[t["tag"] == "ProfitLoss"]:
        row = nci.get(t.at[i, "start"], t.at[i, "end"], tol=0)
        if row is not None and row["value"]:
            t.at[i, "value"] = float(t.at[i, "value"]) - float(row["value"])
            t.at[i, "first_value"] = float(t.at[i, "first_value"]) - float(row["value"])
            t.at[i, "tag"] = "derived: ProfitLoss - NetIncomeLossAttributableToNoncontrollingInterest"
            adjusted += 1
    if adjusted:
        dq.add("derived_value", ni.metric.label, f"{adjusted} period(s): NI attributable to parent = ProfitLoss - NCI income (both tagged for the same period).", severity="warning")
    return MetricTable(ni.metric, t)


def parent_equity(equity: MetricTable, nci: MetricTable, dq: DQLog) -> MetricTable:
    """Where only total equity (including NCI) is tagged, subtract NCI tagged at the same instant."""
    t = equity.table
    total_tags = {"StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "Equity"}
    if t.empty or not t["tag"].isin(total_tags).any():
        return equity
    t = t.copy()
    adjusted = 0
    for i in t.index[t["tag"].isin(total_tags)]:
        row = nci.get(None, t.at[i, "end"], tol=0)
        if row is not None and row["value"]:
            t.at[i, "value"] = float(t.at[i, "value"]) - float(row["value"])
            t.at[i, "first_value"] = float(t.at[i, "first_value"]) - float(row["value"])
            t.at[i, "tag"] = f"derived: {t.at[i, 'tag']} - {row['tag']}"
            adjusted += 1
    if adjusted:
        dq.add("derived_value", equity.metric.label, f"{adjusted} instant(s): parent equity = total equity - noncontrolling interest (both tagged at the same date).", severity="warning")
    return MetricTable(equity.metric, t)


def lessor_revenue(revenue: MetricTable, lease: MetricTable, dq: DQLog) -> MetricTable:
    """Total revenue for lessors that tag contract revenue and lease income separately but no total."""
    t = revenue.table
    if t.empty or lease.table.empty:
        return revenue
    t = t.copy()
    added = 0
    for i in t.index[t["tag"].astype(str).str.startswith("RevenueFromContractWithCustomer")]:
        row = lease.get(t.at[i, "start"], t.at[i, "end"], tol=0)
        # Only when lease income exceeds contract revenue can contract revenue not already be the total
        # (many lessors tag total revenue, lease income included, under the contract-revenue tag).
        if row is not None and row["value"] and float(row["value"]) > float(t.at[i, "value"]):
            t.at[i, "value"] = float(t.at[i, "value"]) + float(row["value"])
            t.at[i, "first_value"] = float(t.at[i, "first_value"]) + float(row["value"])
            t.at[i, "tag"] = f"derived: {t.at[i, 'tag']} + {row['tag']}"
            added += 1
    if added:
        dq.add(
            "derived_value",
            revenue.metric.label,
            f"{added} period(s): total revenue not tagged and lease income exceeds contract revenue; revenue = contract revenue + operating lease income for the same period (lessor presentation).",
            severity="warning",
        )
    return MetricTable(revenue.metric, t)


def capex_components(capex: MetricTable, components: dict[str, MetricTable], dq: DQLog) -> MetricTable:
    """Capex for companies that split PP&E purchases over several cash-flow lines.

    `components` maps each tag in CAPEX_COMPONENT_TAGS to its own single-tag table. When the line chosen
    for a period is a component, the other component lines of the exact same period are added. When it
    is a total tag but construction expenditures of the same period exceed it, the "total" tag holds a
    smaller separate line (e.g. a utility's asset acquisitions) and the component lines are used instead.
    """
    t = capex.table
    if t.empty or not components:
        return capex
    t = t.copy()
    summed, replaced = 0, 0
    construction = "PaymentsForConstructionInProcess"
    for i in t.index:
        tag = t.at[i, "tag"]
        if t.at[i, "taxonomy"] != US or (tag not in CAPEX_COMPONENT_TAGS and tag not in CAPEX_TOTAL_TAGS):
            continue
        start, end = t.at[i, "start"], t.at[i, "end"]
        parts = {}
        for ctag, table in components.items():
            row = table.get(start, end, tol=0)
            if row is not None and row["value"]:
                parts[ctag] = float(row["value"])
        if tag in CAPEX_COMPONENT_TAGS:
            extra = {k: v for k, v in parts.items() if k != tag}
            if not extra:
                continue
            t.at[i, "value"] = float(t.at[i, "value"]) + sum(extra.values())
            t.at[i, "first_value"] = float(t.at[i, "first_value"]) + sum(extra.values())
            t.at[i, "tag"] = "derived: " + " + ".join([tag, *extra])
            summed += 1
        elif construction in parts and parts[construction] > float(t.at[i, "value"]):
            t.at[i, "value"] = sum(parts.values())
            t.at[i, "first_value"] = sum(parts.values())
            t.at[i, "tag"] = "derived: " + " + ".join(parts) + f" (instead of {tag})"
            replaced += 1
    if summed:
        dq.add(
            "derived_value",
            capex.metric.label,
            f"{summed} period(s): PP&E purchases are split over several cash-flow lines ({', '.join(CAPEX_COMPONENT_TAGS)}); "
            "capex = sum of the lines tagged for the same period.",
            severity="warning",
        )
    if replaced:
        dq.add(
            "derived_value",
            capex.metric.label,
            f"{replaced} period(s): construction expenditures (PaymentsForConstructionInProcess) exceed the value tagged as total PP&E purchases, "
            "so that tag holds a smaller separate line; capex = the construction/PP&E component lines of the same period.",
            severity="warning",
        )
    return MetricTable(capex.metric, t)

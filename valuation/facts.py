"""Load and normalize SEC companyfacts JSON (Input A)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .tags import CALENDAR_METRICS, DEI, IFRS, METRICS, US


@dataclass
class CompanyFacts:
    cik: str
    entity_name: str
    facts: pd.DataFrame  # one row per reported fact, filed on or before the as-of date
    filings: pd.DataFrame  # one row per accession
    excluded_after_asof: int  # facts dropped because they were filed after the as-of date
    taxonomies: list[str]


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def read_companyfacts(source) -> dict:
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        return json.loads(Path(source).read_text(encoding="utf-8"))
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
    raise TypeError("Unsupported companyfacts source")


def load_companyfacts(source, as_of: date) -> CompanyFacts:
    raw = read_companyfacts(source)
    rows = []
    for taxonomy, concepts in (raw.get("facts") or {}).items():
        for tag, payload in concepts.items():
            for unit, items in (payload.get("units") or {}).items():
                for item in items:
                    rows.append(
                        (
                            taxonomy,
                            tag,
                            unit,
                            item.get("start"),
                            item.get("end"),
                            item.get("val"),
                            item.get("accn"),
                            item.get("fy"),
                            item.get("fp"),
                            item.get("form"),
                            item.get("filed"),
                            item.get("frame"),
                        )
                    )
    columns = ["taxonomy", "tag", "unit", "start", "end", "val", "accn", "fy", "fp", "form", "filed", "frame"]
    facts = pd.DataFrame(rows, columns=columns)
    if facts.empty:
        raise ValueError("Input A contains no facts.")
    facts["start"] = _to_date(facts["start"])
    facts["end"] = _to_date(facts["end"])
    facts["filed"] = _to_date(facts["filed"])
    facts["val"] = pd.to_numeric(facts["val"], errors="coerce")
    facts = facts.dropna(subset=["end", "filed", "val", "accn"]).copy()
    facts["form"] = facts["form"].fillna("").astype(str)
    facts["kind"] = np.where(facts["start"].isna(), "instant", "duration")
    # Inclusive day count: a 52-week year is 364 days, a 53-week year 371.
    facts["days"] = (facts["end"] - facts["start"]).dt.days + 1

    as_of_ts = pd.Timestamp(as_of)
    excluded = int((facts["filed"] > as_of_ts).sum())
    facts = facts[facts["filed"] <= as_of_ts].copy()
    if facts.empty:
        raise ValueError("Input A has no facts filed on or before the as-of date.")

    facts["currency"] = facts["unit"].map(_unit_currency)
    filings = _build_filings(facts)
    facts = facts.merge(filings[["accn", "dominant_currency", "regime_taxonomy"]], on="accn", how="left")
    cik = str(raw.get("cik", "")).zfill(10)
    taxonomies = sorted(set(facts["taxonomy"]) & {US, IFRS})
    return CompanyFacts(cik, str(raw.get("entityName", "")), facts, filings, excluded, taxonomies)


def _unit_currency(unit: str) -> str | None:
    """'USD' -> 'USD'; 'USD/shares' -> 'USD'; 'shares' -> None."""
    if not isinstance(unit, str):
        return None
    head = unit.split("/")[0]
    if len(head) == 3 and head.isalpha() and head.isupper():
        return head
    return None


def _calendar_tags() -> set[tuple[str, str]]:
    tags = set()
    for key in CALENDAR_METRICS:
        for taxonomy, names in METRICS[key].candidates.items():
            tags.update((taxonomy, name) for name in names)
    return tags


def _build_filings(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per accession: form, filed date, regime and report period."""
    base = (
        facts.sort_values("filed")
        .groupby("accn", as_index=False)
        .agg(form=("form", "first"), filed=("filed", "min"))
    )
    money = facts[facts["currency"].notna() & facts["taxonomy"].isin([US, IFRS])]
    dominant = (
        money.groupby(["accn", "currency"]).size().reset_index(name="n").sort_values(["accn", "n"], ascending=[True, False])
        .drop_duplicates("accn")[["accn", "currency"]]
        .rename(columns={"currency": "dominant_currency"})
    )
    tax_counts = facts[facts["taxonomy"].isin([US, IFRS])].groupby(["accn", "taxonomy"]).size().unstack(fill_value=0)
    regime_tax = pd.Series(US, index=tax_counts.index, name="regime_taxonomy")
    if IFRS in tax_counts.columns:
        us_counts = tax_counts[US] if US in tax_counts.columns else 0
        regime_tax = regime_tax.where(~(tax_counts[IFRS] > us_counts), IFRS)
    regime_tax = regime_tax.reset_index()

    calendar_tags = _calendar_tags()
    keyed = facts[(facts["taxonomy"] + ":" + facts["tag"]).isin({f"{t}:{g}" for t, g in calendar_tags})]
    durations = keyed[keyed["kind"] == "duration"]
    report_period = durations.groupby("accn")["end"].max().rename("report_period").reset_index()
    filings = base.merge(dominant, on="accn", how="left").merge(regime_tax, on="accn", how="left").merge(report_period, on="accn", how="left")
    filings["regime_taxonomy"] = filings["regime_taxonomy"].fillna(US)
    filings["is_amendment"] = filings["form"].str.endswith("/A")
    return filings.sort_values(["filed", "accn"]).reset_index(drop=True)

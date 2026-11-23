"""Data-quality log shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CATEGORIES = (
    "input",
    "coverage",
    "fallback_tag",
    "derived_value",
    "restatement",
    "split",
    "amendment",
    "derived_quarter",
    "quarter_mismatch",
    "ttm_method",
    "eps_crosscheck",
    "shares_method",
    "period_label",
    "fiscal_calendar",
    "taxonomy_change",
    "currency_change",
    "missing_period",
    "price_gap",
    "stale_price",
    "price_basis",
    "ev_choice",
    "extreme_multiple",
    "assumption",
)


@dataclass
class DQLog:
    rows: list[dict] = field(default_factory=list)

    def add(self, category: str, item: str, detail: str, period=None, filing: str | None = None, severity: str = "info") -> None:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown data-quality category {category!r}")
        if isinstance(period, pd.Timestamp):
            period = period.date().isoformat()
        self.rows.append(
            {
                "category": category,
                "severity": severity,
                "item": item,
                "period": "" if period is None else str(period),
                "detail": detail,
                "filing": filing or "",
            }
        )

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["category", "severity", "item", "period", "detail", "filing"])
        order = {name: index for index, name in enumerate(CATEGORIES)}
        frame = pd.DataFrame(self.rows)
        frame["_order"] = frame["category"].map(order)
        return frame.sort_values(["_order", "item", "period"], kind="stable").drop(columns="_order").reset_index(drop=True)

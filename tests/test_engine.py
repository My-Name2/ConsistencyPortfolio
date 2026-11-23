"""Rule checks against the SYNTHETIC fixture (no network, nothing written to disk).

Run:  python -B -m pytest -q tests
Set VALUATION_TEST_RECALC=1 (and have LibreOffice) to also recalculate the
workbook and compare every formula result with the Python engine; that test
uses pytest's temporary folder, so point --basetemp off the C: drive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import fixture  # noqa: E402
from valuation.pipeline import ReviewInputs, run_review  # noqa: E402
from valuation.prices import PriceMeta  # noqa: E402
from valuation.report import build_report  # noqa: E402
from valuation.sources import parse_ticker_list  # noqa: E402
from valuation.stats import compute_stats  # noqa: E402
from valuation.storage import StorageRefused, is_c_drive, output_dir  # noqa: E402
from valuation.workbook import build_workbook  # noqa: E402


@pytest.fixture(scope="module")
def review():
    meta = PriceMeta(ticker="SYNT", split_adjusted=False, dividend_adjusted=False, currency="USD")
    return run_review(ReviewInputs(facts=fixture.companyfacts(), as_of=fixture.AS_OF, prices=fixture.prices(), price_meta=meta, ticker="SYNT"))


def row_for(review, quarter_end: str) -> pd.Series:
    t = review.ttm
    return t[t["quarter_end"] == pd.Timestamp(quarter_end)].iloc[0]


def test_valuation_dates_are_first_filing_dates(review):
    r = row_for(review, "2024-06-30")
    assert r["valuation_date"] == pd.Timestamp(fixture.filed_date(2024, 2))
    assert r["form"] == "10-Q"
    assert row_for(review, "2023-12-31")["form"] == "10-K"
    assert review.ttm["valuation_date"].is_monotonic_increasing


def test_ttm_flow_matches_four_quarters(review):
    r = row_for(review, "2024-06-30")
    quarters = [(2023, 3), (2023, 4), (2024, 1), (2024, 2)]
    assert r["ttm_revenue"] == pytest.approx(sum(fixture.revenue(*q) for q in quarters))
    assert r["ttm_revenue_method"] == "FY+YTD-PYTD"
    assert r["ttm_ocf"] == pytest.approx(sum(fixture.ocf(*q) for q in quarters))
    assert r["ttm_fcf"] == pytest.approx(r["ttm_ocf"] - 4 * fixture.CAPEX)
    assert r["ttm_ebitda"] == pytest.approx(sum(fixture.operating_income(*q) for q in quarters) + 4 * fixture.DA)
    fy = row_for(review, "2023-12-31")
    assert fy["ttm_revenue_method"] == "FY"


def test_derived_quarters_logged_and_correct(review):
    dq = review.dq.frame()
    derived = dq[(dq["category"] == "derived_quarter") & (dq["item"] == "Operating cash flow")]
    assert any("Q4 = FY - 9M" in d for d in derived["detail"])
    from valuation.periods import quarterly_values
    from valuation.dq import DQLog

    qv = quarterly_values(review.tables["ocf"], review.calendar, DQLog())
    q4_2023 = qv[(qv["q_end"] == pd.Timestamp("2023-12-31"))].iloc[0]
    assert q4_2023["source"] == "derived"
    assert q4_2023["value"] == pytest.approx(fixture.ocf(2023, 4))


def test_split_detected_and_normalized(review):
    events = review.basis.events
    assert len(events) == 1 and events[0].factor == 2.0
    assert events[0].effective == fixture.SPLIT_DATE  # located from the price move in unadjusted Input C
    pre = row_for(review, "2021-12-31")  # filed before the split
    assert pre["shares_out"] == pytest.approx(2 * fixture.PRE_SPLIT_SHARES)
    assert pre["avg_diluted_shares"] == pytest.approx(2 * fixture.PRE_SPLIT_SHARES)
    eps = review.ttm.set_index("quarter_end")["eps_ttm"]
    # No artificial halving across the split: EPS moves only with net income.
    before, after = eps[pd.Timestamp("2022-03-31")], eps[pd.Timestamp("2022-06-30")]
    assert after == pytest.approx(row_for(review, "2022-06-30")["ttm_net_income"] / (2 * fixture.PRE_SPLIT_SHARES))
    assert 0.9 < after / before < 1.1


def test_prices_split_adjusted_and_held_fundamentals(review):
    daily = review.daily
    day = pd.Timestamp("2021-07-01")
    assert daily.loc[day, "raw_close"] == pytest.approx(2 * daily.loc[day, "price"])
    held = row_for(review, "2021-03-31")  # valuation date 2021-05-10 is the latest on or before 2021-07-01
    assert daily.loc[day, "valuation_date"] == held["valuation_date"]
    assert daily.loc[day, "eps_ttm"] == pytest.approx(held["eps_ttm"])
    assert daily.loc[day, "mcap"] == pytest.approx(daily.loc[day, "price"] * held["shares_out"])


def test_negative_eps_gives_nm_and_adjusted_exclusions(review):
    r = row_for(review, "2020-09-30")
    assert r["eps_ttm"] < 0
    from valuation.pipeline import ratio_inputs
    from valuation.ratios import compute_all

    values = compute_all(ratio_inputs(r))
    assert values["pe"][0] == "NM"
    assert values["payout"][0] == "NM"
    stats = review.stats.set_index("series")
    negatives = int((review.ttm["eps_ttm"] <= 0).sum())
    assert negatives == 4
    assert stats.at["eps_ttm", "adj_excluded"] == negatives
    assert stats.at["eps_ttm", "adj_sign_changes"] == 2


def test_dividends_ttm_from_ex_dates(review):
    assert row_for(review, "2023-12-31")["dps_ttm"] == pytest.approx(0.40)
    assert row_for(review, "2022-03-31")["dps_ttm"] == pytest.approx(0.40)  # pre-split 0.20 raw = 0.10 on the as-of basis


def test_ev_components(review):
    r = row_for(review, "2024-06-30")
    assert r["total_borrowings"] == pytest.approx(1_000_000.0)
    assert r["ev"] == pytest.approx(r["mcap"] + 1_000_000.0 - r["cash"])
    assert r["tbv"] == pytest.approx(r["equity"] - 200_000.0)


def test_compute_stats_hand_checked():
    dates = list(range(9))
    s = compute_stats(dates, [-2, -1, None, 3, 5, 4, 0, 6, -1])
    assert (s["raw_n"], s["raw_comparisons"], s["raw_increases"], s["raw_decreases"], s["raw_unchanged"]) == (8, 6, 3, 3, 0)
    assert s["raw_record_highs"] == 4 and s["raw_latest_record"] == "no" and s["raw_latest_rank"] == 6
    assert (s["adj_n"], s["adj_comparisons"], s["adj_pairs_skipped"], s["adj_increases"], s["adj_decreases"]) == (4, 2, 4, 1, 1)
    assert s["adj_record_highs"] == 2 and s["adj_latest_record"] == "excluded (≤ 0)"
    assert s["adj_excluded"] == 4 and s["adj_sign_changes"] == 1
    assert s["prior"] == 6 and s["pct_change"] == pytest.approx(-7 / 6)
    assert (s["min"], s["min_date"], s["max"], s["max_date"]) == (-2, 0, 6, 7)


def test_report_has_all_sections(review):
    text = build_report(review)
    for n in range(1, 11):
        assert f"## {n}." in text
    assert "SYNTHETIC" in text


def test_workbook_sheets(review):
    from openpyxl import load_workbook
    import io

    wb = load_workbook(io.BytesIO(build_workbook(review)))
    assert wb.sheetnames[:6] == ["Snapshot", "TTM_Quarterly", "TTM_Stats", "Ratios_Quarterly", "Valuation_Inputs", "Data_Quality"]
    assert str(wb["TTM_Stats"]["E5"].value).startswith("=COUNT(")


@pytest.mark.skipif(os.environ.get("VALUATION_TEST_RECALC") != "1" or shutil.which("soffice") is None, reason="set VALUATION_TEST_RECALC=1 with LibreOffice installed")
def test_workbook_formulas_match_python(review, tmp_path):
    from valuation.verify import compare

    src = tmp_path / "synt.xlsx"
    src.write_bytes(build_workbook(review))
    out = tmp_path / "out"
    subprocess.run(["soffice", "--headless", f"-env:UserInstallation={(tmp_path / 'lo').as_uri()}", "--convert-to", "xlsx", "--outdir", str(out), str(src)], check=True, capture_output=True, timeout=300)
    problems = compare(review, (out / "synt.xlsx").read_bytes())
    assert problems == []


def test_storage_refuses_c_drive():
    assert is_c_drive("C:\\Users\\me\\out") and is_c_drive("c:/x") and is_c_drive("/mnt/c/Users/me")
    assert not is_c_drive("D:\\valuation") and not is_c_drive("E:/x")
    with pytest.raises(StorageRefused):
        output_dir("C:\\valuation")
    assert output_dir(None) is None


def test_ticker_list_parsing():
    assert parse_ticker_list("'TTD','CHTR', 'smci'\nAAPL # comment\nTTD") == ["TTD", "CHTR", "SMCI", "AAPL"]


# ---------------------------------------------------------------- regressions found by comparing with a reference dataset

def _review_with(mutate):
    facts = fixture.companyfacts()
    mutate(facts["facts"])
    meta = PriceMeta(ticker="SYNT", split_adjusted=False, dividend_adjusted=False, currency="USD")
    return run_review(ReviewInputs(facts=facts, as_of=fixture.AS_OF, prices=fixture.prices(), price_meta=meta, ticker="SYNT"))


def test_full_year_value_in_a_10q_does_not_override_the_10k():
    def mutate(f):
        f["us-gaap"]["Revenues"]["units"]["USD"].append(
            {"start": "2023-01-01", "end": "2023-12-31", "val": 1.0, "accn": "0000000001-24-999999", "fy": 2024, "fp": "Q1", "form": "10-Q", "filed": "2024-05-20"}
        )

    r = _review_with(mutate)
    assert row_for(r, "2023-12-31")["ttm_revenue"] == pytest.approx(sum(fixture.revenue(2023, k) for k in (1, 2, 3, 4)))
    dq = r.dq.frame()
    assert dq["detail"].str.contains("a quarterly report shows 1 for this full year").any()


def test_misdated_quarter_is_excluded():
    def mutate(f):
        for item in f["us-gaap"]["Revenues"]["units"]["USD"]:
            if item.get("start") == "2024-04-01" and item["end"] == "2024-06-30":
                item["val"] = 4 * 4 * fixture.revenue(2024, 2)  # a full-year-sized number with quarter dates

    r = _review_with(mutate)
    dq = r.dq.frame()
    assert dq["detail"].str.contains("mis-dated fact").any()
    # The TTM uses FY + YTD - prior YTD, so it is unaffected by the bad quarter.
    assert row_for(r, "2024-06-30")["ttm_revenue"] == pytest.approx(sum(fixture.revenue(*q) for q in [(2023, 3), (2023, 4), (2024, 1), (2024, 2)]))


def test_placeholder_cover_shares_fall_back_to_weighted_basic():
    def mutate(f):
        diluted = f["us-gaap"]["WeightedAverageNumberOfDilutedSharesOutstanding"]
        f["us-gaap"]["WeightedAverageNumberOfSharesOutstandingBasic"] = {"label": "basic", "units": {"shares": [dict(i) for i in diluted["units"]["shares"]]}}
        for item in f["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]:
            item["val"] = 1.0

    r = _review_with(mutate)
    snap = r.snapshot_inputs
    assert snap["shares_out"] == pytest.approx(2 * fixture.PRE_SPLIT_SHARES)
    assert "weighted basic" in snap["shares_source"]


def test_lessor_revenue_adds_lease_income_only_when_it_exceeds_contract_revenue():
    from valuation.dq import DQLog
    from valuation.tags import METRICS
    from valuation.values import MetricTable, lessor_revenue

    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")

    def table(tag, value):
        return pd.DataFrame([{"start": start, "end": end, "days": 365, "value": value, "first_value": value, "tag": tag, "accn": "x"}])

    contract = "RevenueFromContractWithCustomerExcludingAssessedTax"
    lease = MetricTable(METRICS["lease_income"], table("OperatingLeaseLeaseIncome", 1_500.0))
    small = lessor_revenue(MetricTable(METRICS["revenue"], table(contract, 13.0)), lease, DQLog())
    assert small.value(start, end) == pytest.approx(1_513.0)  # REIT-style: contract revenue is a small component
    total = lessor_revenue(MetricTable(METRICS["revenue"], table(contract, 2_800.0)), lease, DQLog())
    assert total.value(start, end) == pytest.approx(2_800.0)  # contract tag already holds total revenue


def _move_capex(f, new_tag, share):
    """Re-tag a share of every PP&E capex fact as a second cash-flow line (new_tag)."""
    items = f["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]["units"]["USD"]
    f["us-gaap"][new_tag] = {"label": new_tag, "units": {"USD": [dict(i, val=i["val"] * share) for i in items]}}
    return items


def test_capex_split_over_component_lines_is_summed():
    # Oil-and-gas style presentation: no PaymentsToAcquirePropertyPlantAndEquipment, two component lines.
    def mutate(f):
        items = _move_capex(f, "PaymentsToAcquireOtherPropertyPlantAndEquipment", 0.25)
        f["us-gaap"]["PaymentsToAcquireOilAndGasPropertyAndEquipment"] = {"label": "og", "units": {"USD": [dict(i, val=i["val"] * 0.75) for i in items]}}
        del f["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]

    r = _review_with(mutate)
    row = row_for(r, "2024-06-30")
    assert row["ttm_capex"] == pytest.approx(4 * fixture.CAPEX)
    assert row["ttm_fcf"] == pytest.approx(row["ttm_ocf"] - 4 * fixture.CAPEX)
    assert r.dq.frame()["detail"].str.contains("split over several cash-flow lines").any()


def test_other_ppe_line_alone_is_capex():
    def mutate(f):
        _move_capex(f, "PaymentsToAcquireOtherPropertyPlantAndEquipment", 1.0)
        del f["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]

    r = _review_with(mutate)
    assert row_for(r, "2024-06-30")["ttm_capex"] == pytest.approx(4 * fixture.CAPEX)
    assert r.dq.frame()["detail"].str.contains("PaymentsToAcquireOtherPropertyPlantAndEquipment used instead of PaymentsToAcquirePropertyPlantAndEquipment").any()


def test_utility_construction_expenditures_replace_a_smaller_total_tag():
    # Utility presentation: "construction expenditures" is the capex line; the PP&E tag holds small asset acquisitions.
    def mutate(f):
        items = _move_capex(f, "PaymentsForConstructionInProcess", 1.0)
        for i in items:
            i["val"] = i["val"] * 0.01

    r = _review_with(mutate)
    assert row_for(r, "2024-06-30")["ttm_capex"] == pytest.approx(4 * fixture.CAPEX)


def test_construction_line_smaller_than_total_is_not_added():
    # A construction-in-process disclosure below the PP&E total may be a breakdown of it: the total stands.
    def mutate(f):
        _move_capex(f, "PaymentsForConstructionInProcess", 0.4)

    r = _review_with(mutate)
    assert row_for(r, "2024-06-30")["ttm_capex"] == pytest.approx(4 * fixture.CAPEX)


def test_furniture_and_fixtures_line_alone_is_capex():
    # Asset managers (e.g. KKR) tag their only PP&E purchase line as furniture and fixtures.
    def mutate(f):
        _move_capex(f, "PaymentsToAcquireFurnitureAndFixtures", 1.0)
        del f["us-gaap"]["PaymentsToAcquirePropertyPlantAndEquipment"]

    r = _review_with(mutate)
    assert row_for(r, "2024-06-30")["ttm_capex"] == pytest.approx(4 * fixture.CAPEX)

def _calendar_from(periods):
    from valuation.dq import DQLog
    from valuation.periods import build_calendar
    from valuation.tags import METRICS
    from valuation.values import MetricTable

    rows = []
    for s, e in periods:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        rows.append({"start": s, "end": e, "days": (e - s).days + 1, "value": 1.0, "first_value": 1.0, "tag": "Revenues", "accn": "x"})
    return build_calendar([MetricTable(METRICS["revenue"], pd.DataFrame(rows))], DQLog(), pd.Timestamp("2026-01-01"))


def test_switch_to_52_53_week_years_keeps_the_following_year():
    # FY2016 first reported on month ends, later restated to end a day earlier; FY2017 starts 2016-10-31
    # (Deere 2016-2017). The one-day overlap must not drop FY2017 and its quarters.
    cal = _calendar_from([
        ("2015-11-01", "2016-01-31"), ("2015-11-01", "2016-04-30"), ("2015-11-01", "2016-07-31"), ("2015-11-01", "2016-10-31"),
        ("2015-11-02", "2016-10-30"),
        ("2016-10-31", "2017-01-29"), ("2016-10-31", "2017-04-30"), ("2016-10-31", "2017-07-30"), ("2016-10-31", "2017-10-29"),
        ("2017-10-30", "2018-01-28"),
    ])
    ends = [q.end for q in cal.quarters]
    assert pd.Timestamp("2017-01-29") in ends and pd.Timestamp("2017-10-29") in ends
    q1_2018 = next(q for q in cal.quarters if q.end == pd.Timestamp("2018-01-28"))
    chain = cal.previous_quarters(q1_2018, 4)
    assert chain is not None and chain[0].end == pd.Timestamp("2017-04-30")


def test_four_quarter_chain_runs_through_a_transition_period():
    # Fiscal year end moved from June to December: a six-month transition period Jul-Dec 2012 (ADM).
    cal = _calendar_from([
        ("2011-07-01", "2011-09-30"), ("2011-07-01", "2011-12-31"), ("2011-07-01", "2012-03-31"), ("2011-07-01", "2012-06-30"),
        ("2012-07-01", "2012-09-30"), ("2012-07-01", "2012-12-31"), ("2012-10-01", "2012-12-31"),
        ("2013-01-01", "2013-03-31"), ("2013-01-01", "2013-06-30"), ("2013-01-01", "2013-09-30"), ("2013-01-01", "2013-12-31"),
    ])
    ends = [q.end for q in cal.quarters]
    assert ends.count(pd.Timestamp("2013-03-31")) == 1  # the transition period does not borrow FY2013's Q1
    q2_2013 = next(q for q in cal.quarters if q.end == pd.Timestamp("2013-06-30"))
    chain = cal.previous_quarters(q2_2013, 4)
    assert chain is not None
    assert [q.end.date().isoformat() for q in chain] == ["2012-09-30", "2012-12-31", "2013-03-31", "2013-06-30"]

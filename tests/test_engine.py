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

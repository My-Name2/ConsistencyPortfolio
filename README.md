# ConsistencyPortfolio

Two Streamlit apps:

- `streamlit_app.py` / `etf_dashboard.py`: the synthetic ETF dashboard.
- `valuation_app.py`: the SEC-filing valuation and fundamentals review (below).

## Valuation and fundamentals review

For each company it builds TTM series, valuation ratios, RAW/ADJUSTED trend statistics, a
data-quality log and red flags from:

| Input | Source | Notes |
|---|---|---|
| A | SEC companyfacts JSON (`data.sec.gov`), or an uploaded file | us-gaap or ifrs-full |
| B | Optional 10-K/20-F excerpts (uploaded text/HTML) | going concern, auditor opinion, material weakness, auditor change |
| C | Yahoo Finance chart API daily history, or an uploaded CSV | fetched close is split-adjusted, not dividend-adjusted; dividends by ex-date |
| D | As-of date, optional current price | default: last close on or before the as-of date |

Every number comes from these inputs. Facts filed after the as-of date are ignored.

### No disk use by default

- The GUI keeps fetched inputs in memory. The workbook, report and daily CSV are built in memory
  and downloaded through the browser.
- The command line writes files only to `--out-dir` (or `VALUATION_OUTPUT_DIR`), and refuses any
  folder on the **C: drive**. Without an output folder it prints to the console and writes nothing.
- Launch commands use `python -B` so no `__pycache__` folders are created. `pytest.ini` turns off
  the pytest cache.

### Run the GUI

```bash
pip install -r requirements.txt
python -B -m streamlit run valuation_app.py --server.headless true --browser.gatherUsageStats false
```

Then open http://localhost:8501. The sidebar loads the default ticker list
(`valuation/default_tickers.txt`). Use **Ticker list** to upload or paste a different list, pick a
company, or type any ticker or CIK.

### Command line

```bash
# one company, console only (nothing written)
python -B -m valuation.cli review AAPL --as-of 2026-11-23
# one company, workbook + report written to D:
python -B -m valuation.cli review AAPL --as-of 2026-11-23 --out-dir D:\valuation
# the whole default list (or --tickers my_list.txt), each company reviewed separately
python -B -m valuation.cli batch --as-of 2026-11-23 --out-dir D:\valuation --workers 4
# local input files instead of fetching
python -B -m valuation.cli review --facts CIK0000320193.json --prices prices.csv \
    --split-adjusted yes --dividend-adjusted no --currency USD --as-of 2026-11-23
```

SEC asks automated clients to identify themselves. Set `SEC_USER_AGENT="Your Name you@example.com"`;
the default is a generic client string.

### Workbook

`Snapshot`, `TTM_Quarterly`, `TTM_Stats`, `Ratios_Quarterly`, `Valuation_Inputs` and
`Data_Quality`, plus `TTM_Build` (the reported components behind each TTM value) and
`Stats_Helpers` (running counts and record-high flags). Ratios and statistics are live formulas.
Daily values are not stored in the workbook.

### Tests

```bash
python -B -m pytest -q
# also recalculate the workbook in LibreOffice and compare every formula with Python
VALUATION_TEST_RECALC=1 python -B -m pytest -q --basetemp D:\tmp\pytest
```

The tests use a synthetic fixture (`tests/fixture.py`), not real company data.

# Synthetic ETF Tracker

## Historical dashboard

`etf_dashboard.py` is the historical dashboard for the 50-stock basket from the uploaded ETF image. It uses the exact supplied weights now stored in `portfolio.csv` and provides:

- synthetic daily NAV indexed to a configurable starting NAV
- comparison with SPY, active return, CAGR, volatility, Sharpe, beta, correlation, and drawdown
- constituent price returns and weighted contributions
- optional CompanyCharts ConsistencyScore and `%Max` lens, with a score-aware attribution table and scatter plot
- adjusted-price mode for a total-return-style proxy, plus price-only mode
- CSV export of the constituent contribution table

The image itself has no embedded creation timestamp. The dashboard therefore defaults to `2026-08-14`, the date associated with this ETF work in the project. Change the inception date in the sidebar if you intended the earlier `2023-08-17` snapshot or another date.

Run it with:

```powershell
streamlit run etf_dashboard.py
```

The historical series uses Yahoo Finance daily data through `yfinance`; it is not a live quote feed and should be refreshed when you want the latest completed trading day.

## CompanyCharts research lens

The dashboard is branded for the CompanyCharts research framework and includes the bundled `consistency_snapshot.csv`, a historical snapshot dated 2023-08-17. It contains the two score fields used in the research:

- `ConsistencyScore adjusted %Max`: positive-only consistency normalized by the maximum possible score for each company's available history.
- `ConsistencyScore TTM adjusted %Max`: the buyback-neutral TTM company-total version using revenue, net income, FCF, gross profit, CFO, and equity.

The sidebar accepts a newer snapshot CSV and will use it instead of the bundled file. `%Max` is displayed as a percentage; the source files may store it as a decimal fraction. The dashboard does not claim that a high score forecasts returns by itself: it shows the score alongside realized price return and portfolio contribution.

## Publish on GitHub and host it

The clean Streamlit Cloud entrypoint is `streamlit_app.py`. The repository only needs these files for the hosted dashboard:

```text
streamlit_app.py
etf_dashboard.py
portfolio.csv
consistency_snapshot.csv
requirements.txt
.streamlit/config.toml
```

To publish from PowerShell, create an empty GitHub repository first, then run this from the project folder:

```powershell
git init
git add streamlit_app.py etf_dashboard.py portfolio.csv requirements.txt .streamlit/config.toml README.md .gitignore
git commit -m "Add custom ETF Streamlit dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Then open [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select the GitHub repository and `main` branch, and set the entrypoint to `streamlit_app.py`. Community Cloud installs dependencies from `requirements.txt` and redeploys when you push updates to GitHub. The repository must be one you own or administer.

## Intraday tracker

The optional live-session tracker remains available in `intraday_tracker.py`. It reads the same 50-stock `portfolio.csv` by default:

```powershell
streamlit run intraday_tracker.py
```

Small Streamlit app for tracking a custom weighted stock basket as a synthetic ETF.

## What it does

- Loads a basket from `portfolio.csv` or an uploaded CSV.
- Normalizes weights automatically.
- Calculates a synthetic NAV starting from a configurable base such as `100`.
- Shows current NAV, minute return, day return, constituent prices, weights, and contributions.
- Plots a simple intraday NAV chart during the session.
- Supports:
  - `yfinance` for no-key, near-live/delayed mode
  - `Alpaca`
  - `Finnhub`
  - `Polygon`

## CSV format

```csv
ticker,target_weight
AAPL,0.20
MSFT,0.20
```

Weights can be decimals, percentages expressed as raw numbers, or any positive numbers. The app normalizes them.

## Run on Windows

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- `yfinance` is the simplest option and requires no API key, but the data is not guaranteed to be real-time.
- Alpaca, Finnhub, and Polygon may provide better live behavior, but actual freshness depends on your plan and exchange permissions.
- `portfolio.csv` contains the exact 50-ticker basket and supplied weights from the source image.

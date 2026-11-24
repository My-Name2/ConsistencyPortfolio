
# ===================================== RUN ============================================
import os as _os
import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPool
from datetime import date as _date

_os.environ["SEC_USER_AGENT"] = SEC_USER_AGENT

try:
    from IPython.display import Markdown as _Markdown, display as _display
except Exception:  # plain Python: print instead
    _Markdown = None

    def _display(obj):
        print(obj)


def _as_of() -> _date:
    return _date.fromisoformat(AS_OF) if isinstance(AS_OF, str) else (AS_OF or _date.today())


def _ticker_list() -> list[str]:
    if TICKERS is None:
        tickers = default_tickers()
    elif isinstance(TICKERS, str):
        tickers = parse_ticker_list(TICKERS)
    else:
        tickers = parse_ticker_list(",".join(TICKERS))
    return tickers[:LIMIT] if LIMIT else tickers


_lock = _threading.Lock()


def review_one(ticker: str):
    """Fetch Inputs A and C into memory and run the review. Returns (ticker, review or None, error)."""
    try:
        price = CURRENT_PRICES.get(ticker)
        review = run_review(fetch_inputs(ticker, _as_of(), price))
        return ticker, review, ""
    except Exception as exc:
        return ticker, None, f"{type(exc).__name__}: {exc}"


def run_all() -> tuple[dict, dict, pd.DataFrame]:
    tickers = _ticker_list()
    reviews, errors, rows = {}, {}, []
    print(f"Reviewing {len(tickers)} ticker(s) as of {_as_of()} with {WORKERS} thread(s); "
          f"output: {OUTPUT_DIR or 'memory only (nothing written)'}")
    with _ThreadPool(max_workers=max(1, WORKERS)) as pool:
        for k, (ticker, review, error) in enumerate(pool.map(review_one, tickers), start=1):
            if review is None:
                errors[ticker] = error
                rows.append({"ticker": ticker, "status": "error", "error": error})
            else:
                reviews[ticker] = review
                rows.append(summarize(review))
            if k % 25 == 0 or k == len(tickers):
                print(f"  {k}/{len(tickers)} done ({len(errors)} error(s))")
    return reviews, errors, pd.DataFrame(rows)


def workbook(ticker: str) -> bytes:
    """The Excel workbook (live formulas) for one reviewed ticker, as bytes."""
    return build_workbook(REVIEWS[ticker.upper()])


def report(ticker: str, show: bool = True) -> str:
    """The markdown report (sections 1-10) for one reviewed ticker."""
    text = build_report(REVIEWS[ticker.upper()])
    if show:
        _display(_Markdown(text) if _Markdown else text)
    return text


def daily(ticker: str) -> pd.DataFrame | None:
    """The daily price-ratio table (held fundamentals) for one reviewed ticker."""
    return REVIEWS[ticker.upper()].daily


def save(ticker: str, folder: str) -> list[str]:
    """Write one company's workbook and report to `folder` (refused on the C: drive)."""
    out = output_dir(folder)
    review = REVIEWS[ticker.upper()]
    stem = _safe_name(f"{review.header['ticker'] or review.header['cik']}_{_as_of().isoformat()}")
    paths = []
    if not review.thin:
        (out / f"{stem}_valuation.xlsx").write_bytes(build_workbook(review))
        paths.append(str(out / f"{stem}_valuation.xlsx"))
    (out / f"{stem}_report.md").write_text(build_report(review), encoding="utf-8")
    paths.append(str(out / f"{stem}_report.md"))
    return paths


def chart(ticker: str, ratio: str = "pe", series: str = "eps_ttm"):
    """Daily ratio over the display window plus the held TTM value as a step line (needs altair)."""
    import altair as alt

    review = REVIEWS[ticker.upper()]
    d = review.daily
    if d is None:
        raise ValueError(f"No daily series for {ticker}: {review.price_block_reason or 'no price data'}")
    w = d[d["in_display_window"]].reset_index()
    df = pd.DataFrame({"date": w["date"], "ratio": w[ratio], "held": pd.to_numeric(w[series], errors="coerce"),
                       "filing": w["form"].astype(str) + " " + w["accn"].astype(str), "valuation_date": w["valuation_date"]})
    tip = ["date:T", alt.Tooltip("ratio:Q", format=",.2f"), alt.Tooltip("held:Q", format=",.4g"), "filing:N", "valuation_date:T"]
    rules = alt.Chart(pd.DataFrame({"v": sorted(set(df["valuation_date"].dropna()))})).mark_rule(strokeDash=[2, 3], opacity=0.4).encode(x="v:T")
    top = alt.layer(rules, alt.Chart(df).mark_line(color="#2a78d6", invalid=None).encode(x="date:T", y=alt.Y("ratio:Q", title=RATIO_BY_KEY[ratio].label), tooltip=tip)).properties(width=760, height=260)
    bottom = alt.layer(rules, alt.Chart(df).mark_line(color="#eb6834", interpolate="step-after", invalid=None).encode(x="date:T", y=alt.Y("held:Q", title=series), tooltip=tip)).properties(width=760, height=180)
    return alt.vconcat(top, bottom)


sources.FILL_UNINDEXED_FILINGS = bool(FILL_MISSING_FILINGS)
INCLUDE_COMPARATIVE_QUARTERS = (REFERENCE_COMPATIBLE if COMPARATIVE_QUARTERS is None else COMPARATIVE_QUARTERS) and True
REVIEWS, ERRORS, SUMMARY = run_all()
_display(SUMMARY)

if OUTPUT_DIR and WRITE_FILES:
    _out = output_dir(OUTPUT_DIR)  # raises StorageRefused for a C: drive folder
    for _t in REVIEWS:
        save(_t, str(_out))
    SUMMARY.to_csv(_out / f"batch_summary_{_as_of().isoformat()}.csv", index=False)
    print(f"Wrote {len(REVIEWS)} report(s) and workbooks to {_out}")

COMPARISON = COMPARISON_DETAIL = None
if REFERENCE_XLSX:
    _opts = ReferenceOptions.reference_compatible() if REFERENCE_COMPATIBLE else ReferenceOptions()
    COMPARISON, COMPARISON_DETAIL = compare(REVIEWS, load_reference(REFERENCE_XLSX), _opts)
    if COMPARISON.empty:
        print(f"No reviewed ticker is in {REFERENCE_XLSX}; nothing to compare.")
    else:
        print(f"Share of tickers whose counts match {REFERENCE_XLSX} exactly ({'reference conventions' if REFERENCE_COMPATIBLE else 'written spec'}):")
        _display(COMPARISON.pivot(index="field", columns="metric", values="exact").reindex(FIELDS))

if SHOW_FIRST_REPORT and REVIEWS:
    report(next(iter(REVIEWS)))

print("Done. Use REVIEWS[ticker], SUMMARY, ERRORS, report(t), workbook(t) -> bytes, daily(t), chart(t), save(t, folder)"
      + (", COMPARISON, COMPARISON_DETAIL" if REFERENCE_XLSX else "") + ".")

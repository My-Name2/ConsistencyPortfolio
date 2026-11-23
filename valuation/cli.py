"""Command line: review one company or a whole ticker list.

Nothing is written unless --out-dir (or VALUATION_OUTPUT_DIR) names a folder
that is not on the C: drive. Without it, results print to the console only.

  python -B -m valuation.cli review AAPL --as-of 2026-11-23 --out-dir D:\\valuation
  python -B -m valuation.cli batch --as-of 2026-11-23 --out-dir D:\\valuation --workers 4
  python -B -m valuation.cli batch --tickers my_list.txt          # summary only, nothing written
  python -B -m valuation.cli review --facts CIK0000320193.json --prices prices.csv \\
         --split-adjusted yes --dividend-adjusted no --currency USD --as-of 2026-11-23
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

from . import sources
from .pipeline import ReviewInputs, run_review
from .prices import PriceMeta
from .redflags import InputBDocument
from .report import build_report
from .storage import StorageRefused, output_dir
from .workbook import build_workbook


def fetch_inputs(identifier: str, as_of: date, current_price: float | None = None, docs: list[InputBDocument] | None = None) -> ReviewInputs:
    ticker, cik = sources.resolve(identifier)
    facts = sources.fetch_companyfacts(cik)
    notes = [f"Input A: SEC companyfacts {facts.url} retrieved {facts.retrieved} (UTC)."]
    prices, meta = None, PriceMeta(ticker=ticker)
    try:
        px = sources.fetch_prices(ticker)
        prices, meta = px.frame, px.meta
        notes.append(
            f"Input C: Yahoo Finance chart API {px.url} retrieved {px.retrieved} (UTC); exchange {px.exchange}; "
            "close is split-adjusted, not dividend-adjusted; dividends are cash amounts on ex-dates; split events with ex-dates."
        )
    except Exception as exc:  # prices are optional: price ratios become "no price data"
        notes.append(f"Input C: not available ({exc}).")
    return ReviewInputs(facts=facts.data, as_of=as_of, prices=prices, price_meta=meta, ticker=ticker, current_price=current_price, filings_text=docs or [], source_notes=notes)


def local_inputs(args) -> ReviewInputs:
    meta = PriceMeta.from_json(args.price_meta) if args.price_meta else PriceMeta()
    from .prices import _as_bool

    if args.split_adjusted is not None:
        meta.split_adjusted = _as_bool(args.split_adjusted)
    if args.dividend_adjusted is not None:
        meta.dividend_adjusted = _as_bool(args.dividend_adjusted)
    if args.currency:
        meta.currency = args.currency.upper()
    facts_path = Path(args.facts)
    data = json.loads(gzip.decompress(facts_path.read_bytes()) if facts_path.suffix == ".gz" else facts_path.read_text(encoding="utf-8"))
    docs = [InputBDocument(Path(p).name, Path(p).read_text(encoding="utf-8", errors="replace")) for p in (args.filing_text or [])]
    notes = [f"Input A: local file {facts_path.name}."]
    if args.prices:
        notes.append(f"Input C: local file {Path(args.prices).name}.")
    return ReviewInputs(
        facts=data, as_of=args.as_of, prices=args.prices, price_meta=meta, dividends=args.dividends, ticker=args.ticker or meta.ticker,
        current_price=args.price, filings_text=docs, source_notes=notes,
    )


def summarize(review) -> dict:
    snap = review.snapshot.set_index("key") if not review.snapshot.empty else pd.DataFrame()
    stats = review.stats.set_index("series") if not review.stats.empty else pd.DataFrame()
    rf = review.red_flags
    dq = review.dq.frame()

    def val(key):
        if key not in snap.index:
            return None
        v = snap.at[key, "value"]
        return v if isinstance(v, str) or v is None else float(v)

    def st(key, field):
        return stats.at[key, field] if key in stats.index else None

    return {
        "ticker": review.header["ticker"],
        "company": review.header["company"],
        "cik": review.header["cik"],
        "status": "thin coverage" if review.thin else "ok",
        "latest_filing": review.header["latest_filing"],
        "price": review.snapshot_inputs.get("price"),
        "price_date": str(review.snapshot_inputs.get("price_date").date()) if review.snapshot_inputs.get("price_date") is not None else "",
        "pe": val("pe"), "ps": val("ps"), "pb": val("pb"), "pfcf": val("pfcf"), "ev_ebitda": val("ev_ebitda"), "fcf_yield": val("fcf_yield"), "div_yield": val("div_yield"),
        "pe_percentile": review.percentiles.get("pe", {}).get("percentile"),
        "revenue_raw_increase_rate": st("ttm_revenue", "raw_increase_rate"),
        "eps_raw_increase_rate": st("eps_ttm", "raw_increase_rate"),
        "eps_adj_increase_rate": st("eps_ttm", "adj_increase_rate"),
        "eps_latest_record": st("eps_ttm", "raw_latest_record"),
        "serious_flags": int(rf["severity"].isin(["serious", "critical"]).sum()) if not rf.empty else 0,
        "flags": "; ".join(sorted(set(rf.loc[rf["severity"].isin(["serious", "critical"]), "flag"]))) if not rf.empty else "",
        "dq_warnings": int(dq["severity"].isin(["warning", "serious", "critical"]).sum()) if not dq.empty else 0,
        "error": "",
    }


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def run_one(identifier: str, as_of: date, out: str | None, current_price: float | None = None) -> dict:
    """Fetch, review and (optionally) write one company. Returns a summary row."""
    try:
        inputs = fetch_inputs(identifier, as_of, current_price)
        review = run_review(inputs)
        row = summarize(review)
        if out:
            folder = Path(out)
            stem = _safe_name(f"{review.header['ticker'] or review.header['cik']}_{as_of.isoformat()}")
            workbook_path = None
            if not review.thin:
                workbook_path = folder / f"{stem}_valuation.xlsx"
                workbook_path.write_bytes(build_workbook(review))
            (folder / f"{stem}_report.md").write_text(build_report(review, str(workbook_path) if workbook_path else None), encoding="utf-8")
            row["workbook"] = str(workbook_path) if workbook_path else ""
        return row
    except Exception as exc:
        return {"ticker": identifier, "status": "error", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="valuation.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--as-of", type=date.fromisoformat, default=date.today(), help="Input D as-of date (default: today)")
    common.add_argument("--out-dir", default=None, help="Folder for workbooks and reports; must not be on the C: drive. Omit to write nothing.")

    one = sub.add_parser("review", parents=[common], help="Review one company")
    one.add_argument("identifier", nargs="?", help="Ticker or CIK (fetches Inputs A and C)")
    one.add_argument("--price", type=float, default=None, help="Input D current price (optional)")
    one.add_argument("--facts", help="Local Input A companyfacts JSON (.json or .json.gz) instead of fetching")
    one.add_argument("--prices", help="Local Input C CSV (date, close[, dividend, split_ratio])")
    one.add_argument("--dividends", help="Local dividends CSV (ex_date, amount)")
    one.add_argument("--price-meta", help="JSON with ticker, split_adjusted, dividend_adjusted, currency")
    one.add_argument("--split-adjusted", choices=["yes", "no"], default=None)
    one.add_argument("--dividend-adjusted", choices=["yes", "no"], default=None)
    one.add_argument("--currency", default=None)
    one.add_argument("--ticker", default=None)
    one.add_argument("--filing-text", nargs="*", help="Input B excerpt files (MD&A, auditor's report, going-concern notes)")

    many = sub.add_parser("batch", parents=[common], help="Review every ticker in a list, each separately")
    many.add_argument("--tickers", default=None, help="Ticker list file (default: the packaged list)")
    many.add_argument("--workers", type=int, default=4)
    many.add_argument("--limit", type=int, default=None, help="Only the first N tickers")
    args = parser.parse_args(argv)

    try:
        out = output_dir(args.out_dir)
    except StorageRefused as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.command == "review":
        if args.facts:
            review = run_review(local_inputs(args))
        elif args.identifier:
            review = run_review(fetch_inputs(args.identifier, args.as_of, args.price))
        else:
            parser.error("give a ticker/CIK or --facts")
        workbook_path = None
        if out and not review.thin:
            stem = _safe_name(f"{review.header['ticker'] or review.header['cik']}_{args.as_of.isoformat()}")
            workbook_path = out / f"{stem}_valuation.xlsx"
            workbook_path.write_bytes(build_workbook(review))
        report = build_report(review, str(workbook_path) if workbook_path else None, full_dq=bool(out))
        if out:
            path = out / f"{_safe_name(review.header['ticker'] or review.header['cik'])}_{args.as_of.isoformat()}_report.md"
            path.write_text(report, encoding="utf-8")
            print(f"Wrote {path}" + (f" and {workbook_path}" if workbook_path else ""))
        else:
            print(report)
        return 0

    tickers = sources.read_ticker_file(args.tickers) if args.tickers else sources.default_tickers()
    if args.limit:
        tickers = tickers[: args.limit]
    print(f"Reviewing {len(tickers)} tickers as of {args.as_of} with {args.workers} worker(s); output: {out or 'memory only (nothing written)'}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, t, args.as_of, str(out) if out else None): t for t in tickers}
        for k, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{k}/{len(tickers)}] {row['ticker']}: {row['status']}" + (f" ({row['error']})" if row.get("error") else ""), flush=True)
    order = {t: i for i, t in enumerate(tickers)}
    summary = pd.DataFrame(rows)
    summary["_o"] = summary["ticker"].map(lambda t: order.get(t, len(order)))
    summary = summary.sort_values("_o").drop(columns="_o")
    if out:
        summary.drop(columns=[c for c in ("trace",) if c in summary]).to_csv(out / f"batch_summary_{args.as_of.isoformat()}.csv", index=False)
        print(f"Wrote {out / f'batch_summary_{args.as_of.isoformat()}.csv'}")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(summary[["ticker", "status", "pe", "ev_ebitda", "pfcf", "serious_flags", "error"]].to_string(index=False))
    failed = summary[summary["status"] == "error"]
    if not failed.empty:
        print(f"{len(failed)} ticker(s) failed: {', '.join(failed['ticker'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

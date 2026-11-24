"""Input A supplement: XBRL facts of periodic filings that SEC companyfacts has not indexed.

SEC's companyfacts API sometimes leaves out a company's recent 10-Q or 10-K, although the filing and its
XBRL are on EDGAR. For example, Duke Energy's 10-Qs filed 2026-08-04 and 2026-11-05 are missing from
companyfacts, which does include its later prospectus filings. Without them the review's latest quarter is
stale.

This module finds those filings with the SEC submissions API (same company, same SEC data) and reads each
one's own XBRL instance into the companyfacts layout. It keeps only what companyfacts keeps:
- numeric facts without dimensions;
- of the us-gaap, ifrs-full, dei, srt and invest taxonomies;
- for the filer's own CIK;
- with the filing's accession number, form, filing date and fiscal year/period focus.

`find_unindexed` only reads; `merge_facts` adds the rows. Both are opt-in in the fetch layer, and the review
lists every filing added this way under Data_Quality. Nothing is written to disk.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, timedelta

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/index.json"
ARCHIVE_FILE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/{name}"

SUPPLEMENT_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A", "20-F", "20-F/A", "40-F", "40-F/A"}
LOOKBACK_YEARS = 3  # only filings this recent are checked against companyfacts
MAX_FILINGS = 8  # at most this many filings are read per company

_XBRLI = "{http://www.xbrl.org/2003/instance}"
_TAXONOMIES = (
    ("us-gaap", re.compile(r"^https?://(www\.)?fasb\.org/us-gaap/")),
    ("srt", re.compile(r"^https?://(www\.)?fasb\.org/srt/")),
    ("dei", re.compile(r"^https?://xbrl\.sec\.gov/dei/")),
    ("invest", re.compile(r"^https?://xbrl\.sec\.gov/invest/")),
    ("ifrs-full", re.compile(r"^https?://xbrl\.ifrs\.org/taxonomy/")),
)


@dataclass
class UnindexedFiling:
    accn: str
    form: str
    filed: str
    report_date: str
    primary_document: str
    facts: int = 0  # facts read from its XBRL instance
    instance: str = ""  # instance file name, or why none was read
    rows: dict = field(default_factory=dict, repr=False)  # {taxonomy: {tag: {unit: [row, ...]}}}


def indexed_accessions(companyfacts: dict) -> set[str]:
    out = set()
    for tags in (companyfacts.get("facts") or {}).values():
        for payload in tags.values():
            for items in (payload.get("units") or {}).values():
                out.update(item.get("accn") for item in items)
    out.discard(None)
    return out


def find_unindexed(companyfacts: dict, submissions: dict, as_of: date) -> list[UnindexedFiling]:
    """Periodic XBRL filings on or before `as_of` (last LOOKBACK_YEARS) that companyfacts does not contain."""
    have = indexed_accessions(companyfacts)
    recent = ((submissions or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    cutoff = (as_of - timedelta(days=round(365.25 * LOOKBACK_YEARS))).isoformat()
    out = []
    for i, form in enumerate(forms):
        filed = (recent.get("filingDate") or [""] * len(forms))[i]
        accn = (recent.get("accessionNumber") or [""] * len(forms))[i]
        is_xbrl = (recent.get("isXBRL") or [1] * len(forms))[i]
        if form not in SUPPLEMENT_FORMS or not is_xbrl or not accn or accn in have:
            continue
        if not filed or filed > as_of.isoformat() or filed < cutoff:
            continue
        out.append(UnindexedFiling(accn, form, filed, (recent.get("reportDate") or [""] * len(forms))[i],
                                   (recent.get("primaryDocument") or [""] * len(forms))[i]))
    out.sort(key=lambda f: (f.filed, f.accn), reverse=True)
    return out[:MAX_FILINGS]


def pick_instance(index: dict, primary_document: str = "") -> str:
    """The XBRL instance in a filing folder: the extracted inline-XBRL instance (<doc>_htm.xml) or a plain one."""
    names = [item.get("name", "") for item in ((index or {}).get("directory") or {}).get("item", [])]
    stem = primary_document.rsplit(".", 1)[0] if primary_document else ""
    if stem and f"{stem}_htm.xml" in names:
        return f"{stem}_htm.xml"
    extracted = [n for n in names if n.endswith("_htm.xml")]
    if extracted:
        return extracted[0]
    plain = [n for n in names if n.endswith(".xml") and not re.search(r"_(cal|def|lab|pre)\.xml$|FilingSummary|MetaLinks", n)]
    return plain[0] if plain else ""


def _taxonomy(namespace: str) -> str | None:
    for name, pattern in _TAXONOMIES:
        if pattern.match(namespace):
            return name
    return None


def _unit_name(unit: ET.Element) -> str:
    def measure(node):
        text = (node.text or "").strip()
        return text.split(":", 1)[1] if ":" in text else text

    divide = unit.find(f"{_XBRLI}divide")
    if divide is not None:
        num = divide.find(f"{_XBRLI}unitNumerator/{_XBRLI}measure")
        den = divide.find(f"{_XBRLI}unitDenominator/{_XBRLI}measure")
        return f"{measure(num)}/{measure(den)}" if num is not None and den is not None else ""
    parts = [measure(m) for m in unit.findall(f"{_XBRLI}measure")]
    return "*".join(parts)


def _number(text: str):
    value = float(text)
    return int(value) if value.is_integer() and abs(value) < 2**53 else value


def parse_instance(data: bytes, cik: str, accn: str, form: str, filed: str) -> dict:
    """Facts of one XBRL instance as {taxonomy: {tag: {unit: [companyfacts-style row, ...]}}}."""
    root = ET.parse(io.BytesIO(data)).getroot()
    own = str(int(cik)) if str(cik).strip().isdigit() else str(cik)
    contexts = {}
    for ctx in root.iter(f"{_XBRLI}context"):
        entity = ctx.find(f"{_XBRLI}entity")
        ident = entity.find(f"{_XBRLI}identifier") if entity is not None else None
        if ident is None or (ident.text or "").strip().lstrip("0") != own:
            continue
        if entity.find(f"{_XBRLI}segment") is not None or ctx.find(f"{_XBRLI}scenario") is not None:
            continue  # dimensional: companyfacts leaves these out
        period = ctx.find(f"{_XBRLI}period")
        instant = period.find(f"{_XBRLI}instant") if period is not None else None
        start = period.find(f"{_XBRLI}startDate") if period is not None else None
        end = period.find(f"{_XBRLI}endDate") if period is not None else None
        if instant is not None:
            contexts[ctx.get("id")] = (None, (instant.text or "").strip()[:10])
        elif start is not None and end is not None:
            contexts[ctx.get("id")] = ((start.text or "").strip()[:10], (end.text or "").strip()[:10])
    units = {u.get("id"): _unit_name(u) for u in root.iter(f"{_XBRLI}unit")}

    focus = {}
    for el in root:
        if el.tag.endswith("}DocumentFiscalYearFocus") or el.tag.endswith("}DocumentFiscalPeriodFocus"):
            if el.get("contextRef") in contexts and (el.text or "").strip():
                focus[el.tag.rsplit("}", 1)[1]] = el.text.strip()
    fy = focus.get("DocumentFiscalYearFocus")
    fy = int(fy) if fy and fy.isdigit() else None
    fp = focus.get("DocumentFiscalPeriodFocus")

    out: dict = {}
    seen = set()
    for el in root:
        ref, unit_ref = el.get("contextRef"), el.get("unitRef")
        if not ref or not unit_ref or ref not in contexts or not el.tag.startswith("{"):
            continue
        if el.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
            continue
        namespace, tag = el.tag[1:].split("}", 1)
        taxonomy = _taxonomy(namespace)
        unit = units.get(unit_ref)
        if taxonomy is None or not unit:
            continue
        try:
            value = _number((el.text or "").strip())
        except ValueError:
            continue
        start, end = contexts[ref]
        key = (taxonomy, tag, unit, start, end)
        if key in seen:  # the same fact can appear more than once in an instance
            continue
        seen.add(key)
        row = {"end": end, "val": value, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
        if start is not None:
            row = {"start": start, **row}
        out.setdefault(taxonomy, {}).setdefault(tag, {}).setdefault(unit, []).append(row)
    return out


def merge_facts(companyfacts: dict, rows: dict) -> int:
    """Add parsed rows to a companyfacts dict in place. Returns the number of rows added."""
    facts = companyfacts.setdefault("facts", {})
    added = 0
    for taxonomy, tags in rows.items():
        for tag, by_unit in tags.items():
            payload = facts.setdefault(taxonomy, {}).setdefault(tag, {"label": tag, "description": "", "units": {}})
            for unit, items in by_unit.items():
                payload.setdefault("units", {}).setdefault(unit, []).extend(items)
                added += len(items)
    return added


def describe(filings: list[UnindexedFiling]) -> list[str]:
    return [f"{f.form} filed {f.filed} (accession {f.accn}, period {f.report_date}): "
            + (f"{f.facts} facts read from {f.instance}" if f.facts else f"not read ({f.instance or 'no XBRL instance'})")
            for f in filings]

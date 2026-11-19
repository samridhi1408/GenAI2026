"""Retrieve and parse open-access STAR Protocols articles from Europe PMC."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
JOURNAL_FILTER = 'JOURNAL:"STAR Protocols" AND OPEN_ACCESS:y'


def _get(url, retries=5):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def search(phrases, date_from=None, date_to=None, limit=1000):
    """Return [{pmcid, date, title}] for articles mentioning any phrase."""
    terms = " OR ".join('"%s"' % p.replace('"', "") for p in phrases)
    query = f"{JOURNAL_FILTER} AND ({terms})"
    if date_from or date_to:
        query += f" AND FIRST_PDATE:[{date_from or '2000-01-01'} TO {date_to or '2100-01-01'}]"
    out, cursor = [], "*"
    while len(out) < limit:
        url = API + "search?" + urllib.parse.urlencode(
            {"query": query, "format": "json", "pageSize": 1000, "cursorMark": cursor, "resultType": "lite"})
        d = json.loads(_get(url))
        rows = d.get("resultList", {}).get("result", [])
        out += [{"pmcid": r["pmcid"], "date": r.get("firstPublicationDate", ""), "title": r.get("title", ""),
                 "pmid": r.get("pmid", ""), "doi": r.get("doi", ""), "pub_year": r.get("pubYear", ""),
                 "first_publication_date": r.get("firstPublicationDate", ""),
                 "first_index_date": r.get("firstIndexDate", ""), "journal": r.get("journalTitle", "")}
                for r in rows if r.get("pmcid")]
        nxt = d.get("nextCursorMark")
        if not rows or not nxt or nxt == cursor:
            break
        cursor = nxt
    return out[:limit]


def fetch_xml(pmcid, cache_dir: Path):
    path = cache_dir / f"{pmcid}.xml"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(API + f"{pmcid}/fullTextXML"))
    return path


def _text(el):
    return " ".join("".join(el.itertext()).split())


def _title(sec):
    t = sec.find("title")
    return _text(t) if t is not None else ""


def _body_text(sec):
    """Section text with one line per paragraph, list item or sub-heading."""
    # Step numbers live in <label> children of list items, not in the paragraph
    # text; keep them, because troubleshooting entries refer to steps by number.
    labels = {}
    for item in sec.iter("list-item"):
        label, first = item.find("label"), item.find("p")
        if label is not None and first is not None and _text(label):
            labels[first] = _text(label)
    lines = []
    for el in sec.iter():
        if el.tag == "title":
            lines.append(f"## {_text(el)}")
        elif el.tag == "p":
            lines.append(f"{labels[el]} {_text(el)}" if el in labels else _text(el))
    seen, out = set(), []
    for line in lines:  # nested <p> inside list items repeat; keep the first
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out)


def parse_article(path: Path):
    """Return a dict with metadata, protocol text and troubleshooting entries.

    Most articles use sibling sections ``Problem N`` / ``Potential solution``.
    Anything else is returned as a single unstructured entry so the extractor
    can still read it."""
    root = ET.parse(path).getroot()
    meta = root.find(".//article-meta")
    ids = {e.get("pub-id-type"): (e.text or "") for e in meta.findall("article-id")}
    title = _text(meta.find(".//article-title")) if meta.find(".//article-title") is not None else ""
    dates = []
    for pd in meta.findall("pub-date"):  # attribute conventions vary; take the earliest full date
        if pd.findtext("year"):
            dates.append("%s-%02d-%02d" % (pd.findtext("year"), int(pd.findtext("month") or 12), int(pd.findtext("day") or 28)))
    date = min(dates) if dates else ""
    # Every date the article states, as written (a missing month or day stays
    # missing), so later analyses can split on whichever date they prefer.
    def _raw(el):
        return "-".join(x.zfill(2) for x in (el.findtext("year"), el.findtext("month"), el.findtext("day")) if x)
    pub_dates = {}
    for pd in meta.findall("pub-date"):
        if pd.findtext("year"):
            pub_dates[pd.get("pub-type") or pd.get("date-type") or "unspecified"] = _raw(pd)
    history = {d.get("date-type") or "unspecified": _raw(d) for d in meta.findall(".//history/date") if d.findtext("year")}
    journal = root.findtext(".//journal-meta//journal-title") or ""
    volume, issue = meta.findtext("volume") or "", meta.findtext("issue") or ""
    abstract = " ".join(_text(a) for a in meta.findall("abstract"))
    body = root.find(".//body")
    sections, trouble = [], None
    for sec in (body.findall("sec") if body is not None else []):
        name = _title(sec)
        if name.strip().lower() == "troubleshooting":
            trouble = sec
        elif not re.match(r"(?i)(resource availability|key resources|acknowledg|author contrib|declaration)", name):
            sections.append({"title": name, "text": _body_text(sec)})
    entries = []
    if trouble is not None:
        current = None
        for sub in trouble.findall("sec"):
            name = _title(sub)
            if re.match(r"(?i)\s*(problem|troubleshooting)\b", name) or current is None:
                current = {"index": len(entries) + 1, "heading": name, "text": _body_text(sub)}
                entries.append(current)
            else:
                current["text"] += "\n" + _body_text(sub)
        if not entries:
            entries = [{"index": 1, "heading": "Troubleshooting", "text": _body_text(trouble)}]
    return {"pmcid": ids.get("pmcid") or path.stem, "doi": ids.get("doi", ""), "title": title, "date": date,
            "abstract": abstract, "sections": sections, "troubleshooting": entries,
            "pmid": ids.get("pmid", ""), "pub_dates": pub_dates, "history_dates": history, "journal": journal,
            "volume": volume, "issue": issue}


def protocol_text(article, max_chars=60000):
    """Protocol body for prompting. Sections are kept whole and in order; if the
    article exceeds ``max_chars`` the longest sections are shortened from the
    end and the cut is marked, so the reader knows text is missing."""
    parts = [(s["title"], s["text"]) for s in article["sections"]]
    total = sum(len(t) for _, t in parts)
    if total > max_chars and parts:
        keep = max_chars / total
        parts = [(n, t if len(t) < 2000 else t[: int(len(t) * keep)] + "\n[... section shortened ...]")
                 for n, t in parts]
    return "\n\n".join(f"# {n}\n{t}" for n, t in parts)

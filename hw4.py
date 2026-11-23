"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi


_HW_COMPANIES: dict[str, str] = {}
_HW_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}
_HW_STOP = set("""a an the and or of in at on to for from with by as its it their
    what which who how much many more higher was were is are did does do had have has
    company companies report reported hold held earn earned generate generated give
    set announce announced end ended year fiscal fy december usd million millions
    billion percent value amount financial statements share shares fourth quarter q4
    annual approximately total number""".split())


def _hw_text(element) -> str:
    return re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()


def _hw_key(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold())
    return " ".join(w for w in words if w not in _HW_SUFFIXES)


def _hw_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z]+|\d+(?:[,.]\d+)*", text.casefold())
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w
            for w in words if w not in _HW_STOP]


def _hw_table_parts(table_html: str) -> tuple[str, list[tuple[str, list[str], list[str]]]]:
    """Read header and body grids; propagate HTML rowspan/colspan without changing cells."""
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return "", []
    caption = _hw_text(table.caption) if table.caption else ""
    grid, is_header, spans = [], [], {}
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue
        cells = tr.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        row, col, next_spans = [], 0, {}

        def take_span():
            nonlocal col
            value, remaining = spans[col]
            row.append(value)
            if remaining > 1:
                next_spans[col] = (value, remaining - 1)
            col += 1

        for cell in cells:
            while col in spans:
                take_span()
            value = _hw_text(cell)
            width = max(1, int(cell.get("colspan", 1)))
            height = max(1, int(cell.get("rowspan", 1)))
            for _ in range(width):
                row.append(value)
                if height > 1:
                    next_spans[col] = (value, height - 1)
                col += 1
        while col in spans:
            take_span()
        spans = next_spans
        header = tr.find_parent("thead") is not None
        if not grid and not table.find("thead"):
            header = all(c.name == "th" and c.get("scope") != "row" for c in cells)
        grid.append(row)
        is_header.append(header)
    width = max(map(len, grid), default=0)
    headers = []
    for col in range(width):
        pieces = []
        for row, header in zip(grid, is_header):
            if header and col < len(row) and row[col] and row[col] not in pieces:
                pieces.append(row[col])
        headers.append(" / ".join(pieces))
    rows = []
    for row, header in zip(grid, is_header):
        if header or not row:
            continue
        # A full-width section title is preserved, but has no numerical cells.
        if len(set(row)) == 1:
            rows.append((row[0], [], []))
        else:
            rows.append((row[0], headers[1:len(row)], row[1:]))
    return caption, rows


def _hw_row_lines(label: str, columns: list[str], values: list[str]) -> list[str]:
    lines, current = [], label
    for column, value in zip(columns, values):
        piece = f" | {column}: {value}"
        if len(current + piece) > 400 and current != label:
            lines.append(current)
            current = label
        current += piece
    if current:
        lines.append(current)
    return lines


def _hw_metadata(file_name: str, soup) -> dict:
    h1 = soup.find("h1")
    if h1 is None:
        raise ValueError(f"No company heading in {file_name}")
    company = re.sub(r"\s*\([^)]*\)\s*$", "", _hw_text(h1)).strip()
    opening = " ".join(_hw_text(e) for e in soup.find_all(["h1", "h2", "p"])[:8])
    if re.search(r"earnings\s+(?:conference\s+)?call", opening, re.I):
        doc_type = "earnings_call"
        match = re.search(r"(?:full\s+year|FY)\s*((?:19|20)\d{2})", opening, re.I)
    elif re.search(r"results\s+review", opening, re.I):
        doc_type = "research_report"
        match = re.search(r"FY\s*((?:19|20)\d{2})", opening, re.I)
    else:
        doc_type = "annual_report"
        match = re.search(r"Annual\s+Report\s+((?:19|20)\d{2})", opening, re.I)
    if match is None:
        raise ValueError(f"Cannot identify fiscal year in {file_name}")
    return {"source": file_name, "company": company,
            "fiscal_year": int(match.group(1)), "doc_type": doc_type}


def _hw_metric(question: str) -> tuple[str, str]:
    """Financial vocabulary, not company-, question- or answer-specific rules."""
    specs = [
        (r"net cash (?:from|provided by) operating|operating cash flow", "cash_flow", r"net cash.*operating"),
        (r"cash(?: and cash equivalents)?", "cash", r"cash and cash equivalents"),
        (r"long[ -]term borrow", "borrowings", r"long.term borrow"),
        (r"short[ -]term borrow", "borrowings", r"short.term borrow"),
        (r"accounts? receivable|receivables?", "receivables", r"(?:receivable.*net|net.*receivable)"),
        (r"inventor(?:y|ies)", "inventories", r"inventor"),
        (r"total assets", "assets", r"^total assets$"),
        (r"total equity", "equity", r"^total equity$"),
        (r"net income|net profit", "net_income", r"^net (?:income|profit)$"),
        (r"operating income|operating profit", "operating_income", r"^operating (?:income|profit)$"),
        (r"gross profit", "gross_profit", r"^gross profit$"),
        (r"employees|headcount", "employees", r"^total(?: employees)?$"),
        (r"chief financial officer|\bcfo\b", "cfo", r"chief financial officer"),
        (r"chief executive officer|\bceo\b", "ceo", r"chief executive officer"),
        (r"audit firm|auditor|audited", "auditor", r"audit"),
        (r"headquarter", "headquarters", r"headquarter"),
        (r"target price", "target", r"target price"),
        (r"investment rating|\brating\b", "rating", r"rating"),
        (r"share repurchase|buyback", "repurchase", r"repurchase|buyback"),
        (r"revenue growth", "growth", r"revenue growth"),
        (r"revenue|sales", "revenue", r"^revenue$"),
    ]
    for pattern, name, row in specs:
        if re.search(pattern, question, re.I):
            return name, row
    return "other", ""


def _hw_required_terms(question: str, company: str, metric: str) -> set[str]:
    text = re.sub(re.escape(company), " ", question, flags=re.I)
    # Remove the financial measure, retaining qualifiers such as a region, segment,
    # or 'largest single customer'. These must also be supported by the evidence.
    measure_words = """revenue sales cash equivalent account receivable net operating
        activity long short term borrowing inventory asset equity income profit gross
        employee headcount""".split()
    more_stop = {"s", "in", "since", "during", "spend", "spent", "per", "statement",
                 "segment", "region", "earn", "generate", "report", "hold", "have"}
    return {w for w in _hw_tokens(text)
            if w.isalpha() and w not in set(measure_words) | more_stop}

# =============================================================================

def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    caption, rows = _hw_table_parts(table_html)
    lines = [caption] if caption else []
    for label, columns, values in rows:
        lines.extend(_hw_row_lines(label, columns, values))
    return "\n".join(lines)

def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.find_all(["script", "style", "nav"]):
        node.decompose()
    metadata = _hw_metadata(file_name, soup)
    _HW_COMPANIES[_hw_key(metadata["company"])] = metadata["company"]
    chunks, section = [], ""

    def emit(text: str, extra: dict) -> None:
        # Only the documented metadata and an original section title form the header.
        header = f"{metadata['company']} | FY{metadata['fiscal_year']} | {section}\n"
        if len(header) > 450:
            header = header[:450] + "\n"
        room = MAX_CHUNK_CHARS - len(header)
        remaining = text.strip()
        while remaining:
            end = min(room, len(remaining))
            if end < len(remaining):
                split = remaining.rfind(" ", max(1, end - 150), end)
                if split > 0:
                    end = split
            body = remaining[:end].strip()
            if body:
                chunks.append(Document(page_content=header + body,
                                       metadata={**metadata, "section": section, **extra}))
            if end == len(remaining):
                break
            # Overlap protects names/numbers at the rare long-paragraph boundary.
            remaining = remaining[max(1, end - 120):].lstrip()

    root = soup.body or soup
    for node in root.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        if node.find_parent("table") is not None:
            continue
        if node.name in {"h1", "h2", "h3"}:
            section = _hw_text(node)
            emit(section, {"kind": "text"})
        elif node.name == "table":
            rendered = table_to_text(str(node))
            caption, rows = _hw_table_parts(str(node))
            # table_to_text is the sole source of the rendered table text.
            lines = rendered.splitlines()
            cursor = 1 if caption else 0
            for label, columns, values in rows:
                count = len(_hw_row_lines(label, columns, values))
                row_text = "\n".join(lines[cursor:cursor + count])
                cursor += count
                years = sorted({int(y) for y in re.findall(r"(?:19|20)\d{2}", " ".join(columns))})
                if not years:
                    years = sorted({int(y) for y in re.findall(r"(?:19|20)\d{2}", caption)})
                emit((caption + "\n" if caption else "") + row_text,
                     {"kind": "table", "row_label": label, "table_caption": caption,
                      "data_years": years})
        elif node.find_parent(["p", "li"]) is None:
            emit(_hw_text(node), {"kind": "text"})
    return chunks


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    q = question.replace("’", "'")
    year_match = re.search(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", q, re.I)
    if re.search(r"earnings\s+(?:conference\s+)?call", q, re.I):
        doc_type = "earnings_call"
    elif re.search(r"\brating\b|target price|\bforecast\b|results review|\bbroker\b|\bestimate\b", q, re.I):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    names = []
    comparison = re.search(r":\s*(.+?)\s+or\s+(.+?)\s*[?.]?\s*$", q, re.I)
    if comparison:
        names = [comparison.group(1), comparison.group(2).rstrip("?. ")]
    else:
        # A possessive captures the full named entity, so 'A Robotics Systems'
        # must never silently resolve to a shorter company called 'A Robotics'.
        possessive = re.search(r"\b((?:[A-Z][A-Za-z0-9&.-]*\s+){1,6}[A-Z][A-Za-z0-9&.-]*)'(?:s\b|(?=\s))", q)
        if possessive:
            names = [possessive.group(1)]
        if not names:
            patterns = [
                r"\b(?:did|does)\s+(.+?)\s+(?:hold|report|earn|generate|announce|have)\b",
                r"\bis\s+(.+?)\s+headquartered\b",
                r"\bfor\s+(.+?)(?=,\s*in\b|\s+in\s+(?:its\b|FY\d)|[?])",
                r"\bgive\s+(.+?)\s+in\s+its\b",
            ]
            for pattern in patterns:
                match = re.search(pattern, q, re.I)
                if match:
                    names = [match.group(1)]
                    break
        if not names:
            padded = " " + _hw_key(q) + " "
            names = [name for key, name in _HW_COMPANIES.items() if f" {key} " in padded]
    cleaned, seen = [], set()
    for name in names:
        name = name.strip(" \t\n?.:;\"'")
        key = _hw_key(name)
        if key and key not in seen:
            cleaned.append(name)
            seen.add(key)
    return {"companies": cleaned,
            "fiscal_year": int(year_match.group(1)) if year_match else None,
            "doc_type": doc_type}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    groups = {}
    for i, chunk in enumerate(chunks):
        key = _hw_key(chunk.metadata.get("company", ""))
        groups.setdefault(key, []).append(i)
        if key:
            _HW_COMPANIES[key] = chunk.metadata["company"]
    tokens = [_hw_tokens(c.page_content) or [""] for c in chunks]
    return {"chunks": chunks, "groups": groups,
            "bm25": BM25Okapi(tokens, k1=1.4, b=0.65) if chunks else None}


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    if not index.get("chunks"):
        return []
    query = parse_query(question)
    names, year, doc_type = query["companies"], query["fiscal_year"], query["doc_type"]
    if not names or any(_hw_key(n) not in index["groups"] for n in names):
        return []
    metric, row_pattern = _hw_metric(question)
    scores = index["bm25"].get_scores(_hw_tokens(question))
    ranked_groups = []
    numeric_measures = {"cash_flow", "cash", "borrowings", "receivables", "inventories",
                        "assets", "equity", "net_income", "operating_income", "gross_profit",
                        "employees", "revenue"}
    for name in names:
        candidates = []
        required = _hw_required_terms(question, name, metric) if metric in numeric_measures and len(names) == 1 and doc_type == "annual_report" else set()
        for i in index["groups"][_hw_key(name)]:
            chunk = index["chunks"][i]
            meta, body = chunk.metadata, chunk.page_content
            if meta.get("doc_type") != doc_type:
                continue
            is_table = meta.get("kind") == "table"
            years = meta.get("data_years", [])
            if year is not None:
                if is_table and years:
                    if year not in years:
                        continue
                elif meta.get("fiscal_year") != year and not re.search(str(year), body):
                    continue
            body_tokens = set(_hw_tokens(body))
            if required and not required.issubset(body_tokens):
                continue
            row = meta.get("row_label", "").casefold()
            caption = meta.get("table_caption", "").casefold()
            section = meta.get("section", "").casefold()
            score = float(scores[i]) * 0.15
            supported = False
            if metric in numeric_measures:
                if metric == "employees":
                    supported = is_table and "employee" in caption and bool(re.search(row_pattern, row))
                elif metric == "revenue" and re.search(r"segment|\bregion\b|generate in", question, re.I):
                    supported = is_table and "revenue" in caption and ("segment" in caption or "region" in caption)
                else:
                    supported = is_table and bool(re.search(row_pattern, row))
                if supported:
                    score += 100
                    if metric == "cash_flow" and "cash flow" in caption:
                        score += 10
                    if metric == "receivables" and "balance sheet" in caption:
                        score += 10
                    if metric == "revenue" and "income statement" in caption:
                        score += 10
            elif metric in {"cfo", "ceo"}:
                supported = bool(re.search(row_pattern, body, re.I))
                score += 100 if supported else 0
                if supported and is_table and "executive" in caption:
                    score += 30
            elif metric == "auditor":
                supported = "audited" in body.casefold() and meta.get("kind") == "text"
                score += 100 if supported else 0
            elif metric == "headquarters":
                supported = bool(re.search(row_pattern, body, re.I))
                score += 100 if supported else 0
            elif metric in {"target", "rating", "repurchase", "growth"}:
                supported = bool(re.search(row_pattern, body, re.I))
                score += 100 if supported else 0
                if metric == "rating" and body.split("\n")[-1].casefold().startswith("rating:"):
                    score += 20
            else:
                terms = set(_hw_tokens(question)) - set(_hw_tokens(name))
                terms = {t for t in terms if t.isalpha()}
                supported = bool(terms) and terms.issubset(body_tokens)
                score += 100 if supported else 0
            if not supported:
                continue
            if year is not None and meta.get("fiscal_year") == year:
                score += 5
            # Prefer compact, unambiguous source passages over broad narrative.
            score -= len(body) / 10000
            candidates.append((score, i))
        if not candidates:
            return []
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        candidates = [pair for pair in candidates if pair[0] >= candidates[0][0] - 15]
        ranked_groups.append(candidates)
    # Round robin guarantees evidence for every named company in comparisons.
    result, seen = [], set()
    per_company = max(1, MAX_RETRIEVED // len(ranked_groups))
    for rank in range(min(per_company, 3)):
        for group in ranked_groups:
            if rank < len(group):
                chunk = index["chunks"][group[rank][1]]
                signature = (chunk.metadata["source"], chunk.page_content)
                if signature not in seen:
                    result.append(chunk)
                    seen.add(signature)
    return result[:MAX_RETRIEVED]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()

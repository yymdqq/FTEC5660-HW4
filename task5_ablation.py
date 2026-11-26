"""Cumulative, one-component-at-a-time HW4 ablation. The submitted hw4.py is never edited.

Offline: python task5_ablation.py --project PATH
Online:  python task5_ablation.py --project PATH --online

Only the five allowed student functions are temporarily replaced in memory for
experimental variants. The teacher generator, prompt, model, main and scorers
remain unchanged. Separate subprocesses reset each experiment's state and usage.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
FEATURES = ("tables", "tokenizer", "metadata", "row_chunks", "headers", "metric_rank", "balance", "evidence_budget", "abstention")
CHANGES = (
    "Naive RAG baseline",
    "HTML tables to labelled text",
    "Financial tokenizer",
    "Query parsing and metadata filter",
    "HTML paragraphs and table-row chunks",
    "Contextual chunk headers",
    "Financial-metric ranking",
    "Per-company balanced selection",
    "Compact evidence budget",
    "Evidence-based abstention",
)
VARIANTS = [{"id": "B0" if i == 0 else f"V{i}", "change": CHANGES[i],
             **{name: j < i for j, name in enumerate(FEATURES)}} for i in range(len(CHANGES))]
CSV_FIELDS = ["id", "category", "question", "parsed", "parse_score", "retrieved", "valid", "retrieval_hit", "answer", "correct", "expected"]
TEACHER_MARKER = b"# =============================================================================\n# Provided code below."


def load_hw4(project: Path):
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location("hw4_task5_under_test", project / "hw4.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_digest(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data[data.index(TEACHER_MARKER):]).hexdigest()


def dataset_digest(folder: Path) -> str:
    result = hashlib.sha256()
    paths = sorted((folder / "corpus").glob("*.html"))
    paths += [folder / name for name in ("questions.json", "ground_truth.json", "tables.json", "docs.json")]
    for path in paths:
        result.update(str(path.relative_to(folder)).replace("\\", "/").encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def validate_ladder() -> None:
    for before, after in zip(VARIANTS, VARIANTS[1:]):
        changed = [name for name in FEATURES if before[name] != after[name]]
        if len(changed) != 1:
            raise ValueError(f"Invalid comparison: {before['id']} to {after['id']}")


def candidate_function(h, abstention: bool):
    """Reuse the submitted ranker without editing its source file or teacher code.

    The two support filters and confidence pruning are the abstention component;
    their positive metric scores remain active before abstention is enabled.
    Selection is handled separately so balancing changes no scoring component.
    """
    tree = ast.parse(inspect.getsource(h.retrieve))
    fn = tree.body[0]
    fn.name = "_task5_candidate_groups"
    selection = next(i for i, stmt in enumerate(fn.body)
                     if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Tuple)
                     and [getattr(x, "id", "") for x in stmt.targets[0].elts] == ["result", "seen"])
    fn.body = fn.body[:selection] + [ast.Return(value=ast.Name(id="ranked_groups", ctx=ast.Load()))]

    class Controls(ast.NodeTransformer):
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if not abstention and test in {"required and (not required.issubset(body_tokens))", "not supported"}:
                return None
            # ast.unparse omits redundant parentheses on some Python versions.
            if not abstention and test == "required and not required.issubset(body_tokens)":
                return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if (not abstention and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == "candidates" and
                isinstance(node.value, ast.ListComp)):
                return None
            return self.generic_visit(node)

    tree = Controls().visit(tree)
    ast.fix_missing_locations(tree)
    scope = {}
    exec(compile(tree, "<task5 student-ranker controls>", "exec"), h.__dict__, scope)
    return scope["_task5_candidate_groups"]


def install_variant(h, config: dict) -> dict:
    from bs4 import BeautifulSoup
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi

    original = {name: getattr(h, name) for name in
                ("table_to_text", "chunk_document", "parse_query", "build_index", "retrieve")}
    preserved = {name: getattr(h, name) for name in ("generate", "format_context", "main", "make_llm", "_chat")}
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, length_function=len)

    def table_to_text(html):
        if config["tables"]:
            return original["table_to_text"](html)
        return BeautifulSoup(html, "html.parser").get_text()

    def chunk_document(file_name, html):
        if config["row_chunks"]:
            chunks = original["chunk_document"](file_name, html)
            if not config["headers"]:
                chunks = [Document(page_content=c.page_content.split("\n", 1)[1], metadata=dict(c.metadata)) for c in chunks]
            return chunks
        soup = BeautifulSoup(html, "html.parser")
        metadata = h._hw_metadata(file_name, soup)
        h._HW_COMPANIES[h._hw_key(metadata["company"])] = metadata["company"]
        if config["tables"]:
            for table in soup.find_all("table"):
                table.replace_with("\n" + original["table_to_text"](str(table)) + "\n")
        # The B0 baseline uses precisely BeautifulSoup.get_text(), no table conversion.
        return [Document(page_content=text, metadata=dict(metadata))
                for text in splitter.split_text(soup.get_text())]

    tokenize = h._hw_tokens if config["tokenizer"] else lambda text: text.lower().split()

    def build_index(chunks):
        groups = {}
        for i, c in enumerate(chunks):
            groups.setdefault(h._hw_key(c.metadata["company"]), []).append(i)
        tokens = [tokenize(c.page_content) or [""] for c in chunks]
        return {"chunks": chunks, "groups": groups,
                "bm25": BM25Okapi(tokens, k1=1.4, b=.65) if chunks else None}

    rank_groups = candidate_function(h, abstention=False) if config["metric_rank"] and not config["abstention"] else None

    def retrieve(index, question):
        if config["abstention"]:
            return original["retrieve"](index, question)
        chunks = index["chunks"]
        if not chunks:
            return []
        scores = index["bm25"].get_scores(tokenize(question))
        global_order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))
        if not config["metadata"]:
            return [chunks[i] for i in global_order[:h.MAX_RETRIEVED]]
        query = original["parse_query"](question)
        if config["metric_rank"]:
            groups = rank_groups(index, question)
        else:
            groups = []
            for name in query["companies"]:
                candidates = []
                for i in index["groups"].get(h._hw_key(name), []):
                    c = chunks[i]
                    if c.metadata["doc_type"] != query["doc_type"]:
                        continue
                    year = query["fiscal_year"]
                    years = c.metadata.get("data_years", [])
                    if year is not None:
                        if c.metadata.get("kind") == "table" and years:
                            if year not in years:
                                continue
                        elif c.metadata["fiscal_year"] != year and not re.search(str(year), c.page_content):
                            continue
                    candidates.append((float(scores[i]), i))
                candidates.sort(key=lambda x: (-x[0], x[1]))
                groups.append(candidates)
        # All pre-final variants deliberately keep a global fallback and never abstain.
        if not groups or any(not group for group in groups):
            return [chunks[i] for i in global_order[:h.MAX_RETRIEVED]]
        if config["balance"]:
            selected = []
            if config["evidence_budget"]:
                per_company = min(3, max(1, h.MAX_RETRIEVED // len(groups)))
            else:
                per_company = h.MAX_RETRIEVED
            for rank in range(per_company):
                for group in groups:
                    if rank < len(group):
                        selected.append(group[rank][1])
            return [chunks[i] for i in selected[:h.MAX_RETRIEVED]]
        flattened = sorted((x for group in groups for x in group), key=lambda x: (-x[0], x[1]))
        # V6 and V7 use the same candidate scores and the same maximum budget (5).
        # Their sole difference is whether that budget is allocated per company.
        return [chunks[i] for _, i in flattened[:h.MAX_RETRIEVED]]

    h.table_to_text = table_to_text
    h.chunk_document = chunk_document
    h.parse_query = original["parse_query"] if config["metadata"] else lambda question: {}
    h.build_index = build_index
    h.retrieve = retrieve
    for name, obj in preserved.items():
        assert getattr(h, name) is obj, f"Teacher function {name} was changed"
    return original


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare(h, folder: Path):
    started = time.perf_counter()
    tables = h.load_json(folder / "tables.json")
    docs = h.load_json(folder / "docs.json")
    gold = h.load_json(folder / "ground_truth.json")
    questions = h.load_json(folder / "questions.json")
    before = h.USAGE.llm_calls
    table_scores = [h.table_score(h.table_to_text(t["html"]), t) for t in tables]
    by_file = {name: h.chunk_document(name, html) for name, html in h.load_corpus(folder / "corpus").items()}
    doc_problems = {name: h.document_problems(by_file[name], None, meta, name) for name, meta in docs.items()}
    assert h.USAGE.llm_calls == before, "Forbidden model call in parsing or chunking"
    chunks = [c for name in sorted(by_file) for c in by_file[name]]
    parsed = {qid: h.parse_query(q) for qid, q in questions.items()}
    index = h.build_index(chunks)
    index_seconds = time.perf_counter() - started
    retrieved = {qid: h.retrieve(index, q) for qid, q in questions.items()}
    identities = {id(c) for c in chunks}
    rows, evidence = [], {}
    for qid, q in questions.items():
        g, cs = gold[qid], retrieved[qid]
        valid = isinstance(cs, list) and len(cs) <= h.MAX_RETRIEVED and all(id(c) in identities for c in cs)
        sources = sorted({c.metadata["source"] for c in cs})
        rows.append({"id": qid, "category": g["category"], "question": q,
                     "parsed": json.dumps(parsed[qid]), "parse_score": round(h.parse_score(parsed[qid], g), 3),
                     "retrieved": ";".join(sources), "valid": valid,
                     "retrieval_hit": h.retrieval_hit(cs, g) if g["evidence"] else "",
                     "answer": "", "correct": "", "expected": ""})
        missing = [i + 1 for i, group in enumerate(g["evidence"])
                   if not any(c.metadata["source"] == source and h.contains(c.page_content, text)
                              for c in cs for source, text in group)]
        evidence[qid] = {"question": q, "category": g["category"], "answerable": bool(g["evidence"]),
                         "expected_companies": g["query"]["companies"], "requested_year": g["query"]["fiscal_year"],
                         "required_facts": g["evidence"],
                         "missing_fact_groups": missing, "chunks": [
                             {"source": c.metadata["source"], "company": c.metadata["company"],
                              "fiscal_year": c.metadata["fiscal_year"], "content": c.page_content} for c in cs]}
    answerable = [r for r in rows if gold[r["id"]]["evidence"]]
    unanswerable = [r for r in rows if not gold[r["id"]]["evidence"]]
    recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
    kept = sum(bool(retrieved[r["id"]]) for r in answerable) / max(len(answerable), 1)
    rejected = sum(not retrieved[r["id"]] for r in unanswerable) / max(len(unanswerable), 1)
    points = {"1.1": 8 * sum(table_scores) / len(table_scores),
              "1.2": 7 * sum(not x for x in doc_problems.values()) / len(doc_problems),
              "2.1": 5 * sum(r["valid"] and bool(retrieved[r["id"]]) for r in answerable) / max(len(answerable), 1),
              "2.2": 10 * sum(r["parse_score"] for r in rows) / len(rows),
              "2.3": 15 * h.clip((recall - .5) / .4), "3": 10 * h.clip(kept + rejected - 1)}
    category_recall = {category: sum(r["retrieval_hit"] for r in answerable if r["category"] == category) /
                       sum(r["category"] == category for r in answerable)
                       for category in sorted({r["category"] for r in answerable})}
    metrics = {"python": platform.python_version(), "documents": len(by_file), "questions": len(rows),
               "chunks": len(chunks), "recall_at_5": recall, "kept": kept, "rejected": rejected,
               "parse_accuracy": sum(r["parse_score"] for r in rows) / len(rows),
               "category_recall": category_recall, "points_offline": points,
               "offline_points": sum(points.values()), "offline_maximum": 55,
               "index_seconds": index_seconds, "offline_seconds": time.perf_counter() - started,
               "answer_accuracy": None, "end_to_end_seconds": None,
               "student_usage": h.USAGE.summary(), "generator_usage": h.GENERATOR_USAGE.summary()}
    return metrics, rows, evidence, index, retrieved, questions


def child(args) -> int:
    h = load_hw4(args.project)
    config = next(v for v in VARIANTS if v["id"] == args.variant)
    before_hash = digest(args.project / "hw4.py")
    teacher_hash = protected_digest(args.project / "hw4.py")
    original = install_variant(h, config)
    output = args.output / config["id"]
    output.mkdir(parents=True, exist_ok=True)
    metrics, rows, evidence, index, retrieved, questions = prepare(h, args.project / args.data_folder)
    metrics.update(variant=config["id"], change=config["change"], features={k: config[k] for k in FEATURES},
                   hw4_sha256=before_hash, provided_sha256=teacher_hash,
                   script_sha256=digest(Path(__file__)), dataset_sha256=dataset_digest(args.project / args.data_folder))
    # The final configuration must return exactly the submitted implementation's passages.
    if config["id"] == VARIANTS[-1]["id"]:
        reference_index = original["build_index"](index["chunks"])
        for qid, q in questions.items():
            expected = original["retrieve"](reference_index, q)
            actual = retrieved[qid]
            signature = lambda cs: [(c.metadata["source"], c.page_content) for c in cs]
            assert signature(actual) == signature(expected), f"Final retrieval differs for {qid}"
        metrics["final_matches_submitted_retrieval"] = True
    write_csv(output / "results_offline.csv", rows)
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.online:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY is missing; put it in your project's local .env")
        started = time.perf_counter()
        original_argv = sys.argv[:]
        try:
            # This calls the unchanged official runner, including the unchanged fixed generator.
            sys.argv = [str(args.project / "hw4.py"), "--data-folder", str(args.project / args.data_folder),
                        "--output", str(output / "results.csv")]
            h.main()
        finally:
            sys.argv = original_argv
        metrics["end_to_end_seconds"] = time.perf_counter() - started
        with (output / "results.csv").open(encoding="utf-8", newline="") as f:
            online_rows = list(csv.DictReader(f))
        offline_map = {r["id"]: r for r in rows}
        for row in online_rows:
            baseline = offline_map[row["id"]]
            assert row["retrieved"] == baseline["retrieved"], "Retrieval changed between offline and online passes"
            assert row["retrieval_hit"] == str(baseline["retrieval_hit"]), "Recall changed between passes"
        metrics["answer_accuracy"] = sum(r["correct"] == "True" for r in online_rows) / len(online_rows)
        metrics["wrong_answer_ids"] = [r["id"] for r in online_rows if r["correct"] != "True"]
        metrics["category_accuracy"] = {category: sum(r["correct"] == "True" for r in online_rows if r["category"] == category) /
                                        sum(r["category"] == category for r in online_rows)
                                        for category in sorted({r["category"] for r in online_rows})}
        metrics["student_usage"] = h.USAGE.summary()
        metrics["generator_usage"] = h.GENERATOR_USAGE.summary()
        metrics["generator_error_count"] = 0  # parent additionally scans the official log
        metrics["status"] = "ONLINE_COMPLETE"
    else:
        metrics["status"] = "OFFLINE_ONLY: generator accuracy and API time not measured"
    assert digest(args.project / "hw4.py") == before_hash, "The submitted source was changed"
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n{config['id']}: recall={metrics['recall_at_5']:.1%}; kept={metrics['kept']:.1%}; "
          f"rejected={metrics['rejected']:.1%}; offline points={metrics['offline_points']:.2f}/55", flush=True)
    return 0


def percent(value) -> str:
    return "Not measured" if value is None else f"{value:.1%}"


def collect(output: Path) -> list[dict]:
    return [json.loads((output / c["id"] / "metrics.json").read_text(encoding="utf-8"))
            for c in VARIANTS if (output / c["id"] / "metrics.json").exists()]


def error_counts(output: Path, variant: str) -> dict:
    directory = output / variant
    with (directory / "results_offline.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (directory / "evidence.json").open(encoding="utf-8") as f:
        evidence = json.load(f)
    answerable = [r for r in rows if evidence[r["id"]]["answerable"]]
    unanswerable = [r for r in rows if not evidence[r["id"]]["answerable"]]
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    counts = {"variant": variant,
              "answerable_retrieval_misses": sum(r["retrieval_hit"] == "False" for r in answerable),
              "answerable_false_abstentions": sum(not evidence[r["id"]]["chunks"] for r in answerable),
              "unanswerable_false_keeps": sum(bool(evidence[r["id"]]["chunks"]) for r in unanswerable),
              "wrong_answer_with_retrieval_hit": None, "wrong_answer_with_retrieval_miss": None,
              "unanswerable_wrong_answers": None}
    if metrics.get("answer_accuracy") is not None:
        with (directory / "results.csv").open(encoding="utf-8") as f:
            online = list(csv.DictReader(f))
        counts.update(wrong_answer_with_retrieval_hit=sum(
            r["correct"] == "False" and r["retrieval_hit"] == "True" for r in online),
            wrong_answer_with_retrieval_miss=sum(
                r["correct"] == "False" and r["retrieval_hit"] == "False" for r in online),
            unanswerable_wrong_answers=sum(r["correct"] == "False" and not evidence[r["id"]]["answerable"] for r in online))
    return counts


def build_analysis(metrics: list[dict], output: Path) -> tuple[list[str], list[str]]:
    changes, errors = [], []
    for previous, current in zip(metrics, metrics[1:]):
        delta = (current["recall_at_5"] - previous["recall_at_5"]) * 100
        rejected_delta = (current["rejected"] - previous["rejected"]) * 100
        sentence = (f"**{previous['variant']} → {current['variant']} — {current['change']}.** "
                    f"Recall@5 changed by {delta:+.1f} percentage points; unanswerable rejection changed by "
                    f"{rejected_delta:+.1f} points.")
        if current["answer_accuracy"] is not None and previous["answer_accuracy"] is not None:
            sentence += f" Answer accuracy changed by {(current['answer_accuracy'] - previous['answer_accuracy']) * 100:+.1f} points."
        explanations = {
            "V1": "The only change is table serialization; the splitter, metadata, whitespace tokens and BM25 settings stay fixed. Labels and years become explicit, although repeated headers can also change lexical scores.",
            "V2": "Only tokenization changes: financial words, numbers, stop words and plural normalization replace lower-cased whitespace tokens. Better matching is a hypothesis; the measured delta determines whether it helped here.",
            "V3": "Only the query parser and metadata candidate filter are enabled. The requested year may be a prior-year table column or a forecast year, rather than the document fiscal year. Empty filters still use a global fallback, keeping abstention disabled.",
            "V4": "Only the chunking component changes to HTML paragraphs and table rows. Each financial row carries its caption and original column labels. Inline contextual headers remain disabled, so this comparison isolates chunk boundaries and row metadata.",
            "V5": "Only the inline contextual chunk header is added. Required metadata and the generator's existing metadata header are present in every variant. Extra inline text can help context but can also dilute lexical ranking.",
            "V6": "Only the candidate-ranking function changes. Original metric labels and table captions receive relevance bonuses; no candidate is rejected merely for lacking metric support yet, and balancing remains disabled.",
            "V7": "Only selection changes from globally merging company candidates to round-robin allocation per company. Candidate scores and the maximum evidence budget remain identical to V6; compare the comparison-category recall specifically.",
            "V8": "Only the evidence budget changes: at most three passages for one company, or two per company in a two-company comparison. Candidate scoring, balanced allocation and the pre-abstention global fallback remain unchanged. This tests whether a smaller context can retain all required facts.",
            "V9": "Only evidence validation and abstention are enabled: exact entity availability, year/metric/qualifier support, and confidence pruning determine whether to return passages or an empty list. The global fallback is removed. Check kept together with rejected, because rejecting everything would score zero.",
        }
        sentence += " " + explanations[current["variant"]]
        if delta == 0:
            sentence += " This run shows no marginal recall gain for this change; it should not be described as an improvement in recall."
        elif delta < 0:
            sentence += " Recall decreased in this controlled comparison; the regression is retained and reported."
        changes.append(sentence)
        prev_path, cur_path = output / previous["variant"] / "results_offline.csv", output / current["variant"] / "results_offline.csv"
        with prev_path.open(encoding="utf-8") as f:
            before = {r["id"]: r for r in csv.DictReader(f)}
        with cur_path.open(encoding="utf-8") as f:
            after = {r["id"]: r for r in csv.DictReader(f)}
        repaired = [qid for qid in before if before[qid]["retrieval_hit"] == "False" and after[qid]["retrieval_hit"] == "True"]
        regressed = [qid for qid in before if before[qid]["retrieval_hit"] == "True" and after[qid]["retrieval_hit"] == "False"]
        if repaired or regressed:
            changes[-1] += (f" Repaired question IDs: {', '.join(repaired) or 'none'}; "
                            f"regressed IDs: {', '.join(regressed) or 'none'}.")
    if metrics:
        last = metrics[-1]
        first_evidence = json.loads((output / metrics[0]["variant"] / "evidence.json").read_text(encoding="utf-8"))
        last_evidence = json.loads((output / last["variant"] / "evidence.json").read_text(encoding="utf-8"))
        for qid, initial in first_evidence.items():
            final = last_evidence[qid]
            if initial["answerable"] and initial["missing_fact_groups"] and not final["missing_fact_groups"]:
                wrong_sources = sorted({c["company"] for c in initial["chunks"]})
                final_sources = sorted({c["company"] for c in final["chunks"]})
                errors.append(f"**{qid} ({initial['category']})** — {initial['question']} "
                              f"B0 missed required fact group(s) {initial['missing_fact_groups']} and retrieved company metadata "
                              f"{wrong_sources}; {last['variant']} covered every required fact with company metadata {final_sources}. "
                              f"The exact passages are in `{output.name}/{metrics[0]['variant']}/evidence.json` and "
                              f"`{output.name}/{last['variant']}/evidence.json`. This is an observed retrieval failure, not proof of a wrong generated answer.")
                present_companies = {re.sub(r"[^a-z0-9 ]", "", c["company"].lower()) for c in initial["chunks"]}
                absent = [name for name in initial["expected_companies"]
                          if not any(name.lower() in company for company in present_companies)]
                trapped = [text for group in initial["required_facts"] for source, text in group
                           if any(c["source"] == source and text in c["content"] for c in initial["chunks"])
                           and not any(re.search(r"(?<![\d.,])" + re.escape(text) + r"(?![\d]|[.,]\d)", c["content"])
                                       for c in initial["chunks"] if c["source"] == source)]
                if absent:
                    errors[-1] += " Observed failure mechanism: no retrieved passage carries the required company metadata for " + ", ".join(absent) + "."
                elif trapped:
                    errors[-1] += " Observed failure mechanism: required numeric text appears inside concatenated cell text rather than as a whole numeric value (" + ", ".join(sorted(set(trapped))) + ")."
                else:
                    errors[-1] += " The required whole fact is absent from the retrieved passages; the evidence alone does not establish whether ranking, chunk boundaries or year selection is the sole cause."
                if len(errors) == 3:
                    break
        false_keeps = [qid for qid, e in first_evidence.items() if not e["answerable"] and e["chunks"]]
        fixed_rejections = [qid for qid in false_keeps if not last_evidence[qid]["chunks"]]
        if fixed_rejections:
            qid = fixed_rejections[0]
            errors.append(f"**{qid} (unanswerable)** — {first_evidence[qid]['question']} "
                          f"B0 returned {len(first_evidence[qid]['chunks'])} passages despite the absent answer; "
                          f"{last['variant']} returned an empty list. The same before/after evidence files contain this case.")
        remaining = [qid for qid, e in last_evidence.items() if (e["answerable"] and e["missing_fact_groups"]) or
                     (not e["answerable"] and e["chunks"])]
        errors.append(f"Remaining final retrieval/abstention failures on this public set: {', '.join(remaining) or 'none'}. "
                      "This does not establish private-set performance. Rules may fail on new wording, an unrecognized metric, or a different document layout.")
        if last.get("answer_accuracy") is None:
            errors.append("Generated-answer errors were not measured in this offline run. Correct evidence does not guarantee that the fixed generator selects the right year, current officer, or forecast revision.")
        elif last.get("wrong_answer_ids"):
            errors.append("Final generated-answer failures: " + ", ".join(last["wrong_answer_ids"]) + ". Inspect `results.csv` and the evidence before attributing them to retrieval or generation.")
        else:
            errors.append("The completed final online run has no generated-answer errors on these 60 public questions; the preceding variants and their per-question answers remain necessary to evaluate each design change.")
        if any(m.get("answer_accuracy") is not None for m in metrics):
            errors.append("The table below separates generated-answer failures with a retrieval hit from failures with missing evidence. "
                          "A hit plus a wrong answer is consistent with a context ambiguity, year/column selection or generation failure; "
                          "it does not prove the LLM alone is responsible. `error_breakdown.csv` contains all counts.")
    return changes, errors


def write_report(args) -> None:
    metrics = collect(args.output)
    write_csv(args.output / "ablation_summary.csv", [
        {"variant": m["variant"], "change": m["change"], "chunks": m["chunks"],
         "recall_at_5": m["recall_at_5"], "comparison_recall": m["category_recall"].get("comparison"),
         "kept": m["kept"], "rejected": m["rejected"], "parse_accuracy": m["parse_accuracy"],
         "offline_points_out_of_55": m["offline_points"], "offline_seconds": m["offline_seconds"],
         "answer_accuracy": m["answer_accuracy"], "end_to_end_seconds": m["end_to_end_seconds"],
         "generator_calls": m["generator_usage"]["llm_calls"], "status": m["status"]} for m in metrics])
    expected_count = len(VARIANTS)
    fingerprints = {(m["hw4_sha256"], m["dataset_sha256"], m["script_sha256"]) for m in metrics}
    if len(fingerprints) > 1:
        raise ValueError("Experiment fingerprints differ. Rerun the variants together before comparing them.")
    complete = len(metrics) == expected_count and all(m.get("status") == "ONLINE_COMPLETE" and m.get("generator_error_count", 0) == 0 for m in metrics)
    breakdown = [error_counts(args.output, m["variant"]) for m in metrics]
    write_csv(args.output / "error_breakdown.csv", breakdown)
    changes, errors = build_analysis(metrics, args.output)
    lines = ["## Homework 4 solution", "", "### Pipeline and design", "", "```mermaid", "flowchart TD",
             '  A["HTML reports"] --> B["Parse headers, paragraphs and tables"]',
             '  B --> C["Source-only chunks and metadata"]',
             '  C --> D["Lexical BM25 index"]',
             '  Q["Question"] --> P["Parse companies, year and source type"]',
             '  P --> R["Filter and rank evidence"]',
             '  D --> R',
             '  R --> S{"Sufficient supported evidence?"}',
             '  S -->|Yes| T["Balance passages across companies"]',
             '  T --> G["Fixed DeepSeek generator"]',
             '  S -->|No| N["NOT_FOUND"]',
             '  G --> O["Number or name"]', "```", "",
             "My pipeline extracts the report company, fiscal year and document type from the HTML heading, "
             "serializes each table with its original caption, row labels, column labels and values, and keeps "
             "paragraphs and table rows in source-only chunks below 1,500 characters. Every chunk carries source, "
             "company, fiscal_year and doc_type metadata; contextual headers make the source and section explicit. "
             "A BM25 index uses financial tokenization without embeddings. The question parser identifies exact "
             "company names, including companies absent from the corpus, the requested year and the requested "
             "document type. Retrieval filters candidates by this metadata, recognizes that a requested year may "
             "be a prior-year column or a forecast, and ranks financial rows and personnel passages using their "
             "metric labels and captions. Per-company selection protects comparisons from retrieving passages "
             "about only one company. Evidence checks reject missing companies, unsupported years, undisclosed "
             "metrics and unsupported qualifiers; sufficient evidence is passed to the unchanged fixed generator, "
             "otherwise an empty list produces NOT_FOUND. This design addresses wrong-company matches, lost "
             "row/column bindings, outdated officers, forecasts and unsupported questions while avoiding extra model calls.",
             "", "### Controlled experimental setup", "",
             "All variants use the same 40 public documents and 60 questions, the same provided scoring functions, "
             "the same fixed generator (deepseek-v4-flash, temperature 0), and at most five retrieved chunks. "
             "BM25 k1=1.4 and b=0.65 are fixed across every row; B0 uses lower-cased whitespace tokens. "
             "The RecursiveCharacterTextSplitter has chunk_size=1200 and chunk_overlap=150 and stays unchanged "
             "until the chunking component is replaced. Every row retains the required document metadata. "
             "The teacher-supplied generator already formats metadata headers for every variant; V5 changes only "
             "the additional inline header inside chunk text. Every pre-V9 variant retains a global BM25 fallback "
             "when metadata yields no candidates, so abstention is introduced only at V9. "
             "V6 and V7 use identical candidate scores and equal maximum evidence budgets; only company allocation changes. "
             "V8 then changes the budget alone. For V9, a check asserts that all 60 retrieved passage lists exactly match the submitted implementation. "
             "Each adjacent feature vector differs in exactly one boolean component, verified before execution. "
             "No benchmark answers or evidence labels are used to select retrieved chunks; labels are read only for evaluation.",
             "", f"Python: {metrics[0]['python'] if metrics else 'not run'}. "
             f"Submitted hw4.py SHA-256: `{metrics[0]['hw4_sha256'] if metrics else 'not run'}`. "
             "The script verifies that this file and its teacher-provided section remain unchanged.",
             "", "### Ablation results", "",
             "**Status: " + ("all ten online variants completed without generator errors." if complete else
             "offline ablation completed; online accuracy/time is not yet fully measured. This is a draft until all online runs finish.") + "**",
             "", "Recall@5 requires every needed fact of an answerable question to appear in retrieved evidence. "
             "Kept is the answerable retention rate; rejected is the unanswerable rejection rate. "
             "Offline time includes parsing, chunking, indexing, query parsing, retrieval and offline evaluation, "
             "but excludes generation. Online time covers the unchanged official main(), including generation; "
             "it excludes dependency installation and the separate offline diagnostic pass. All timings are one-run "
             "observations, not guarantees. Missing online values are not estimates.", "",
             "| Variant | One change from preceding row | Chunks | Recall@5 | Comparison recall | Kept | Rejected | Parse | Offline points /55 | Offline s | Answer accuracy | Online s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in metrics:
        online_time = "Not measured" if m["end_to_end_seconds"] is None else f"{m['end_to_end_seconds']:.2f}"
        lines.append(f"| {m['variant']} | {m['change']} | {m['chunks']} | {percent(m['recall_at_5'])} | "
                     f"{percent(m['category_recall'].get('comparison'))} | {percent(m['kept'])} | {percent(m['rejected'])} | "
                     f"{percent(m['parse_accuracy'])} | {m['offline_points']:.2f} | {m['offline_seconds']:.2f} | "
                     f"{percent(m['answer_accuracy'])} | {online_time} |")
    lines += ["", "B0 is the required naive baseline: BeautifulSoup.get_text(), no table conversion, "
              "RecursiveCharacterTextSplitter, required metadata, global whitespace-token BM25 top-5, "
              "no query parsing and no abstention. Its low Task 1.1 or 2.2 scores are expected baseline behavior, "
              "not a claim that these components were implemented incorrectly in the submitted final system.",
              "", "### Analysis of each change", ""]
    for item in changes:
        lines += [item, ""]
    lines += ["### Error analysis and limits", ""]
    for item in errors:
        lines += [item, ""]
    lines += ["| Variant | Answerable retrieval misses /46 | False abstentions /46 | Unanswerable false keeps /14 | Wrong answer with retrieval hit | Wrong answer with retrieval miss | Unanswerable wrong answers |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for b in breakdown:
        values = [str(v) if v is not None else "Not measured" for key, v in b.items() if key != "variant"]
        lines.append("| " + b["variant"] + " | " + " | ".join(values) + " |")
    lines += [""]
    lines += ["One cumulative run per variant cannot establish statistical significance or isolate interactions "
              "among components beyond the chosen order. In particular, equal Recall@5 does not establish equal "
              "answer accuracy, and perfect public results do not establish private performance. "
              "The small, structured fictional corpus favors rule-based metric and entity handling; a broader "
              "corpus could require more robust parsing or semantic retrieval, which is outside this assignment's rules.",
              "", "### Reproduction and evidence", "",
              "Install the original requirements, set DEEPSEEK_API_KEY in the local .env, then run:", "", "```powershell",
              'python task5_ablation.py --project . --online --write-readme', "```", "",
              "The script runs ten variants sequentially in isolated subprocesses, each with a 29-minute "
              "external limit. A timed-out or failed run is marked incomplete and is not reused as a valid result. "
              "Ten experiments may take more than 30 minutes in total; the assignment's limit applies to each run. "
              "No model, prompt, scoring function, teacher runner or submitted student source file is changed. "
              "Only the five permitted student function bindings change temporarily inside each experimental process.", "",
              f"Evidence directory: `{args.output.name}/`. Each variant contains `results_offline.csv`, "
              "`evidence.json`, `metrics.json` and `run_log.txt`. After an online run it additionally contains "
              "the official runner's `results.csv` with generated answers and correctness flags. "
              "`ablation_summary.csv` aggregates the measured values. Commit the script, README report and "
              "the experiment results used in the table; never commit .env or index files.", ""]
    report = "\n".join(lines)
    (args.output / "Homework_4_solution.md").write_text(report, encoding="utf-8")
    (args.output / "run_status.json").write_text(json.dumps({"variants_expected": expected_count, "variants_available": len(metrics),
        "online_complete": complete, "pending_online": [m["variant"] for m in metrics if m.get("status") != "ONLINE_COMPLETE"],
        "note": "The report must not be described as a fully measured online ablation until online_complete is true."}, indent=2), encoding="utf-8")
    if args.write_readme:
        target = args.project / "README.md"
        existing = target.read_text(encoding="utf-8") if target.exists() else "# FTEC5660-HW4\n"
        start, end = "<!-- HW4_TASK5_START -->", "<!-- HW4_TASK5_END -->"
        section = start + "\n" + report + "\n" + end
        if start in existing and end in existing:
            left, tail = existing.split(start, 1)
            _, right = tail.split(end, 1)
            updated = left + section + right
        else:
            updated = existing.rstrip() + "\n\n" + section + "\n"
        if target.exists() and existing != updated:
            (args.output / "README_before_task5.md").write_text(existing, encoding="utf-8")
        target.write_text(updated, encoding="utf-8")
    print(f"Report: {args.output / 'Homework_4_solution.md'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--write-readme", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=[v["id"] for v in VARIANTS])
    parser.add_argument("--reuse", action="store_true", help="reuse complete compatible experiments; otherwise rerun")
    parser.add_argument("--variant", choices=[v["id"] for v in VARIANTS], help=argparse.SUPPRESS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.project = args.project.resolve()
    args.output = (args.output or args.project / "task5_results").resolve()
    validate_ladder()
    if not (args.project / "hw4.py").exists():
        parser.error("--project must contain your completed hw4.py")
    if args.child:
        return child(args)
    if args.online:
        from dotenv import load_dotenv
        load_dotenv(args.project / ".env")
        if not os.environ.get("DEEPSEEK_API_KEY"):
            parser.error("Online ablation needs your local DEEPSEEK_API_KEY; no files were changed")
    args.output.mkdir(parents=True, exist_ok=True)
    submitted_hash = digest(args.project / "hw4.py")
    data_hash = dataset_digest(args.project / args.data_folder)
    script_hash = digest(Path(__file__))
    (args.output / "experiment_manifest.json").write_text(json.dumps({
        "variants": VARIANTS, "fixed_settings": {"bm25_k1": 1.4, "bm25_b": .65,
        "recursive_chunk_size": 1200, "recursive_chunk_overlap": 150,
        "max_retrieved": 5, "external_limit_seconds_per_variant": 29 * 60},
        "hw4_sha256": submitted_hash, "dataset_sha256": data_hash, "script_sha256": script_hash}, indent=2), encoding="utf-8")
    failed = []
    for config in VARIANTS:
        if args.variants and config["id"] not in args.variants:
            continue
        directory = args.output / config["id"]
        metric_path = directory / "metrics.json"
        if args.reuse and metric_path.exists():
            old = json.loads(metric_path.read_text(encoding="utf-8"))
            compatible = (old.get("hw4_sha256") == submitted_hash and old.get("dataset_sha256") == data_hash and
                          old.get("script_sha256") == script_hash)
            if compatible and (not args.online or old.get("status") == "ONLINE_COMPLETE"):
                print(f"Reusing {config['id']}", flush=True)
                continue
        directory.mkdir(parents=True, exist_ok=True)
        # Remove stale metrics, so failed runs cannot masquerade as current results.
        if metric_path.exists():
            metric_path.unlink()
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--variant", config["id"],
                   "--project", str(args.project), "--data-folder", args.data_folder, "--output", str(args.output)]
        if args.online:
            command.append("--online")
        print(f"Running {config['id']}: {config['change']}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            with (directory / "run_log.txt").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=args.project, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=29 * 60)
            if completed.returncode != 0:
                failed.append(config["id"])
                print(f"FAILED {config['id']}; see {directory / 'run_log.txt'}", flush=True)
            elif args.online:
                text = (directory / "run_log.txt").read_text(encoding="utf-8")
                errors = text.count("Generator failed on ")
                if errors:
                    data = json.loads(metric_path.read_text(encoding="utf-8"))
                    data.update(status="ONLINE_API_ERRORS", generator_error_count=errors)
                    metric_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                    failed.append(config["id"])
                    print(f"API ERRORS {config['id']}: {errors}. Rerun this variant.", flush=True)
        except subprocess.TimeoutExpired:
            failed.append(config["id"])
            if metric_path.exists():
                metric_path.unlink()
            (directory / "FAILED_TIMEOUT.txt").write_text("Incomplete: exceeded the external 29-minute limit.\n", encoding="utf-8")
            print(f"TIMEOUT {config['id']}; not a successful result", flush=True)
        assert digest(args.project / "hw4.py") == submitted_hash, "hw4.py changed unexpectedly"
    write_report(args)
    if failed:
        print("Incomplete experiments: " + ", ".join(failed), flush=True)
        return 1
    print("Finished. Offline runs do not measure generator accuracy." if not args.online else
          "Finished. Check run_status.json and the generated README section.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

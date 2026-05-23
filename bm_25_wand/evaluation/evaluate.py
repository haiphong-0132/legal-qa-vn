from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from bm_25_wand.utils import tokenize_underthesea_text
from bm_25_wand.engine import BM25SearchEngine
from bm_25_wand.indexing import build_champion_index, load_tokenized_corpus
from bm_25_wand.wand_algorithm import WAND_Algo

TOP_KS = (1, 5, 10, 20)
MAX_EVAL_K = max(TOP_KS)


def read_jsonl(path: str | Path) -> Iterable[dict]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_queries(path: str | Path, tokenized: bool):
    queries: dict[str, list[str]] = {}
    for item in read_jsonl(path):
        query_id = item.get("qid", item.get("query-id", item.get("id", item.get("_id"))))
        if query_id is None:
            raise KeyError("Query records must contain 'qid', 'query-id', 'id', or '_id'.")

        if tokenized:
            tokens = item.get("tokens")
            if not isinstance(tokens, list):
                raise TypeError("Tokenized query records must contain a list field named 'tokens'.")
            query_terms = [str(tok) for tok in tokens if str(tok).strip()]
        else:
            text = str(item.get("text", ""))
            query_terms = list(tokenize_underthesea_text(text))

        queries[str(query_id)] = query_terms

    if not queries:
        raise ValueError("Query file is empty.")

    return queries


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for item in read_jsonl(path):
        query_id = item.get("query-id", item.get("qid", item.get("_id")))
        corpus_id = item.get("corpus-id", item.get("doc_id"))
        if query_id is None or corpus_id is None:
            raise KeyError("Qrels records must contain query and corpus identifiers.")
        qrels[str(query_id)].add(str(corpus_id))
    if not qrels:
        raise ValueError("Qrels file is empty.")
    return qrels


def bm25_search(
    query_terms: list[str],
    inverted_index: dict[str, list[tuple[int, float]]],
    champion_index: dict[str, list[tuple[int, float]]] | None = None,
) -> list[int]:
    candidate_scores: dict[int, float] = defaultdict(float)
    posting_source = champion_index if champion_index is not None else inverted_index

    for term in set(query_terms):
        posting_list = posting_source.get(term)
        if not posting_list:
            continue

        for doc_idx, score in posting_list:
            candidate_scores[doc_idx] += score

    if not candidate_scores:
        return []

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_idx for doc_idx, _ in ranked]


def wand_search(
    query_terms: list[str],
    top_k: int,
    inverted_index: dict[str, list[tuple[int, float]]],
    term_max_score: dict[str, float],
) -> tuple[list[int], int]:
    ranked, fully_evaluated = WAND_Algo(
        query_terms=query_terms,
        top_k=top_k,
        inverted_index=inverted_index,
        term_max_score=term_max_score,
    )
    return [doc_idx for _score, doc_idx in ranked], fully_evaluated


def compute_metrics(ranked_doc_ids: list[str], relevant_doc_ids: set[str]) -> dict[int, tuple[float, float, float]]:
    results: dict[int, tuple[float, float, float]] = {}
    relevant_count = len(relevant_doc_ids)

    for k in TOP_KS:
        top = ranked_doc_ids[:k]
        hits = [1 if doc_id in relevant_doc_ids else 0 for doc_id in top]

        recall = sum(hits) / relevant_count if relevant_count else 0.0

        rr = 0.0
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                rr = 1.0 / rank
                break

        dcg = 0.0
        for rank, is_hit in enumerate(hits, start=1):
            if is_hit:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(relevant_count, k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)) if ideal_hits else 0.0
        ndcg = dcg / idcg if idcg > 0 else 0.0

        results[k] = (recall, rr, ndcg)

    return results


def aggregate_metrics(rows: list[dict[int, tuple[float, float, float]]]) -> dict[int, tuple[float, float, float]]:
    totals = {k: [0.0, 0.0, 0.0] for k in TOP_KS}
    for row in rows:
        for k in TOP_KS:
            values = row[k]
            for idx in range(3):
                totals[k][idx] += values[idx]

    count = len(rows)
    return {k: tuple(value / count for value in totals[k]) for k in TOP_KS}


def evaluate(
    label: str,
    doc_ids: list[str],
    inverted_index: dict[str, list[tuple[int, float]]],
    queries: dict[str, list[str]],
    qrels: dict[str, set[str]],
    champion_index: dict[str, list[tuple[int, float]]] | None = None,
) -> dict[str, object]:
    query_ids = [query_id for query_id in queries if query_id in qrels]
    if not query_ids:
        raise ValueError("No overlapping query ids between queries and qrels.")

    times: list[float] = []
    per_query_rows: list[dict[int, tuple[float, float, float, float]]] = []

    start_total = time.perf_counter()
    for query_id in query_ids:
        start = time.perf_counter()
        ranked_indices = bm25_search(
            queries[query_id],
            inverted_index,
            champion_index=champion_index,
        )
        times.append(time.perf_counter() - start)
        ranked_doc_ids = [doc_ids[idx] for idx in ranked_indices[:MAX_EVAL_K]]
        per_query_rows.append(compute_metrics(ranked_doc_ids, qrels[query_id]))
    total_time = time.perf_counter() - start_total

    metrics = aggregate_metrics(per_query_rows)
    return {
        "label": label,
        "query_count": len(query_ids),
        "latency_ms": 1000.0 * (total_time / len(query_ids)),
        "p50_ms": 1000.0 * statistics.median(times),
        "p95_ms": 1000.0 * sorted(times)[max(0, int(0.95 * len(times)) - 1)],
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="champion_bm25.evaluate",
        description="Evaluate BM25 full scoring and BM25 champion-list scoring.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus-tokenized", help="Tokenized corpus JSONL with doc_id and tokens")
    source.add_argument("--model-dir", help="Directory containing prebuilt bm25_model.joblib")
    parser.add_argument("--qrels", required=True, help="Qrels JSONL with query-id and corpus-id")
    parser.add_argument("--queries-tokenized", help="Tokenized query JSONL with qid and tokens")
    parser.add_argument("--queries-raw", help="Raw query JSONL with _id/qid and text")
    parser.add_argument("--champion-size", type=int, default=0, help="Champion list size per term")
    parser.add_argument("--output", help="Optional output text file to save results")
    parser.add_argument(
        "--mode",
        choices=["full", "champion", "wand", "both"],
        default="both",
        help="Run full BM25, champion list BM25, WAND, or both",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.queries_tokenized:
        queries = load_queries(args.queries_tokenized, tokenized=True)
    elif args.queries_raw:
        queries = load_queries(args.queries_raw, tokenized=False)
    else:
        raise ValueError("Provide either --queries-tokenized or --queries-raw.")

    qrels = load_qrels(args.qrels)

    if args.model_dir:
        engine = BM25SearchEngine.load(args.model_dir)
        doc_ids = [str(doc.doc_id) for doc in engine.documents]
        inverted_index = engine.inverted_index
        prebuilt_champion_index = engine.champion_index
        champion_size = engine.champion_size
        term_max_score = engine.term_max_score
    else:
        doc_ids, doc_lengths, avgdl, inverted_index, idf, term_max_score = load_tokenized_corpus(
            args.corpus_tokenized
        )
        prebuilt_champion_index = None
        champion_size = args.champion_size

    champion_index = None
    champion_build_seconds = 0.0
    if args.mode in {"champion", "both"}:
        if prebuilt_champion_index is not None:
            champion_index = prebuilt_champion_index
        else:
            start_build = time.perf_counter()
            champion_index = build_champion_index(
                inverted_index=inverted_index,
                champion_size=args.champion_size,
            )
            champion_build_seconds = time.perf_counter() - start_build

    output_lines: list[str] = []
    output_lines.append(f"corpus_docs={len(doc_ids)}")
    output_lines.append(f"queries_loaded={len(queries)}")
    output_lines.append(f"qrels_queries={len(qrels)}")
    if champion_index is not None:
        output_lines.append(f"champion_size={champion_size}")
        output_lines.append(f"champion_build_s={champion_build_seconds:.2f}")

    rows = []
    if args.mode in {"full", "both"}:
        rows.append(
            evaluate(
                label="BM25 full",
                doc_ids=doc_ids,
                inverted_index=inverted_index,
                queries=queries,
                qrels=qrels,
                champion_index=None,
            )
        )

    if args.mode in {"champion", "both"}:
        rows.append(
            evaluate(
                label=f"BM25 champion={champion_size}",
                doc_ids=doc_ids,
                inverted_index=inverted_index,
                queries=queries,
                qrels=qrels,
                champion_index=champion_index,
            )
        )

    if args.mode == "wand":
        times: list[float] = []
        per_query_rows: list[dict[int, tuple[float, float, float]]] = []
        total_fully_evaluated = 0
        query_ids = [query_id for query_id in queries if query_id in qrels]
        if not query_ids:
            raise ValueError("No overlapping query ids between queries and qrels.")

        start_total = time.perf_counter()
        for query_id in query_ids:
            start = time.perf_counter()
            ranked_indices, fully_evaluated = wand_search(
                queries[query_id],
                top_k=MAX_EVAL_K,
                inverted_index=inverted_index,
                term_max_score=term_max_score,
            )
            times.append(time.perf_counter() - start)
            total_fully_evaluated += fully_evaluated
            ranked_doc_ids = [doc_ids[idx] for idx in ranked_indices[:MAX_EVAL_K]]
            per_query_rows.append(compute_metrics(ranked_doc_ids, qrels[query_id]))
        total_time = time.perf_counter() - start_total

        metrics = aggregate_metrics(per_query_rows)
        rows.append(
            {
                "label": "BM25 wand",
                "query_count": len(query_ids),
                "latency_ms": 1000.0 * (total_time / len(query_ids)),
                "p50_ms": 1000.0 * statistics.median(times),
                "p95_ms": 1000.0 * sorted(times)[max(0, int(0.95 * len(times)) - 1)],
                "metrics": metrics,
            }
        )
        output_lines.append(f"wand_fully_evaluated_total={total_fully_evaluated}")
        output_lines.append(
            f"wand_fully_evaluated_avg={total_fully_evaluated / len(query_ids):.2f}"
        )

    output_lines.append("model\tlatency_ms\tp50_ms\tp95_ms\tR@1\tR@5\tR@10\tR@20\tMRR@10\tnDCG@10")
    for row in rows:
        metrics = row["metrics"]
        output_lines.append(
            f"{row['label']}\t{row['latency_ms']:.2f}\t{row['p50_ms']:.2f}\t{row['p95_ms']:.2f}\t"
            f"{metrics[1][0]:.4f}\t{metrics[5][0]:.4f}\t{metrics[10][0]:.4f}\t{metrics[20][0]:.4f}\t"
            f"{metrics[10][1]:.4f}\t{metrics[10][2]:.4f}"
        )

    for line in output_lines:
        print(line)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
# uv run python -m bm_25_wand.evaluation.evaluate --corpus-tokenized data/zalo_legal/tokens.jsonl --qrels data/zalo_legal/qrels/train.jsonl --queries-raw data/zalo_legal/queries.jsonl --mode full --output outputs/bm25_full.txt
# uv run python -m bm_25_wand.evaluation.evaluate --corpus-tokenized data/zalo_legal/tokens.jsonl --qrels data/zalo_legal/qrels/train.jsonl --queries-raw data/zalo_legal/queries.jsonl --mode wand --output outputs/bm25_wand.txt
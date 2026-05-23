from pathlib import Path
import os
import json
import argparse
from typing import Iterable

from phong import InvertedRetriever
from .evaluate import (
    load_qrels,
    precision_at_k,
    recall_at_k,
    average_precision,
    reciprocal_rank,
    hit_rate,
)
from tqdm import tqdm
INDEX_DIR = Path("D:/PTIT/BTL/NLP/phong/inverted_index/inverted")
CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "data_zalo" / "corpus_tokenized.jsonl"
KS = [1, 5, 10, 20]


def evaluate_inverted(retriever: InvertedRetriever, qrels_files: Iterable[str], queries_path: str, predict_out: str, tf_scheme: str = "log"):
    qlist = load_qrels(list(qrels_files), queries_path)
    iterator = tqdm(qlist, desc="Evaluating queries")

    metrics = {k: {"precision": 0.0, "recall": 0.0, "ap": 0.0, "hit": 0.0} for k in KS}
    mrr_sum = 0.0
    nq = 0
    import time
    latencies_ms = []
    warmup = 5

    os.makedirs(Path(predict_out).parent, exist_ok=True)

    with open(predict_out, "w", encoding="utf-8") as pf:
        for q in iterator:
            qid = q.get("qid")
            qtext = q.get("query", "")
            relevant = set(q.get("relevant", []))

            max_k = max(KS)
            t0 = time.perf_counter()
            results = retriever.search(qtext, top_k=max_k, tf_scheme=tf_scheme)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            retrieved_ids = []
            scores = []

            for score, doc_id in results:
                if doc_id is None:
                    continue
                retrieved_ids.append(doc_id)
                scores.append(score)

            pf.write(json.dumps({"qid": qid, "query": qtext, "predicted": retrieved_ids, "scores": scores}, ensure_ascii=False) + "\n")

            # Evaluate metrics
            for k in KS:
                prec = precision_at_k(relevant, retrieved_ids, k)
                rec = recall_at_k(relevant, retrieved_ids, k)
                ap = average_precision(relevant, retrieved_ids, k)
                h = hit_rate(relevant, retrieved_ids, k)

                metrics[k]["precision"] += prec
                metrics[k]["recall"] += rec
                metrics[k]["ap"] += ap
                metrics[k]["hit"] += h

            mrr_sum += reciprocal_rank(relevant, retrieved_ids)
            nq += 1

    # aggregate
    agg = {}
    if nq == 0:
        return {}

    for k in KS:
        agg[k] = {
            "precision": metrics[k]["precision"] / nq,
            "recall": metrics[k]["recall"] / nq,
            "map": metrics[k]["ap"] / nq,
            "hit_rate": metrics[k]["hit"] / nq,
        }

    agg["mrr"] = mrr_sum / nq

    vals = latencies_ms[warmup:] if len(latencies_ms) > warmup else latencies_ms
    vals.sort()
    def _pct(sorted_vals, p):
        if not sorted_vals:
            return None
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == k:
            return sorted_vals[f]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    agg["latency_ms"] = {"p50": _pct(vals, 50), "p90": _pct(vals, 90), "mean": (sum(vals) / len(vals) if vals else None), "count": len(vals)}

    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--corpus", default=str(CORPUS_DIR))
    parser.add_argument("--qrels", nargs="+", default=["data/data_zalo/qrels/test.jsonl", "data/data_zalo/qrels/train.jsonl"])
    parser.add_argument("--queries", default="data/data_zalo/queries.jsonl")
    parser.add_argument("--predict-out", default="phong/evaluate/results/predict_inverted.jsonl")
    parser.add_argument("--metrics-out", default="phong/evaluate/results/metrics_inverted.json")
    parser.add_argument("--tf-scheme", choices=["log", "augmented"], default="log")


    args = parser.parse_args()

    retriever = InvertedRetriever()

    try:
        retriever.load(args.index_dir)
    except Exception:
        retriever.build(args.corpus, tf_scheme=args.tf_scheme)
        try:
            retriever.save(args.index_dir)
        except Exception:
            pass

    agg = evaluate_inverted(retriever, args.qrels, args.queries, args.predict_out, tf_scheme=args.tf_scheme)

    print("Evaluation summary:")
    print(json.dumps(agg, ensure_ascii=False, indent=2))

    metrics_path = Path(args.metrics_out)
    os.makedirs(metrics_path.parent, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(agg, mf, ensure_ascii=False, indent=2)
    print(f"Wrote metrics to: {metrics_path}")


if __name__ == '__main__':
    main()

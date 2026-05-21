from pathlib import Path
import os
import json
import argparse
from typing import Iterable

from .evaluate import (
    load_qrels,
    precision_at_k,
    recall_at_k,
    average_precision,
    reciprocal_rank,
    hit_rate,
)
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.remote_client import RemoteAPIClient
from src.indexing.embedding.remote_embedding import RemoteEmbeddingModel
from src.indexing.vector_store.schemas import ChromaConfig
from src.indexing.vector_store.chroma_store import ChromaStore
from src.search.search import SearchService

KS = [1,5,10,20]


def evaluate_dense(chroma_config: ChromaConfig, qrels_files: Iterable[str], queries_path: str, predict_out: str, tf_scheme: str = "remote"):
    qlist = load_qrels(list(qrels_files), queries_path)
    iterator = tqdm(qlist, desc="Evaluating queries")

    metrics = {k: {"precision": 0.0, "recall": 0.0, "ap": 0.0, "hit": 0.0} for k in KS}
    mrr_sum = 0.0
    nq = 0
    import time
    latencies_ms = []
    warmup = 5

    os.makedirs(Path(predict_out).parent, exist_ok=True)

    api_client = RemoteAPIClient()
    embed_model = RemoteEmbeddingModel(api_client)
    store = ChromaStore(chroma_config)
    service = SearchService(chroma_store=store, embedding_model=embed_model, reranker=None, collection_name=chroma_config.collection_name)

    with open(predict_out, "w", encoding="utf-8") as pf:
        for q in iterator:
            qid = q.get('qid')
            qtext = q.get('query', "")
            relevant = set(q.get('relevant', []))

            max_k = max(KS)
            t0 = time.perf_counter()
            results = service.search(query=qtext, top_k_retrieve=max_k, use_rerank=False)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            retrieved_ids = [r.chunk_id for r in results if r.chunk_id]
            scores = [r.distance for r in results]

            pf.write(json.dumps({"qid": qid, "query": qtext, "predicted": retrieved_ids, "scores": scores}, ensure_ascii=False) + "\n")

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
    parser.add_argument('--collection', default='legal_zalo')
    parser.add_argument('--persist-dir', default=str(Path('chroma_db').resolve()))
    parser.add_argument('--qrels', nargs='+', default=['data/data_zalo/qrels/test.jsonl', 'data/data_zalo/qrels/train.jsonl'])
    parser.add_argument('--queries', default='data/data_zalo/queries.jsonl')
    parser.add_argument('--predict-out', default='phong/evaluate/results/predict_dense.jsonl')
    parser.add_argument('--metrics-out', default='phong/evaluate/results/metrics_dense.json')
    parser.add_argument('--tf-scheme', default='remote')
    args = parser.parse_args()

    chroma_cfg = ChromaConfig(collection_name=args.collection, persist_directory=args.persist_dir, is_persist=True)

    agg = evaluate_dense(chroma_cfg, args.qrels, args.queries, args.predict_out, tf_scheme=args.tf_scheme)

    print("Evaluation summary:")
    print(json.dumps(agg, ensure_ascii=False, indent=2))

    metrics_path = Path(args.metrics_out)
    os.makedirs(metrics_path.parent, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        json.dump(agg, mf, ensure_ascii=False, indent=2)
    print(f"Wrote metrics to: {metrics_path}")


if __name__ == '__main__':
    main()

import json
import math
import heapq
from pathlib import Path
from collections import Counter, defaultdict
from ..utils import tfidf_query_vector, tokenize
from typing import Dict, List, Tuple

def build_index(corpus_path: str, tf_scheme: str = "log", **kwargs):
    postings = defaultdict(list)
    df = Counter()
    doc_counts: Dict[str, Counter] = {}

    docs_meta = {}

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            
            doc_id = str(obj.get('_id'))
            
            docs_meta[doc_id] = {k: obj.get(k) for k in ('title', 'text')}
            
            text = f"{obj.get('title','')} {obj.get('text','')}"
            tokens = tokenize(text).split()

            c = Counter(tokens)
           
            doc_counts[doc_id] = c
            
            for t in c.keys():
                df[t] += 1

    N = len(doc_counts)

    if N == 0:
        return {}, {}, {}, {}

    idf = {t: math.log((1 + N) / (1 + df[t])) + 1.0 for t in df}

    doc_norm = {}

    for doc_id, c in doc_counts.items():
        max_tf = max(c.values()) if c else 1
        norm2 = 0.0
        for t, count in c.items():
            if tf_scheme == 'augmented':
                tf_w = 0.5 + 0.5 * (count / max_tf)
            else:
                tf_w = 1.0 + math.log(count)

            doc_tf_idf = tf_w * idf.get(t, 0.0)

            postings[t].append([doc_id, doc_tf_idf])

            norm2 += doc_tf_idf * doc_tf_idf

        doc_norm[doc_id] = math.sqrt(norm2)

    return dict(postings), idf, doc_norm, docs_meta


def save_index(index_dir: str, postings: Dict[str, List[Tuple[str, float]]], idf: Dict[str, float], doc_norm: Dict[str, float], docs_meta: Dict[str, dict]):
    p = Path(index_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / 'postings.json', 'w', encoding='utf-8') as f:
        json.dump(postings, f, ensure_ascii=False)
    with open(p / 'idf.json', 'w', encoding='utf-8') as f:
        json.dump(idf, f, ensure_ascii=False)
    with open(p / 'doc_norm.json', 'w', encoding='utf-8') as f:
        json.dump(doc_norm, f, ensure_ascii=False)
    with open(p / 'docs_meta.json', 'w', encoding='utf-8') as f:
        json.dump(docs_meta, f, ensure_ascii=False)


def load_index(index_dir: str):
    p = Path(index_dir)
    with open(p / 'postings.json', 'r', encoding='utf-8') as f:
        postings = json.load(f)
    with open(p / 'idf.json', 'r', encoding='utf-8') as f:
        idf = json.load(f)
    with open(p / 'doc_norm.json', 'r', encoding='utf-8') as f:
        doc_norm = json.load(f)
    with open(p / 'docs_meta.json', 'r', encoding='utf-8') as f:
        docs_meta = json.load(f)
    return postings, idf, doc_norm, docs_meta


def search(query: str, postings: Dict[str, List[Tuple[str, float]]], idf: Dict[str, float], doc_norm: Dict[str, float], top_k: int = 10, tf_scheme: str = 'log', **kwargs):
    query_vec = tfidf_query_vector(query, idf, tf_scheme=tf_scheme)
    if not query_vec:
        return []

    acc = {}
    for t, q_w in query_vec.items():
        for doc_id, doc_tf_idf in postings.get(t, []):
            acc[doc_id] = acc.get(doc_id, 0.0) + (doc_tf_idf * q_w)

    scored_iter = ((val / max(doc_norm.get(doc_id, 1.0), 1e-12), doc_id) for doc_id, val in acc.items())
    top = heapq.nlargest(top_k, scored_iter, key=lambda x: x[0])
    return top


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='data/data_zalo/corpus_tokenized.jsonl')
    parser.add_argument('--index-dir', default='phong/inverted_index/inverted')
    parser.add_argument('--tf-scheme', choices=['log','augmented'], default='log')
    args = parser.parse_args()

    postings, idf, doc_norm, docs_meta = build_index(args.corpus, tf_scheme=args.tf_scheme)
    save_index(args.index_dir, postings, idf, doc_norm, docs_meta)
    print('Built inverted index; terms:', len(postings))
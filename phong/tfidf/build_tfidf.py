import heapq
import json
import math
from pathlib import Path
from typing import Dict
from collections import Counter
from ..utils import tfidf_query_vector, cosine_similarity, tokenize

def build_corpus_index(corpus_tokenized_path: str, tf_scheme: str = "log", **kwargs):
    docs = []
    doc_counts = []
    df = Counter()

    with open(corpus_tokenized_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            docs.append(doc)

            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = tokenize(text).split()

            # Tính tần suất từ trong doc
            c = Counter(tokens)

            doc_counts.append(c)
            
            # Đếm số doc chứa từ
            for t in c.keys():
                df[t] += 1
    
    N = len(doc_counts)

    if N == 0:
        return docs, [], {}
    
    # Tính IDF
    idf = {
        t: math.log((1 + N) / (1 + df[t])) + 1.0 for t in df
    }

    doc_vectors = []

    for c in doc_counts:
        vec = {}
        norm2 = 0.0
        max_tf = max(c.values()) if c else 1
        for t, count in c.items():
            if count <= 0:
                continue

            # TF weight: log or augmented
            if tf_scheme == "augmented":
                tf_w = 0.5 + 0.5 * (count / max_tf)
            else:
                tf_w = 1.0 + math.log(count)

            w = tf_w * idf.get(t, 0.0)

            vec[t] = w

            norm2 += w * w

        norm = math.sqrt(norm2)

        for t in vec:
            if norm > 0:
                vec[t] /= norm

        doc_vectors.append(vec)

    return docs, doc_vectors, idf


def save_index(index_dir: str, doc_vectors, idf):
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vecs_path = Path(index_dir) / "doc_vectors.jsonl"
    idf_path = Path(index_dir) / "idf.json"

    with open(vecs_path, "w", encoding="utf-8") as f:
        for v in doc_vectors:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    with open(idf_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(idf, ensure_ascii=False))


def load_index(index_dir: str):
    vecs_path = Path(index_dir) / "doc_vectors.jsonl"
    idf_path = Path(index_dir) / "idf.json"

    if not (vecs_path.exists() and idf_path.exists()):
        raise FileNotFoundError(f"Index files missing in {index_dir}")

    docs_path = Path(__file__).parent.parent.parent / "data" / "data_zalo" / "corpus_tokenized.jsonl"

    docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))

    doc_vectors = []
    with open(vecs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc_vectors.append(json.loads(line))

    with open(idf_path, "r", encoding="utf-8") as f:
        idf = json.load(f)

    return docs, doc_vectors, idf


def search(query: str, docs: list[Dict], doc_vectors: list[Dict[str, float]], idf: Dict[str, float], top_k: int = 5, tf_scheme: str = "log", **kwargs):
    query_vec = tfidf_query_vector(query, idf, tf_scheme=tf_scheme)

    if not query_vec:
        return []

    scored_docs = ((doc, cosine_similarity(query_vec, doc_vec)) for doc, doc_vec in zip(docs, doc_vectors))

    return heapq.nlargest(top_k, scored_docs, key=lambda x: x[1])
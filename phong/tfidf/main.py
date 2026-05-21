from pathlib import Path
import os
from phong import build_corpus_index, search, save_index, load_index

INDEX_DIR = Path("D:/PTIT/BTL/NLP/phong/tfidf/tfidf_index")

CORPUS_DIR = Path(__file__).parent.parent.parent / "data" / "data_zalo" / "corpus_tokenized.jsonl"
TOP_K = 5
docs = doc_vecs = idf = None

def search_query(query: str):
    docs = doc_vecs = idf = None
    try:
        docs, doc_vecs, idf = load_index(INDEX_DIR)
        print(f"Loaded index from: {INDEX_DIR} ({len(docs)} docs)")
    except Exception as exc:
        print(f"Index not usable at {INDEX_DIR}, rebuilding: {exc}")

    if docs is None:
        print(f"Building index from: {CORPUS_DIR}")
        docs, doc_vecs, idf = build_corpus_index(CORPUS_DIR)
        print(f"Documents loaded: {len(docs)}")
        save_index(INDEX_DIR, doc_vecs, idf)
        print(f"Saved index to: {INDEX_DIR}")

    results = search(query, docs, doc_vecs, idf, top_k=TOP_K)
    # print(results)
    print("Top results:")
    for doc, score in results:
        title = doc.get('title') or (doc.get('text') or '')[:120]
        print(f"{score:.4f}\t{title}")

search_query("nhượng quyền")
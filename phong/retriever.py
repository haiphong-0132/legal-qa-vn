class Retriever:
    def build(self, corpus_path: str, **kwargs) -> None:
        raise NotImplementedError("Must implement build method")

    def save(self, index_dir: str) -> None:
        raise NotImplementedError("Must implement save method")

    def load(self, index_dir: str) -> None:
        raise NotImplementedError("Must implement load method")
    
    def search(self, query: str, top_k: int = 5) -> list:
        raise NotImplementedError("Must implement search method")
    
class TfidfRetriever(Retriever):
    def __init__(self):
        self.docs = None
        self.doc_vectors = None
        self.idf = None

    def build(self, corpus_path: str, **kwargs):
        from .tfidf.build_tfidf import build_corpus_index
        docs, doc_vectors, idf = build_corpus_index(corpus_path, **kwargs)
        self.docs = docs
        self.doc_vectors = doc_vectors
        self.idf = idf

    def save(self, index_dir: str):
        from .tfidf.build_tfidf import save_index
        if self.doc_vectors is None or self.idf is None:
            raise RuntimeError("Index not built")
        save_index(index_dir, self.doc_vectors, self.idf)

    def load(self, index_dir: str):
        from .tfidf.build_tfidf import load_index
        docs, doc_vectors, idf = load_index(index_dir)
        self.docs = docs
        self.doc_vectors = doc_vectors
        self.idf = idf

    def search(self, query: str, top_k: int = 5, **kwargs) -> list:
        from .tfidf.build_tfidf import search
        if self.docs is None or self.doc_vectors is None or self.idf is None:
            raise RuntimeError("Index not loaded or built")
        return search(query, self.docs, self.doc_vectors, self.idf, top_k=top_k, **kwargs)


class InvertedRetriever(Retriever):
    def __init__(self):
        self.postings = None
        self.idf = None
        self.doc_norm = None
        self.docs_meta = None

    def build(self, corpus_path: str, **kwargs):
        from .inverted_index.inverted_index import build_index
        postings, idf, doc_norm, docs_meta = build_index(corpus_path, **kwargs)
        self.postings = postings
        self.idf = idf
        self.doc_norm = doc_norm
        self.docs_meta = docs_meta

    def save(self, index_dir: str):
        from .inverted_index.inverted_index import save_index
        if self.postings is None or self.idf is None or self.doc_norm is None:
            raise RuntimeError("Index not built")
        save_index(index_dir, self.postings, self.idf, self.doc_norm, self.docs_meta or {})

    def load(self, index_dir: str):
        from .inverted_index.inverted_index import load_index
        postings, idf, doc_norm, docs_meta = load_index(index_dir)
        self.postings = postings
        self.idf = idf
        self.doc_norm = doc_norm
        self.docs_meta = docs_meta

    def search(self, query: str, top_k: int = 5, **kwargs) -> list:
        from .inverted_index.inverted_index import search as inverted_search
        if self.postings is None or self.idf is None or self.doc_norm is None:
            raise RuntimeError("Index not loaded or built")
        return inverted_search(query, self.postings, self.idf, self.doc_norm, top_k=top_k, **kwargs)

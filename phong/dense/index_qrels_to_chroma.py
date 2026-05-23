import os
import json
import argparse
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from src.api.remote_client import RemoteAPIClient
from src.indexing.embedding.remote_embedding import RemoteEmbeddingModel
from src.indexing.embedding.utils import create_embedding_request
from src.indexing.vector_store.schemas import ChromaConfig
from src.indexing.vector_store.chroma_store import ChromaStore
from src.indexing.vector_store.vectorstore import VectorStorePipeline


def load_qrel_doc_ids(qrels_files: List[str]) -> set:
    ids = set()
    for qf in qrels_files:
        with open(qf, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get('corpus-id') or obj.get('corpus_id')
                if cid is None:
                    cid = obj.get('corpusId')
                if cid is not None:
                    ids.add(str(cid))
    return ids


def index_docs(corpus_path: str, doc_ids: set, collection: str, persist_dir: str, batch_size: int = 64):
    api_client = RemoteAPIClient()
    embed_model = RemoteEmbeddingModel(api_client)

    chroma_cfg = ChromaConfig(collection_name=collection, persist_directory=persist_dir, is_persist=True)
    store = ChromaStore(chroma_cfg)

    requests = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get('_id') or obj.get('id') or obj.get('doc_id'))
            if doc_id not in doc_ids:
                continue
            title = obj.get('title') or ''
            text = obj.get('text') or obj.get('content') or ''
            full = (title + '\n' + text).strip()
            metadata = {k: v for k, v in (('title', title),) if v}
            req = create_embedding_request(text=full, chunk_id=doc_id, metadata=metadata)
            requests.append(req)

    if not requests:
        print("No matching documents found in corpus for given qrels")
        return

    print(f"Embedding and upserting {len(requests)} docs into collection={collection}")
    all_results = []
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        results = embed_model.embed(batch, batch_size=len(batch))
        all_results.extend(results)
        pipeline = VectorStorePipeline(embeddings=results)
        pipeline.run(store, batch_size=0)

    print(f"Indexed {len(all_results)} vectors into Chroma collection '{collection}' (persist={persist_dir})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qrels', nargs='+', required=True)
    parser.add_argument('--corpus', default='data/data_zalo/corpus.jsonl')
    parser.add_argument('--collection', default='legal_zalo')
    parser.add_argument('--persist-dir', default=str(Path('chroma_db').resolve()))
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    doc_ids = load_qrel_doc_ids(args.qrels)
    print(f"Found {len(doc_ids)} unique doc ids in qrels")
    index_docs(args.corpus, doc_ids, args.collection, args.persist_dir, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

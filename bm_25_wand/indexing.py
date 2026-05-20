from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> Iterable[dict]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_tokenized_corpus(path: str | Path, k1: float = 1.5, b: float = 0.75):
    doc_ids: list[str] = []
    doc_lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for doc_idx, item in enumerate(read_jsonl(path)):
        doc_id = item.get("doc_id", item.get("id", item.get("_id")))
        if doc_id is None:
            raise KeyError("Tokenized legal corpus records must contain 'doc_id', 'id', or '_id'.")

        tokens = item.get("tokens")
        if not isinstance(tokens, list):
            raise TypeError("Tokenized legal corpus records must contain a list field named 'tokens'.")

        token_list = [str(tok) for tok in tokens if str(tok).strip()]
        doc_ids.append(str(doc_id))
        doc_lengths.append(len(token_list))

        term_freqs = Counter(token_list)
        for term, tf in term_freqs.items():
            postings[term].append((doc_idx, tf))

    if not doc_ids:
        raise ValueError("Tokenized legal corpus is empty.")

    avgdl = float(sum(doc_lengths) / len(doc_lengths))
    total_docs = len(doc_ids)
    raw_inverted_index = dict(postings)
    idf = {
        term: math.log(1.0 + (total_docs - len(posting_list) + 0.5) / (len(posting_list) + 0.5))
        for term, posting_list in raw_inverted_index.items()
    }
    scored_index: dict[str, list[tuple[int, float]]] = {}
    term_max_score: dict[str, float] = {}
    for term, posting_list in raw_inverted_index.items():
        idf_term = idf.get(term, 0.0)
        scored_postings: list[tuple[int, float]] = []
        max_score = 0.0
        for doc_idx, tf in posting_list:
            dl = doc_lengths[doc_idx]
            denom_norm = k1 * (1.0 - b + b * (dl / avgdl)) if avgdl > 0 else k1
            numer = tf * (k1 + 1.0)
            denom = tf + denom_norm
            score = idf_term * (numer / denom)
            scored_postings.append((doc_idx, score))
            if score > max_score:
                max_score = score

        scored_index[term] = scored_postings
        term_max_score[term] = max_score

    return doc_ids, doc_lengths, avgdl, scored_index, idf, term_max_score


def build_champion_index(
    inverted_index: dict[str, list[tuple[int, float]]],
    champion_size: int,
) -> dict[str, list[tuple[int, float]]]:
    champion_index: dict[str, list[tuple[int, float]]] = {}
    for term, posting_list in inverted_index.items():
        scored_postings = sorted(posting_list, key=lambda item: item[1], reverse=True)
        champion_index[term] = scored_postings[:champion_size]

    return champion_index
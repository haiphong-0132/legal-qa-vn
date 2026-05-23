import json

def precision_at_k(relevant_set: set, retrieved: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    
    topk = retrieved[:k]

    return sum(1 for d in topk if d in relevant_set) / k

def recall_at_k(relevant_set: set, retrieved: list[str], k: int) -> float:
    if len(relevant_set) == 0:
        return 0.0
    
    topk = retrieved[:k]
    
    return sum(1 for d in topk if d in relevant_set) / len(relevant_set)

def average_precision(relevant_set: set, retrieved: list[str], k: int = None) -> float:
    hit_count = 0
    score_sum = 0.0
    if k is None:
        k = len(retrieved)
    
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant_set:
            hit_count += 1
            score_sum += hit_count / i
    
    if hit_count == 0:
        return 0.0
    
    return score_sum / len(relevant_set)

def reciprocal_rank(relevant_set: set, retrieved: list[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant_set:
            return 1.0 / i
    return 0.0

def hit_rate(relevant_set: set, retrieved: list[str], k: int) -> float:
    topk = retrieved[:k]
    return 1.0 if any(d in relevant_set for d in topk) else 0.0

def load_qrels(qrels_path: list[str], queries_path: str) -> list[dict]:
    query_to_relevants = {}
    for qrels_file in qrels_path:
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                qid = str(data.get('query-id'))
                docid = str(data.get('corpus-id'))

                query_to_relevants.setdefault(qid, set()).add(docid)
    
    query_to_texts = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            qid = str(data.get('_id'))
            text = str(data.get('text'))
            query_to_texts[qid] = text

    qlist = []
    for qid, rels in query_to_relevants.items():
        qlist.append({'qid': qid, 'query': query_to_texts.get(qid, ''), 'relevant': rels})
    
    return qlist
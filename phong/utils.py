import re
import math
from collections import Counter
from typing import Dict
from underthesea import word_tokenize

def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    code_re = re.compile(r"\b[\wÀ-ỹ]+(?:[\/\-][\wÀ-ỹ]+)+\b", flags=re.UNICODE)
    codes = []

    def _mask(m):
        idx = len(codes)
        codes.append(m.group(0))
        return f"__CODE{idx}__"

    masked = code_re.sub(_mask, text)

    cleaned = re.sub(r"[^\w\s]", " ", masked, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    for i, orig in enumerate(codes):
        norm = re.sub(r"[\/\-]+", "_", orig)
        cleaned = cleaned.replace(f"__CODE{i}__", norm)

    return cleaned

def tokenize(text: str) -> str:
    text = normalize(text)
    text = word_tokenize(text, format="text")
    return text

def tfidf_query_vector(query: str, idf: Dict[str, float], tf_scheme: str = "log", **kwargs) -> Dict[str, float]:
    tokens = tokenize(query).split()
    tf = Counter(tokens)

    vec = {}
    norm2 = 0.0

    max_tf = max(tf.values()) if tf else 1
    for t, count in tf.items():
        if count <= 0:
            continue

        if tf_scheme == "augmented":
            tf_w = 0.5 + 0.5 * (count / max_tf)
        else:
            tf_w = 1.0 + math.log(count)

        w = tf_w * idf.get(t, 0.0)

        vec[t] = w
        norm2 += w * w

    norm = math.sqrt(norm2)

    if norm > 0:
        for t in list(vec.keys()):
            vec[t] /= norm

    return vec


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    if len(a) > len(b):
        a, b = b, a

    dot = sum(a.get(t, 0.0) * b.get(t, 0.0) for t in a)

    return dot

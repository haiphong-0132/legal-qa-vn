from __future__ import annotations

import re
from functools import lru_cache

_NON_ALNUM_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = text.replace("_", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split()


def prepare_phobert_text(text: str) -> str:
    text = text.lower().strip()
    text = _WHITESPACE_RE.sub(" ", text)

    try:
        from underthesea.pipeline.word_tokenize import word_tokenize

        text = word_tokenize(text, format="text")
    except Exception:
        pass

    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


VIETNAMESE_STOPWORDS = {
    "bị", "bởi", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", 
    "chưa", "chuyện", "có", "có_thể", "cứ", "của", "cùng", "cũng", "đã", "đang", 
    "đây", "để", "đến", "đều", "điều", "do", "đó", "được", "dưới", "gì", "khi",
    "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "một_cách", "nay", "này", 
    "nên", "nếu", "ngay", "nhiều", "như", "nhưng", "những", "nơi", "nữa", "phải", 
    "qua", "ra", "rằng", "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", 
    "thì", "trên", "trong", "trước", "từ", "từng", "và", "vẫn", "vào", "vậy", 
    "vì", "việc", "với", "vừa", "hoặc"
}

@lru_cache(maxsize=8192)
def tokenize_underthesea_text(text: str) -> tuple[str, ...]:
    text = text.lower()
    # Lọc bỏ dấu câu bằng regex _NON_ALNUM_RE có sẵn
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    try:
        from underthesea.pipeline.word_tokenize import word_tokenize

        tokenized = word_tokenize(text, format="text")
    except Exception:
        tokenized = text

    tokenized = _WHITESPACE_RE.sub(" ", tokenized).strip()
    if not tokenized:
        return ()

    # Loại bỏ stopword
    tokens = [tok for tok in tokenized.split() if tok and tok not in VIETNAMESE_STOPWORDS]
    return tuple(tokens)


if __name__ == "__main__":
    text = "Đây là một ví dụ về văn bản cần được chuẩn hóa và token hóa."
    print("Original:", text)
    print("Normalized:", normalize_text(text))
    print("Tokens:", tokenize_text(text))
    print("Phobert Text:", prepare_phobert_text(text))
    print("Underthesea Tokens:", tokenize_underthesea_text(text))
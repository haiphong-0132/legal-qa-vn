"""Schemas for embedding module."""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Một đơn vị chunk cần embedding, hoặc truy vấn của người dùng
    
    Attributes:
        chunk_id: Duy nhất, lấy từ DocumentNode.id
        num_chunk: Số thứ tự của chunk trong văn bản, dùng để kiểm tra thứ tự khi trả về kết quả
        text: Nội dung cần embedding
        metadata: Thông tin metadata của chunk, dùng để lưu vào vector database
    """
    chunk_id: str | None = None
    num_chunk: Optional[int] = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
class EmbeddingResult(BaseModel):
    """Kết quả embed 1 chunk
    
    Attributes:
        chunk_id: ID của chunk
        num_chunk: Số thứ tự của chunk
        text: Nội dung được embed
        vector: Vector embedding
        token_count: Số token của chunk để kiểm tra có vượt giới hạn mô hình hay không
        metadata: Metadata của chunk, có thể dùng để lưu vào vector database
    """
    chunk_id: str | None = None
    num_chunk: Optional[int] = None
    text: str
    vector: List[float]
    token_count: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict
from enum import Enum
from datetime import date

class RelationType(str, Enum):
    """
    Quan hệ giữa các văn bản pháp luật, có thể là hướng dẫn thi hành, sửa đổi bổ sung, thay thế, bãi bỏ, tạm thời áp dụng, giải thích, hoặc liên quan.
    """
    huong_dan_thi_thanh = 'huong_dan_thi_thanh' # Nghị định/ Thông tư/ Nghị quyết hướng dẫn
    sua_doi_bo_sung = 'sua_doi_bo_sung' # Sửa một phần văn bản khác
    thay_the = 'thay_the' # Thay thế toàn bộ văn bản cũ
    bai_bo = 'bai_bo' # Bãi bỏ hoàn toàn văn bản cũ
    dinh_chi_hieu_luc = 'dinh_chi_hieu_luc' # Tạm đình chỉ hiệu lực
    tam_thoi_ap_dung = 'tam_thoi_ap_dung' # Thí điểm cơ chế chưa có trong Luật hiện hành
    giai_thich = 'giai_thich' # Quan hệ giải thích chính thức điều khoản luật
    lien_quan = 'lien_quan' # Tham chiếu nhau, không trực tiếp

class DocumentRelation(BaseModel):
    """
    Đại diện cho một quan hệ giữa 2 node văn bản pháp luật.
    """
    id: int=None
    original_doc_id: str
    related_doc_id: str=None
    relation_type: RelationType=None
    description: Optional[str]=None

class Document(BaseModel):
    """
    Đại diện một văn bản pháp luật trong hệ thống
    """
    so_hieu: str
    name: str
    type: str # Nghị định/ Thông tư/ Nghị quyết/ Luật/ Văn bản khác...
    co_quan_ban_hanh: str
    ngay_ban_hanh: date
    ngay_co_hieu_luc: date
    file_path: str
    md_path: Optional[str] = None

    @model_validator(mode='after')
    def validate_dates(self):
        if self.ngay_co_hieu_luc and self.ngay_co_hieu_luc < self.ngay_ban_hanh:
            raise ValueError('Ngày có hiệu lực không thể trước ngày ban hành')
        return self

class HierarchicalChunkInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    payload: Dict[str, Any] | List[Dict[str, Any]] = Field(
        alias="json",
        description="Raw JSON đầu vào dùng cho hierarchical chunker."
    )


class ChunkMetadata(BaseModel):
    """Metadata cho 1 chunk.
    document_id dùng để truy vết văn bản gốc của chunk.
    section_id dung de truy vết vi tri chunk trong cau truc van ban.
    Hien tai voi hierarchical chunker, format thuc te la: "<section.id>"
    vi du: "dieu_6.diem_2".
    """
    document_id: str
    section_id: str


class ChunkDocument(BaseModel):
    text: str
    metadata: ChunkMetadata


#
class ChunkDocumentForHierarchical(BaseModel):
    """Chunk output cho hierarchical gom metadata + tieu de + noi dung + ref."""

    metadata: ChunkMetadata
    tieu_de: Optional[str] = None
    noi_dung: Optional[str] = None
    ref: List[str] = Field(default_factory=list)


class EmbeddingRequest(BaseModel):
    """Một đơn vị chunk cần embedding, hoặc truy vấn của người dùng"""
    chunk_id: str | None = None  # Duy nhất, lấy từ ChunkMetadata.section_id hoặc tự tạo khi khởi tạo
    text: str

class EmbeddingResult(BaseModel):
    """Kết quả embed 1 chunk"""
    chunk_id: str | None = None
    text: str
    vector: List[float]     # Vector embedding
    token_count: Optional[int] = None   # Số token của chunk để kiểm tra có vượt giới hạn mô hình hay không

# Store in vector database

class ChromaConfig(BaseModel):
    collection_name: str
    persist_directory: Optional[str] = None      # Nơi lưu trữ
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"  # Khoảng cách sử dụng trong ChromaDB
    is_persist: bool = False

    @model_validator(mode='after')
    def validate_persistence(self):
        if self.is_persist and not self.persist_directory:
            raise ValueError('persist_directory is required when is_persist=True')
        return self

class ChromaUpsertRequest(BaseModel):
    """Dữ liệu cần upsert vào ChromaDB"""
    chunk_id: str
    vector: List[float]         # Lấy từ EmbeddingResult.vector
    text: str                   # Lấy từ ChunkDocument.text
    metadata: Dict[str, Any]    # Lấy từ ChunkMetadata tương ứng và có thể thêm thông tin khác nếu cần

class ChromaQueryRequest(BaseModel):
    """Yêu cầu truy vấn từ ChromaDB"""
    query_vector: List[float]                   # Embeding của câu truy vấn
    top_k: int = Field(5, gt=0)
    filter: Optional[Dict[str, Any]] = None     # Bộ lọc theo metadata nếu cần

class ChromaQueryResult(BaseModel):
    """Kết quả trả về từ ChromaDB sau khi truy vấn"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
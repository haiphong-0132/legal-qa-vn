import os
import sys
import json
from pathlib import Path
import logging

# Thêm thư mục gốc vào sys.path để import
sys.path.append(str(Path(__file__).resolve().parents[0]))

from src.api.nvidia_api_client import OpenAICompatibleClient
from src.indexing.vector_store import ChromaStore, ChromaConfig
from system.database.db_respository import get_database, DocumentRelationRepository, DocumentMetadataRepository
from src.search.search import SearchService
from src.agent.tools import LegalAgentTools
from src.agent.schemas import SubQuestion, Intent
from src.api.remote_client import RemoteAPIClient
from src.indexing.embedding.remote_embedding import RemoteEmbeddingModel
from src.search.reranker import RemoteReranker
from src.api.nvidia_api_client import OpenAICompatibleClient
from src.agent.nodes import (
    make_analyze_node,
    make_doc_relation_node,
    make_legal_query_node,
    make_general_node
)


COLLECTION_NAME="legal_documents"
CHROMA_DIR="chroma_db"
# Tắt log quá dài của HTTP
logging.getLogger("httpx").setLevel(logging.WARNING)

def setup_environment():
    print("=== KHỞI TẠO MÔI TRƯỜNG ===")
    llm = OpenAICompatibleClient()
    
    # Init Chroma
    chroma_store = ChromaStore(ChromaConfig(
        collection_name=COLLECTION_NAME,
        is_persist=True,
        persist_directory=CHROMA_DIR,
        distance_metric='cosine'
    ))
    # llm=OpenAICompatibleClient()
    # Init DB
    db_manager = get_database()
    session = db_manager.get_session()
    meta_repo = DocumentMetadataRepository(session)
    relation_repo = DocumentRelationRepository(session)
    api_client = RemoteAPIClient()
    embedding_model = RemoteEmbeddingModel(api_client)
    reranker = RemoteReranker(api_client)
    
    search_service = SearchService(
        chroma_store=chroma_store,
        embedding_model=embedding_model,
        reranker=reranker
    )
    
    tools = LegalAgentTools(
        search_service=search_service,
        chroma_store=chroma_store,
        meta_repo=meta_repo,
        relation_repo=relation_repo,
        api_client=api_client
    )
    
    return llm, tools

def test_analyze(llm):
    print("\n" + "="*60)
    print("TEST NODE: ANALYZE (PHÂN RÃ & ĐỊNH TUYẾN INTENT)")
    print("="*60)
    
    analyze_node = make_analyze_node(llm)
    
    state = {
        "question": "Cho tôi thông tin về Luật Đất đai và mức phạt nồng độ cồn xe máy hiện nay?",
        "context_text": [],
        "tool_outputs": [],
        "messages": []
    }
    
    print("❓ QUESTION:", state["question"])
    result = analyze_node(state)
    
    print("\n🎯 KẾT QUẢ PHÂN RÃ:")
    for sq in result.get("sub_questions", []):
        print(f" - [Intent: {sq.intent.value.upper()}] Query: {sq.query}")

def test_doc_relation(llm, tools):
    print("\n" + "="*60)
    print("TEST NODE: DOC_RELATION (QUAN HỆ VÀ THÔNG TIN VĂN BẢN)")
    print("="*60)
    
    doc_relation_node = make_doc_relation_node(tools, llm)
    
    state = {
        "question": "Cho tôi thông tin cơ bản về BỘ LUẬT HÌNH SỰ",
        "current_sub_question": SubQuestion(query="Cho tôi thông tin cơ bản về BỘ LUẬT HÌNH SỰ", intent=Intent.DOC_RELATION),
        "context_text": [],
        "tool_outputs": [],
        "messages": []
    }
    
    print("❓ SUB-QUESTION:", state["current_sub_question"].query)
    result = doc_relation_node(state)
    
    print("\n📑 CONTEXT TEXT EXTRACTED:")
    if result.get("context_text"):
        print(result["context_text"][0])
    else:
        print("Không có context_text")

def test_legal_query(llm, tools):
    print("\n" + "="*60)
    print("TEST NODE: LEGAL_QUERY (TÌM KIẾM THEO ĐIỀU KHOẢN CHÍNH XÁC)")
    print("="*60)
    
    legal_query_node = make_legal_query_node(tools, llm)
    
    state = {
        "question": "Khoản 1 Điều 16 nghị định 15/2025/NĐ-CP",
        "current_sub_question": SubQuestion(query="Khoản 1 Điều 16 nghị định 15/2025/NĐ-CP", intent=Intent.LEGAL_QUERY),
        "context_text": [],
        "tool_outputs": [],
        "messages": []
    }
    
    print("❓ SUB-QUESTION:", state["current_sub_question"].query)
    result = legal_query_node(state)
    
    print("\n📑 CONTEXT TEXT EXTRACTED:")
    if result.get("context_text"):
        print(result["context_text"][0])
    else:
        print("Không có context_text")

def test_general(llm, tools):
    print("\n" + "="*60)
    print("TEST NODE: GENERAL (TÌM KIẾM SEMANTIC VECTOR)")
    print("="*60)
    
    general_node = make_general_node(tools, llm)
    
    state = {
        "question": "Đơn vị kinh doanh vận tải sử dụng hợp đồng vận tải bằng hợp đồng điện tử phải đáp ứng điều kiện gì?",
        "current_sub_question": SubQuestion(query="Đơn vị kinh doanh vận tải sử dụng hợp đồng vận tải bằng hợp đồng điện tử phải đáp ứng điều kiện gì?", intent=Intent.GENERAL),
        "context_text": [],
        "tool_outputs": [],
        "messages": []
    }
    
    print("❓ SUB-QUESTION:", state["current_sub_question"].query)
    result = general_node(state)
    
    print("\n📑 CONTEXT TEXT EXTRACTED (Hiển thị 500 ký tự đầu):")
    if result.get("context_text"):
        print(result["context_text"][0][:500] + "\n...[CÒN TIẾP]")
    else:
        print("Không có context_text")

if __name__ == "__main__":
    llm, tools = setup_environment()
    
    # Bạn có thể comment/uncomment để test riêng từng node
    test_analyze(llm)
    test_doc_relation(llm, tools)
    test_legal_query(llm, tools)
    test_general(llm, tools)

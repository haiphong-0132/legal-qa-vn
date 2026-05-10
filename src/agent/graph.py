from __future__ import annotations
from langgraph.graph import START, END, StateGraph
from src.agent.schema import Intent
from src.agent.state import AgentState
from src.agent.nodes import(
    make_analyze_node,
    make_fallback_node,
    make_doc_retrieve_node,
    make_legal_query_node,
    make_general_node,
    make_doc_relation_node,
    make_evaluate_chunks_node,
    make_generate_response_node,
)
import logging
from typing import Any

logger = logging.getLogger(__name__)

def route_intent(state: AgentState) -> str:
    """
    Hàm định tuyến (Router) dựa trên intent từ node analyze.
    """
    intent = state.get("intent")
    
    # Chuyển đổi intent sang tên node tương ứng.
    # Cần đảm bảo giá trị của state["intent"] khớp với các key này.
    if intent == Intent.DOC_RETRIEVE:
        return "doc_retrieve"
    elif intent == Intent.LEGAL_QUERY:
        return "legal_query"
    elif intent == Intent.GENERAL:
        return "general"
    elif intent == Intent.DOC_RELATION:
        return "doc_relation"
    else:
        return "fallback"

def build_graph(
    llm: Any,
    db_client: Any,
    retriever: Any,
    doc_retriever: Any
) -> StateGraph:

    workflow = StateGraph(AgentState)
    
    # 1. Thêm các node với tham số đã được căn chỉnh khớp với definitions trong nodes.py
    workflow.add_node("analyze", make_analyze_node(llm))
    workflow.add_node("fallback", make_fallback_node(llm))
    workflow.add_node("doc_retrieve", make_doc_retrieve_node(retriever))
    workflow.add_node("legal_query", make_legal_query_node(db_client, llm))
    workflow.add_node("general", make_general_node(retriever, llm))
    workflow.add_node("doc_relation", make_doc_relation_node(doc_retriever)) # dùng doc_retriever thay vì db_client
    workflow.add_node("evaluate_chunks", make_evaluate_chunks_node(llm))
    workflow.add_node("generate_response", make_generate_response_node(llm))

    # 2. Luồng thực thi: Bắt đầu -> Analyze
    workflow.add_edge(START, "analyze")

    # 3. Phân nhánh: Analyze -> 5 nhánh dựa vào hàm route_intent
    workflow.add_conditional_edges(
        "analyze",
        route_intent,
        {
            "doc_retrieve": "doc_retrieve",
            "legal_query": "legal_query",
            "general": "general",
            "doc_relation": "doc_relation",
            "fallback": "fallback",
        }
    )

    # 4. Gom các nhánh (trừ fallback) về node evaluate_chunks
    workflow.add_edge("doc_retrieve", "evaluate_chunks")
    workflow.add_edge("legal_query", "evaluate_chunks")
    workflow.add_edge("general", "evaluate_chunks")
    workflow.add_edge("doc_relation", "evaluate_chunks")
    
    # Nhánh fallback (nếu không tìm thấy chunks) có thể đi thẳng tới generate_response 
    workflow.add_edge("fallback", "generate_response")

    # 5. Đánh giá chunks -> Sinh câu trả lời -> END
    workflow.add_edge("evaluate_chunks", "generate_response")
    workflow.add_edge("generate_response", END)

    compiled = workflow.compile()
    return compiled
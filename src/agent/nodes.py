import json
import re
from typing import Dict, Any, Callable
from pydantic import ValidationError
from .state import AgentState
from .schemas import AnalyzerOutput, SubQuestion, Intent, LegalCitation
from .prompts import ANALYZE_PROMPT, LEGAL_QUERY_PROMPT, DOC_RELATION_PROMPT

def make_analyze_node(llm_client: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Tạo node phân tích câu hỏi:
    - Phân rã, chia nhỏ câu hỏi.
    - Xác định intent của câu hỏi.
    """
    def analyze_node(state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "")
        
        # 1. Ghép câu hỏi của user vào prompt mẫu
        prompt = ANALYZE_PROMPT + f"\n\nCâu hỏi người dùng: {question}"
        
        try:
            # 2. Gọi remote client (hàm generate mà bạn tự viết)
            # Nó sẽ trả về chuỗi text
            raw_response = llm_client.generate(prompt=prompt)
            
            # 3. Làm sạch chuỗi trả về (Rất quan trọng với local LLM!)
            # Local LLM hay bị tật thêm ```json ... ``` bao quanh kết quả. Ta cần cắt bỏ nó đi.
            clean_json_str = raw_response
            match = re.search(r"```(?:json)?(.*?)```", raw_response, re.DOTALL)
            if match:
                clean_json_str = match.group(1).strip()
                
            # 4. Ép chuỗi text thành Dictionary chuẩn của Python
            json_dict = json.loads(clean_json_str)
            
            # 5. Dùng Pydantic ép kiểu và kiểm tra lỗi (Đây là lý do cần AnalyzerOutput)
            # Nếu JSON thiếu trường (ví dụ thiếu 'intent' hoặc 'query'), Pydantic sẽ ném ra ValidationError ngay lập tức.
            output_obj = AnalyzerOutput.model_validate(json_dict)
            
            # 6. Thành công! Trả mảng sub_questions về cho AgentState
            return {"sub_questions": output_obj.sub_questions}
            
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            # Nếu con LLM sinh rác, hoặc lỗi mạng... ta phải có phương án dự phòng
            # để luồng Graph không bị sập. Ta ép nó chạy vào nhánh CHITCHAT (hoặc GENERAL).
            print(f"[Analyze Node Error]: {e}")
            fallback_sq = SubQuestion(
                query=question,
                intent=Intent.CHITCHAT
            )
            return {"sub_questions": [fallback_sq]}
            
    return analyze_node

def make_fallback_node(llm: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Tạo nhánh Fallback: Xử lý khi không xác định được intent rõ ràng.
    """
    def fallback_node(state: AgentState) -> Dict[str, Any]:
        sq = state.get("current_sub_question")
        query = sq.query if sq else state.get("question", "")
        
        message = (
            f"Mình chưa rõ ý của bạn ở câu: '{query}'. "
            "Bạn có thể cung cấp thêm thông tin chi tiết hơn, "
            "hoặc chỉ định rõ văn bản/điều luật liên quan được không?"
        )
        # Ghi thẳng vào biến answer để trả về cho người dùng
        return {"answer": message}
        
    return fallback_node

# def make_doc_retrieve_node(retriever: Any) -> Callable[[AgentState], Dict[str, Any]]:
#     """
#     Tạo nhánh doc_retrieve:
#     - Vector search (metadata chunk, doc meta)
#     - Trả về context.
#     """
#     def doc_retrieve_node(state: AgentState) -> Dict[str, Any]:
#         sq = state.get("current_sub_question")
#         # Vector search dựa trên sq.query và sq.keywords
#         pass
#         
#     return doc_retrieve_node

def make_legal_query_node(retriever: Any, llm_client: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Tạo nhánh legal_query:
    - Trích xuất metadata từ câu hỏi bằng LLM
    - Gọi chunk_metadata_search để lấy chính xác điều khoản
    """
    def legal_query_node(state: AgentState) -> Dict[str, Any]:
        sq = state.get("current_sub_question")
        query = sq.query if sq else state.get("question", "")
        print(f"\n[Legal Query Node] Đang trích xuất metadata cho: '{query}'...")
        
        # 1. Dùng LLM để trích xuất cấu trúc văn bản
        prompt = LEGAL_QUERY_PROMPT + f"\n\nCâu hỏi: {query}"
        try:
            raw_response = llm_client.generate(prompt=prompt)
            clean_json = raw_response
            match = re.search(r"```(?:json)?(.*?)```", raw_response, re.DOTALL)
            if match:
                clean_json = match.group(1).strip()
                
            json_dict = json.loads(clean_json)
            citation = LegalCitation.model_validate(json_dict)
            
            so_hieu_to_search = citation.so_hieu
            
            # 1.5. Nếu chỉ có tên văn bản (không có số hiệu), gọi DB để tìm số hiệu chuẩn
            if not so_hieu_to_search and citation.ten_van_ban:
                print(f"[Legal Query Node] Tìm số hiệu chuẩn cho tên văn bản: '{citation.ten_van_ban}'...")
                doc_output = retriever.doc_metadata_search(
                    ten_van_ban=citation.ten_van_ban, 
                    limit=1
                )
                if doc_output.success and doc_output.documents:
                    so_hieu_to_search = doc_output.documents[0].so_hieu
                    print(f"[Legal Query Node] Đã tìm thấy số hiệu chuẩn: {so_hieu_to_search}")
                else:
                    print(f"[Legal Query Node] Không tìm thấy văn bản nào có tên '{citation.ten_van_ban}'")
            
            # 2. Gọi tool tìm kiếm chính xác
            print(f"[Legal Query Node] Tìm metadata: so_hieu={so_hieu_to_search}, dieu={citation.dieu}, khoan={citation.khoan}")
            tool_output = retriever.chunk_metadata_search(
                so_hieu=so_hieu_to_search,
                phan=citation.phan,
                chuong=citation.chuong,
                muc=citation.muc,
                dieu=citation.dieu,
                khoan=citation.khoan,
                diem=citation.diem,
                top_k=5 # Lấy 5 chunk nếu có trùng lặp hoặc lấy các chunk con
            )
            
            # 3. Trả về context và tool_output
            if tool_output.success and tool_output.display_text:
                context_str = f"--- TRÍCH XUẤT CHÍNH XÁC CHO: '{query}' ---\n{tool_output.display_text}"
            else:
                context_str = f"--- KHÔNG TÌM THẤY ĐIỀU LUẬT CHÍNH XÁC CHO: '{query}' ---\n{tool_output.display_text or ''}"
                
            return {
                "context_text": [context_str],
                "tool_outputs": [tool_output]
            }
            
        except Exception as e:
            print(f"[Legal Query Error] {e}")
            # Fallback về text trống nếu lỗi
            return {"context_text": [f"--- LỖI TRÍCH XUẤT CHO: '{query}' ---"]}
        
    return legal_query_node

def make_general_node(retriever: Any, llm: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Tạo nhánh general:
    - Vector search -> Trả về context_text
    """
    def general_node(state: AgentState) -> Dict[str, Any]:
        sq = state.get("current_sub_question")
        query = sq.query if sq else state.get("question", "")
        
        print(f"\n[General Node] Đang tìm kiếm ngữ nghĩa cho: '{query}'...")
        
        # Gọi tool vector_search từ LegalAgentTools (truyền qua tham số retriever)
        tool_output = retriever.vector_search(
            query=query,
            top_k_retrieve=50, # Lấy 20 kết quả thô
            top_k_rerank=5,    # Giữ lại 5 kết quả tốt nhất
            use_rerank=True
        )
        
        # Đóng gói kết quả thành text để nối vào context_text chung
        if tool_output.success and tool_output.display_text:
            context_str = f"--- KẾT QUẢ TÌM KIẾM CHO: '{query}' ---\n{tool_output.display_text}"
        else:
            context_str = f"--- TÌM KIẾM CHO '{query}' KHÔNG CÓ KẾT QUẢ ---\n{tool_output.display_text or ''}"
            
        # Trả về cả context_text (dạng chuỗi) và tool_outputs (dạng object thô)
        # Để node evaluate_chunks phía sau có thể lôi object thô ra rerank/deduplicate nếu cần.
        return {
            "context_text": [context_str],
            "tool_outputs": [tool_output]
        }
    return general_node

def make_doc_relation_node(retriever: Any, llm_client: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Tạo nhánh doc_relation:
    - Trích xuất tên/số hiệu văn bản từ câu hỏi bằng LLM
    - Gọi doc_relation_search để lấy thông tin hiệu lực và quan hệ thay thế
    """
    def doc_relation_node(state: AgentState) -> Dict[str, Any]:
        sq = state.get("current_sub_question")
        query = sq.query if sq else state.get("question", "")
        
        print(f"\n[Doc Relation Node] Đang trích xuất văn bản mục tiêu cho: '{query}'...")
        
        # 1. Dùng LLM để trích xuất danh tính văn bản
        prompt = DOC_RELATION_PROMPT + f"\n\nCâu hỏi: {query}"
        try:
            raw_response = llm_client.generate(prompt=prompt)
            clean_json = raw_response
            match = re.search(r"```(?:json)?(.*?)```", raw_response, re.DOTALL)
            if match:
                clean_json = match.group(1).strip()
                
            json_dict = json.loads(clean_json)
            citation = LegalCitation.model_validate(json_dict)
            
            so_hieu_to_search = citation.so_hieu
            
            # 2. Tìm số hiệu chuẩn nếu chỉ có tên văn bản
            if not so_hieu_to_search and citation.ten_van_ban:
                print(f"[Doc Relation Node] Tìm số hiệu chuẩn cho tên văn bản: '{citation.ten_van_ban}'...")
                doc_output = retriever.doc_metadata_search(
                    ten_van_ban=citation.ten_van_ban, 
                    limit=1,
                    fuzzy_threshold=0.6
                )
                if doc_output.success and doc_output.documents:
                    so_hieu_to_search = doc_output.documents[0].so_hieu
                    print(f"[Doc Relation Node] Đã tìm thấy số hiệu chuẩn: {so_hieu_to_search}")
                else:
                    print(f"[Doc Relation Node] Không tìm thấy văn bản nào có tên '{citation.ten_van_ban}'")
            
            # 3. Tra cứu quan hệ bằng so_hieu_to_search
            if so_hieu_to_search:
                print(f"[Doc Relation Node] Tra cứu quan hệ cho số hiệu: {so_hieu_to_search}")
                tool_output = retriever.doc_relation_search(so_hieu=so_hieu_to_search)
            else:
                # Không fallback sang vector search ở nhánh này!
                # Nhánh doc_relation yêu cầu phải tìm đích danh văn bản.
                from src.agent.schemas import ToolOutput
                tool_output = ToolOutput(
                    tool_name="doc_relation_search",
                    success=False,
                    display_text=f"Không tìm thấy văn bản '{citation.ten_van_ban or query}' trong CSDL hiện tại để tra cứu thông tin/quan hệ."
                )

            # Đóng gói kết quả
            if tool_output.success and tool_output.display_text:
                context_str = f"--- QUAN HỆ VĂN BẢN CHO: '{query}' ---\n{tool_output.display_text}"
            else:
                context_str = f"--- LỖI TÌM KIẾM CHO '{query}' ---\n{tool_output.display_text or ''}"
                
            return {
                "context_text": [context_str],
                "tool_outputs": [tool_output]
            }
            
        except Exception as e:
            print(f"[Doc Relation Error] {e}")
            return {"context_text": [f"--- LỖI TRÍCH XUẤT CHO: '{query}' ---"]}

    return doc_relation_node

def make_evaluate_chunks_node(llm: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Tạo node tổng hợp và đánh giá:
    - Nhận kết quả từ 5 nhánh trên.
    - Đánh giá chất lượng của các chunk tìm kiếm được (deduplicate, rerank, filter noise).
    """
    def evaluate_chunks_node(state: AgentState) -> Dict[str, Any]:
        pass
        
    return evaluate_chunks_node

def make_generate_response_node(llm: Any) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Tạo node sinh câu trả lời:
    - Đưa context (đã qua đánh giá) vào LLM để sinh câu trả lời cuối cùng.
    """
    def generate_response_node(state: AgentState) -> Dict[str, Any]:
        pass
        
    return generate_response_node

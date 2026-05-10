import uvicorn
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import build_search_service
from src.api import RemoteAPIClient
from src.rag import RAGService
from fastapi import UploadFile, File, Form
from system.replace_file_service import ReplaceFileService
from system.database.db_service import DocumentDatabaseService
import shutil
import os

app = FastAPI(title="Legal QA Frontend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    use_remote_embedding: bool = True
    use_remote_rerank: bool = True

class ChatResponse(BaseModel):
    query: str
    answer: str
    context: str

# Khởi tạo RAG service lazily
search_service = None
api_client = None
rag_service = None

# def init_services(req: ChatRequest):
#     global search_service, api_client, rag_service
#     if not search_service:
#         search_service = build_search_service(
#             use_remote_embedding=req.use_remote_embedding,
#             use_rerank=True,
#             use_remote_rerank=req.use_remote_rerank,
#         )
#     if not api_client:
#         api_client = RemoteAPIClient()
    
#     rag_service = RAGService(
#         search_service=search_service,
#         api_client=api_client,
#         top_k_retrieve=req.top_k_retrieve,
#         top_k_rerank=req.top_k_rerank,
#         use_rerank=True,
#     )

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        init_services(req)
        result = rag_service.answer(query=req.query)
        
        # Trích xuất các source documents để làm citation
        context = ""
        if result.context:
            context = result.context
            
        return ChatResponse(
            query=result.query,
            answer=result.answer,
            context=context
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
def list_documents():
    try:
        db_service = DocumentDatabaseService()
        documents = db_service.get_all_documents(limit=1000)
        # Filter only active documents if needed, or return all
        return documents
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/replace-document")
async def replace_document(
    file: UploadFile = File(...),
    replaced_so_hieu: str = Form(...)
):
    try:
        # Create temp directory if not exists
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Build ChromaStore for ReplaceFileService
        from main import build_chroma_store
        chroma_store = build_chroma_store()
        
        replace_service = ReplaceFileService(chroma_store=chroma_store)
        result = replace_service.process(new_file_path=file_path, replaced_so_hieu=replaced_so_hieu)
        
        # Optional: cleanup temp file
        # os.remove(file_path)
        
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting API Server for Frontend at http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)

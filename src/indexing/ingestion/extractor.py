"""
- Sửa giá trị `INPUT_PATH`.
- Chạy: `python extractor.py`.
Luồng dữ liệu:
- `extract_file(INPUT_PATH)`:
    - Nếu là .pdf  -> `extract_pdf_text` 
    - Nếu là .docx -> `extract_docx_text`
"""
from pathlib import Path


def extract_file(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".pdf":
        from src.indexing.ingestion.pdf_extractor import extract_pdf_text
        return extract_pdf_text(str(file_path))
    
    elif ext == ".docx":
        from src.indexing.ingestion.docx_extractor import extract_docx_text
        return extract_docx_text(str(file_path))
    
    elif ext == ".doc":
        try:
            from doc2docx import convert
            docx_path = file_path.with_suffix('.docx')
            convert(str(file_path), str(docx_path))
        except Exception as e:
            raise ValueError(f"Không thể chuyển đổi .doc sang .docx: {e}")
        
        from src.indexing.ingestion.docx_extractor import extract_docx_text
        
        text = extract_docx_text(str(docx_path))

        try:
            docx_path.unlink()
        except Exception as e:
            pass

        return text  

    else:
        raise ValueError(
            f"Định dạng file không được hỗ trợ: {ext} (chỉ hỗ trợ .pdf, .docx, .doc)"
        )
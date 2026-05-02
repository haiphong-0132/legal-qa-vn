from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = PROJECT_ROOT / ".temp" / "Thong-tu-108-2020-TT-BTC.docx"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def shorten(text: str | None, max_length: int) -> str:
    if not text:
        return ""

    text = " ".join(str(text).split())
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def make_llm_client(enable_llm_refs: bool):
    """
    Tạo LLM client nếu bật reference extraction bằng LLM.

    Không hard-fail ở health_check để tránh script chết sớm nếu health endpoint timeout.
    Nếu /generate lỗi thật, parser sẽ log lỗi và ref có thể rỗng.
    """
    if not enable_llm_refs:
        return None

    from src.api import RemoteAPIClient

    client = RemoteAPIClient()

    if hasattr(client, "health_check"):
        try:
            is_ok = client.health_check()
            if not is_ok:
                print(
                    "⚠️ Warning: RemoteAPIClient health_check failed. "
                    "Script vẫn tiếp tục; nếu /generate lỗi thì reference có thể rỗng."
                )
        except Exception as exc:
            print(
                "⚠️ Warning: RemoteAPIClient health_check error. "
                f"Script vẫn tiếp tục. Error: {exc}"
            )

    return client


def run_chunking(
    file_path: Path,
    limit: int,
    output_path: Path | None,
    use_llm_refs: bool,
    print_full_json: bool,
) -> None:
    from src.indexing.chunker import create_chunker

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    llm_client = make_llm_client(use_llm_refs)

    chunker = create_chunker(
        strategy="hierarchical",
        use_llm_refs=use_llm_refs,
        llm_client=llm_client,
    )

    metadata, chunks = chunker.create_document_node(str(file_path))

    print(f"File: {file_path}")
    print("Strategy: hierarchical")
    print(f"use_llm_refs: {use_llm_refs}")
    print(f"Số hiệu: {getattr(metadata, 'so_hieu', '')}")
    print(f"Tên văn bản: {getattr(metadata, 'ten_van_ban', '')}")
    print(f"Tổng số chunk: {len(chunks)}")

    ref_chunk_count = sum(1 for chunk in chunks if getattr(chunk, "reference", None))
    ref_count = sum(len(chunk.reference or []) for chunk in chunks)

    print(f"Số chunk có reference: {ref_chunk_count}")
    print(f"Tổng số reference: {ref_count}")

    print("\nThống kê theo type:")
    for chunk_type, count in Counter(chunk.type or "unknown" for chunk in chunks).most_common():
        print(f"  {chunk_type}: {count}")

    print(f"\nPreview {min(limit, len(chunks))} chunk đầu:")
    print("-" * 100)

    for index, chunk in enumerate(chunks[:limit], start=1):
        print(f"#{index:03d} id={chunk.id} | type={chunk.type} | parent={chunk.parent_id or ''}")

        title = shorten(chunk.title, 120)
        content = shorten(chunk.content, 220)
        full_text = shorten(chunk.full_text, 220)
        parent_context = shorten(chunk.parent_context, 160)
        references = ", ".join(chunk.reference or [])

        if title:
            print(f"  title: {title}")
        if parent_context:
            print(f"  parent_context: {parent_context}")
        if content:
            print(f"  content: {content}")
        elif full_text:
            print(f"  full_text: {full_text}")
        if references:
            print(f"  ref: {references}")

    print("-" * 100)

    payload = {
        "file": str(file_path),
        "strategy": "hierarchical",
        "use_llm_refs": use_llm_refs,
        "metadata": dump_model(metadata),
        "total_chunks": len(chunks),
        "ref_chunk_count": ref_chunk_count,
        "ref_count": ref_count,
        "chunks": [dump_model(chunk) for chunk in chunks],
    }

    json_str = json.dumps(payload, ensure_ascii=False, indent=2)

    auto_json = PROJECT_ROOT / "chunk" / f"{getattr(metadata, 'so_hieu', '')}.json"
    auto_json.parent.mkdir(parents=True, exist_ok=True)
    auto_json.write_text(json_str, encoding="utf-8")
    print(f"\n✅ File JSON tự sinh: {auto_json}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")
        print(f"✅ File JSON đầy đủ: {output_path}")

    if print_full_json:
        print("\n" + "=" * 100)
        print("JSON OUTPUT (FULL CHUNKING RESULT)")
        print("=" * 100)
        print(json_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test hierarchical chunking cho văn bản pháp luật, có hỗ trợ LLM reference extraction."
    )

    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help=f"Đường dẫn file .docx/.doc/.pdf, mặc định: {DEFAULT_FILE}",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Số chunk đầu tiên cần in preview.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Nếu truyền vào, lưu toàn bộ chunks ra file JSON.",
    )

    parser.add_argument(
        "--no-llm-refs",
        action="store_true",
        help="Tắt LLM reference extraction. Parser/chunker vẫn chạy nhưng reference có thể rỗng.",
    )

    parser.add_argument(
        "--no-print-json",
        action="store_true",
        help="Không in full JSON ra console, chỉ ghi file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_chunking(
        file_path=args.file,
        limit=max(args.limit, 0),
        output_path=args.output,
        use_llm_refs=not args.no_llm_refs,
        print_full_json=not args.no_print_json,
    )
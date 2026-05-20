from __future__ import annotations

import argparse
import json
from pathlib import Path
from bm_25_wand.utils import tokenize_underthesea_text

tokenize_text = tokenize_underthesea_text

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenize",
        description="Tokenize JSONL files and cache tokenized text as JSONL.",
    )
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--id-field", default="", help="Optional explicit id field name")
    parser.add_argument(
        "--text-fields",
        default="title,text",
        help="Comma-separated legal corpus fields to join and tokenize",
    )
    return parser


def _resolve_id(obj: dict, explicit_field: str) -> str:
    if explicit_field:
        value = obj.get(explicit_field)
        if value is None:
            raise KeyError(f"Missing id field: {explicit_field}")
        return str(value)

    for key in ("_id", "doc_id", "query-id"):
        if key in obj and obj[key] is not None:
            return str(obj[key])

    raise KeyError("Could not resolve an id field. Pass --id-field explicitly.")


def _resolve_text(obj: dict, fields: list[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = obj.get(field)
        if value:
            parts.append(str(value))
    return " ".join(parts).strip()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    text_fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            item_id = _resolve_id(obj, args.id_field)
            text = _resolve_text(obj, text_fields)
            tokens = list(tokenize_text(text))
            fout.write(
                json.dumps(
                    {
                        "id": item_id,
                        "doc_id": item_id,
                        "tokens": tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    print(f"Wrote {count} tokenized rows to {output_path}")


if __name__ == "__main__":
    main()
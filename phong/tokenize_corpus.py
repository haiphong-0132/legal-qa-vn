import json
from pathlib import Path
from tqdm import tqdm
from .utils import tokenize

def tokenize_corpus(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} documents from {input_path}")

    tokenized_data = []

    for item in tqdm(data, desc="Tokenizing corpus", unit="doc", total=len(data)):
        title = item.get("title", "")
        text = item.get("text", "")

        tokenized_title = tokenize(title)
        tokenized_text = tokenize(text)

        tokenized_item = {
            "_id": item.get("_id", ""),
            "title": tokenized_title,
            "text": tokenized_text
        }
        tokenized_data.append(tokenized_item)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tokenized_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Tokenized corpus saved to {output_path}")

if __name__ == "__main__": 
    INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "data_zalo" / "corpus.jsonl"
    OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "data_zalo" / "corpus_tokenized.jsonl"

    tokenize_corpus(INPUT_PATH, OUTPUT_PATH)

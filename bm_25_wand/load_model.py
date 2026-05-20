from __future__ import annotations

import argparse
from pathlib import Path

from bm_25_wand.engine import BM25SearchEngine
def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="bm25_load_model",
		description="Load a BM25 model and print summary stats.",
	)
	parser.add_argument("--model-dir", required=True, help="Directory containing bm25_model.joblib")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	model_dir = Path(args.model_dir)
	engine = BM25SearchEngine.load(model_dir)

	doc_count = len(engine.documents)
	vocab_size = len(engine.inverted_index)
	avgdl = engine.avgdl
	champion_enabled = engine.champion_size > 0
	champion_terms = len(engine.champion_index)
	has_term_max = bool(engine.term_max_score)

	print(f"model_dir={model_dir}")
	print(f"documents={doc_count}")
	print(f"vocab_size={vocab_size}")
	print(f"avgdl={avgdl:.2f}")
	print(f"champion_size={engine.champion_size}")
	print(f"champion_enabled={champion_enabled}")
	print(f"champion_terms={champion_terms}")
	print(f"term_max_score_loaded={has_term_max}")


if __name__ == "__main__":
	main()
# uv run python -m bm_25_wand.load_model --model-dir models/bm25
'''
model_dir=models\bm25
documents=61425
vocab_size=35979
avgdl=238.04
champion_size=0
champion_enabled=False
champion_terms=0
term_max_score_loaded=True
'''
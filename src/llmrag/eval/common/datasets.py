from typing import List, Dict, Tuple
import json, os

def load_json_dataset(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_qa_pairs(dataset: List[Dict]) -> List[Tuple[str, str]]:
    """
    Expected shape in your current JSONs:
      [{ "metadata": {...}, "input_text": "...", "qa_pairs": [{"question": "...", "answer": "..."}] }, ...]
    """
    qa = []
    for item in dataset:
        for qa_pair in item.get("qa_pairs", []):
            q = qa_pair.get("question", "").strip()
            a = qa_pair.get("answer", "").strip()
            if q and a:
                qa.append((q, a))
    return qa

def prepare_rag_evaluation_data(file_path: str) -> List[Tuple[str, str]]:
    dataset = load_json_dataset(file_path)
    return extract_qa_pairs(dataset)

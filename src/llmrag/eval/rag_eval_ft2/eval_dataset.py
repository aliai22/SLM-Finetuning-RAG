import json
import os
from typing import List, Dict, Tuple

def load_json_dataset(file_path: str) -> List[Dict]:
    """
    Load the synthetic QA dataset from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List[Dict]: A list of dictionaries containing metadata, input_text, and qa_pairs.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def extract_qa_pairs(dataset: List[Dict]) -> List[Tuple[str, str, str]]:
    """
    Extracts QA pairs from the dataset and associates them with their respective input texts.
    Skips entries where 'qa_pairs' is not a list or contains fewer than two items.

    Args:
        dataset (List[Dict]): Parsed JSON dataset.

    Returns:
        List[Tuple[str, str, str]]: List of (question, answer, input_text) tuples.
    """
    qa_list = []

    for entry in dataset:
        input_text = entry.get("input_text", "").strip()
        qa_pairs = entry.get("qa_pairs")

        # Ensure 'qa_pairs' is a list and contains more than one element
        if not isinstance(qa_pairs, list) or len(qa_pairs) < 2:
            continue  # Skip this entry

        for qa in qa_pairs:
            question = qa.get("Question", "").strip()
            answer = qa.get("Answer", "").strip()

            if question and answer and input_text:
                qa_list.append((question, answer, input_text))

    return qa_list


def preprocess_text(text: str) -> str:
    """
    Preprocesses text by normalizing whitespace, converting to lowercase, and stripping.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    return " ".join(text.lower().strip().split())


def prepare_rag_evaluation_data(file_path: str) -> List[Dict]:
    """
    Prepares the dataset for RAG evaluation by extracting and preprocessing QA pairs.

    Args:
        file_path (str): Path to the JSON dataset.

    Returns:
        List[Dict]: Processed dataset with 'question', 'answer', and 'context' for evaluation.
    """
    dataset = load_json_dataset(file_path)
    qa_pairs = extract_qa_pairs(dataset)

    processed_data = []
    for question, answer, context in qa_pairs:
        processed_data.append({
            "question": preprocess_text(question),
            "answer": preprocess_text(answer),
            "context": preprocess_text(context)
        })

    return processed_data

import jsonlines
from typing import List

def load_dataset(path: str) -> List[str]:
    """
    Loads a dataset from a JSONL file where each line contains a 'question' and 'answer'.
    Converts each pair into a formatted string like "Q: ...\nA: ..."
    """
    qa_pairs = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            question = obj.get("question", "").strip()
            answer = obj.get("answer", "").strip()

            # Skip if either is empty
            if not question or not answer:
                continue

            qa_text = f"Q: {question}\nA: {answer}"
            qa_pairs.append(qa_text)

    print(f"✅ Loaded {len(qa_pairs)} QA pairs from {path}")
    return qa_pairs


class CustomString(str):
    """
    Extends str to hold additional metadata (e.g., page number or index).
    """
    def __new__(cls, content, metadata=None):
        obj = super().__new__(cls, content)
        obj.page_content = content
        obj.metadata = metadata
        return obj


def preprocess_dataset(data: List[str]) -> List[CustomString]:
    """
    Converts raw string data into CustomString objects with index metadata.
    """
    qa_chunks = [CustomString(content, {"index": idx}) for idx, content in enumerate(data)]
    print(f"✅ Preprocessed dataset into {len(qa_chunks)} chunks")
    return qa_chunks


def batchify(data: List, batch_size: int):
    """
    Yields successive batches of the specified size from the input list.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]
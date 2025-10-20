import jsonlines
from typing import List

# def load_dataset(path:str):
#     qa_pairs = []
#     with jsonlines.open(path) as reader:
#         for obj in reader:
#             qa_text = f"Q: {obj['question']}\nA: {obj['answer']}"
#             # text = obj['answer']
#             qa_pairs.append(qa_text)
#     # filtered = [a["answer"] for a in qa_pairs if a["answer"] is not None]
#     flattened = [qa for qa in qa_pairs]
#     print("Dataset Loaded Successfully!")
#     return flattened

def load_dataset(path:str):
    all_contents = []
    all_metas = []
    # i=0
    with jsonlines.open(path) as reader:
        for obj in reader:
            if len(obj["content"]) >= 512:
                all_contents.append(obj["content"])
                all_metas.append(obj["metadata"])
            # print(len(obj['content']))
            # i+=1
            # if i==10:
            #     break
    print(f"Dataset Loaded Successfully! Total Size {len(all_contents)}")
    return all_contents, all_metas

# # Custom class that extends str
# class CustomString(str):
#     def __new__(cls, content, metadata):
#         obj = super().__new__(cls, content)
#         obj.page_content = content
#         obj.metadata = metadata
#         return obj

# def preprocess_dataset(data:list):
#     qa_chunks = [CustomString(content, {"index": idx}) for idx, content in enumerate(data)]
#     print("Dataset Preprocessed Successfully!")
#     return qa_chunks

# Custom class that extends str to store metadata
class CustomString(str):
    def __new__(cls, content, metadata):
        obj = super().__new__(cls, content)
        obj.page_content = content
        obj.metadata = metadata
        return obj

def preprocess_dataset(data: list, metadata_list: list):
    if len(data) != len(metadata_list):
        raise ValueError("Data and metadata lists must have the same length")

    qa_chunks = [CustomString(content, metadata) for content, metadata in zip(data, metadata_list)]
    print("Dataset Preprocessed Successfully!")
    return qa_chunks


def batchify(data: List, batch_size: int):
    """Yield successive batches from the dataset."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

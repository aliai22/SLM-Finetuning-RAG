import json
import jsonlines
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

def preprocessing_QAs(input_file, output_file):
    total_count = 0  # Count of all entries
    valid_count = 0  # Count of valid entries

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            total_count += 1  # Increment total count
            data = json.loads(line)  # Load each line as a JSON object
            
            if data.get("question") is not None and data.get("answer") is not None:
                # if valid_count >= 20000:
                #     break  # Stop processing if we reach the limit
                valid_count += 1  # Increment valid count
                outfile.write(json.dumps(data) + "\n")  # Write valid pairs to the new file

    print(f"Total QA pairs before filtering: {total_count}")
    print(f"Total QA pairs after filtering: {valid_count}")
    print(f"Removed {total_count - valid_count} QA pairs.")
    # print(f"Stopped after reaching {valid_count} valid QA pairs.")
    dataset = load_dataset("json", data_files = output_file)
    # rename columns
    dataset = dataset.rename_column("question", "anchor")
    dataset = dataset.rename_column("answer", "positive")
    # Add an id column to the dataset
    dataset = dataset["train"].add_column("id", range(len(dataset["train"])))
    # split dataset into a 10% test set
    dataset = dataset.train_test_split(test_size=0.1)
    # save datasets to disk
    dataset["train"].to_json("./finetuning_embeddModel/train_dataset.json", orient="records")
    dataset["test"].to_json("./finetuning_embeddModel/test_dataset.json", orient="records")

def loading_dataset(test_path:str, train_path:str):
    test_dataset = load_dataset("json", data_files=test_path, split="train")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])
    # Convert the datasets to dictionaries
    corpus = dict(
        zip(corpus_dataset["id"], corpus_dataset["positive"])
    )  # Our corpus (cid => document)
    queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )  # Our queries (qid => question)
    
    # Create a mapping of relevant document (1 in our case) for each query
    relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    return train_dataset, test_dataset, corpus, queries, relevant_docs

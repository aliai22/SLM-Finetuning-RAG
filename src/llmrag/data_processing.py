# import pandas as pd
# import numpy as np
from datasets import load_dataset

# file_path = ""

def dataset_load(file_path):
    dataset = load_dataset("json", data_files=file_path)
    return dataset

def dataset_split(dataset):
    split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
    return split_dataset
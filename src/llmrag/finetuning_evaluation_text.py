from LLM_Model import load_local_llm
from finetuning_textbooks.data_loading import group_texts, PDFtext_to_chunks, chunks_to_jsonl, tokenize_function, get_pdf_config, loading_dataset
from finetuning import prepare_peft_model, prepare_peft_trainer

import os
from glob import glob
import torch
from functools import partial

import time
import torch
import threading
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown
import pandas as pd


llm_dir = "./LLMs"
model_id = "microsoft/phi-2"

base_llm, tokenizer, eval_tokenizer = load_local_llm(local_dir=llm_dir,
               base_model_id=model_id,
               )

generate_dataset = False # Change if PDFs are already processed and dataset is created

books_dir = "./AIbooks_dataset/AI-books"

file_paths = sorted(glob(f"{books_dir}/*"))
config_file = "./AIbooks_dataset/pdfs_config.json"
data_output_path = "./AIbooks_dataset/text_finetuningData.jsonl"

if generate_dataset:
    for i, file_path in enumerate(sorted(file_paths)):
        skip_start_pages, skip_last_pages, header_lines, footer_lines = get_pdf_config(file_path=config_file, index=i)
        text_chunks = PDFtext_to_chunks(pdf_path=file_path,
                        skip_start_pages=skip_start_pages,
                        skip_last_pages=skip_last_pages,
                        header_lines=header_lines,
                        footer_lines=footer_lines)
        chunks_to_jsonl(data=text_chunks,
                        output_file=data_output_path)
        print(f"Dataset Prepared and Saved to {data_output_path} for {os.path.basename(file_path)}.\n")

train_dataset, test_dataset = loading_dataset(file_path=data_output_path)

print(type(test_dataset))
print(test_dataset[0])
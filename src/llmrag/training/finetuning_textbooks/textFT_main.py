from LLM_Model import load_local_llm
from .data_loading import group_texts, PDFtext_to_chunks, chunks_to_jsonl, tokenize_function, get_pdf_config, loading_dataset
from finetuning import prepare_peft_model, prepare_peft_trainer

import os
from glob import glob
import torch
from functools import partial

from helpers import get_max_len, preprocess_dataset, print_number_of_trainable_model_parameters


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


tokenize_with_tokenizer = partial(tokenize_function, tokenizer)
tokenized_trainData = train_dataset.map(tokenize_with_tokenizer,
                                        batched=True,
                                        num_proc=4,
                                        remove_columns=["content", "metadata"])

tokenized_valData = test_dataset.map(tokenize_with_tokenizer,
                                     batched=True,
                                     num_proc=4,
                                     remove_columns=["content", "metadata"])

lm_trainDataset = tokenized_trainData.map(
    group_texts,
    batched=True,
    batch_size=32,
    num_proc=4,
)
lm_valDataset = tokenized_valData.map(
    group_texts,
    batched=True,
    batch_size=32,
    num_proc=4,
)

peft_model = prepare_peft_model(model=base_llm,
                   )
training_data_dir = "./finetuning_textbooks/finetuning_checkpoints/final_checkpoint"
peft_trainer = prepare_peft_trainer(output_dir=training_data_dir,
                     peft_model=peft_model,
                     training_dataset=lm_trainDataset,
                     evaluation_dataset=lm_valDataset,
                     tokenizer=tokenizer)


print("Original LLM Trainable Parameters:\n")
print(print_number_of_trainable_model_parameters(base_llm))

print("Peft Model Trainable Parameters:\n")
print(print_number_of_trainable_model_parameters(peft_model))
# import gc
# gc.collect()
# torch.cuda.empty_cache()

# print("Training Started!")

# # Initialize NVIDIA Management Library
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)  # Assuming you are using GPU 0

# # Log file
# log_file = "./finetuning_textbooks/gpu_training_log2.0.csv"

# # Function to log GPU stats in real-time
# gpu_stats = []
# training_running = True  # Flag to control logging
# start_time = time.time()  # Start time for training

# def monitor_gpu():
#     global gpu_stats
#     save_interval = 60  # Save logs every 5 minutes (300 seconds)
#     last_save_time = time.time()
    
#     while training_running:
#         utilization = nvmlDeviceGetUtilizationRates(handle).gpu  # GPU usage in %
#         memory_used = nvmlDeviceGetMemoryInfo(handle).used / 1024**3  # Convert to GB
#         current_time = time.time() - start_time
#         gpu_stats.append((current_time, utilization, memory_used))

#         # Save logs every 5 minutes
#         if time.time() - last_save_time >= save_interval:
#             save_logs()
#             last_save_time = time.time()

#         time.sleep(5)  # Log every 5 seconds

# def save_logs():
#     """Save GPU usage logs periodically to a CSV file."""
#     df = pd.DataFrame(gpu_stats, columns=["Time (s)", "GPU Utilization (%)", "Memory Used (GB)"])
#     df.to_csv(log_file, index=False)
#     print(f"Logs saved to {log_file}")

# # Start GPU monitoring in a separate thread
# gpu_thread = threading.Thread(target=monitor_gpu)
# gpu_thread.start()

# # Start training

# peft_trainer.train(resume_from_checkpoint="./finetuning_textbooks/finetuning_checkpoints/final_checkpoint/checkpoint-5000")

# # Stop GPU monitoring
# training_running = False
# gpu_thread.join()

# # Final save of logs
# save_logs()

# # Compute total time and peak memory usage
# end_time = time.time()
# total_time = end_time - start_time
# peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB

# # Print Results
# print(f"Total Training Time: {total_time:.2f} seconds")
# print(f"Peak GPU Memory Usage: {peak_memory:.2f} GB")

# # Shutdown NVIDIA library
# nvmlShutdown()
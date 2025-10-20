from LLM_Model import load_llm, save_llm_locally, load_local_llm
from data_processing import dataset_load, dataset_split
from finetuning import prepare_peft_model, prepare_peft_trainer
from helpers import get_max_len, preprocess_dataset, print_number_of_trainable_model_parameters

from transformers import set_seed
import torch
import os

import time
import torch
import threading
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown
import pandas as pd

model_name = "microsoft/phi-2"

original_llm, tokenizer, eval_tokenizer = load_llm(base_model_id=model_name)

print("Model Loaded Successfully from HuggingFace!")


llm_path = "./LLMs"
if not os.path.exists(f"{llm_path}/llm"):
    save_llm_locally(original_llm, tokenizer, eval_tokenizer, llm_path)

print("Model Saved Successfully to Local Directory!")

model, tokenizer, eval_tokenizer = load_local_llm(llm_path)

print("Model Loaded Successfully from Local Directory!")

output_dir = "./Finetuning_Checkpoints_filtered1.0/final-checkpoint"
dataset_path = "uniqueQA_dataset0.7.jsonl"

dataset = dataset_load(file_path=dataset_path)
print("Data File Loaded Successfully!")

splitted_dataset = dataset_split(dataset=dataset)
print("Data Split Created Successfully!")

print(splitted_dataset['train'])
print(splitted_dataset['test'])

max_length = get_max_len(model=model)

seed = 42
set_seed(seed)

print("Dataset Preprocessing .....")

train_dataset = preprocess_dataset(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["train"])
eval_dataset = preprocess_dataset(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["test"])

print("Dataset Preprocessed!")

print("Original LLM Trainable Parameters:\n")
print(print_number_of_trainable_model_parameters(model))

peft_model = prepare_peft_model(model=model)

print("Peft Model Trainable Parameters:\n")
print(print_number_of_trainable_model_parameters(peft_model))

peft_trainer = prepare_peft_trainer(output_dir=output_dir,
                                    peft_model=peft_model,
                                    training_dataset=train_dataset,
                                    evaluation_dataset=eval_dataset,
                                    tokenizer=tokenizer)

import gc
gc.collect()
torch.cuda.empty_cache()

# ckpt_dir = "Finetuning Checkpoints/final-checkpoint/checkpoint-2000"

print("Training Started!")

# Initialize NVIDIA Management Library
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # Assuming you are using GPU 0

# Log file
log_file = "gpu_training_log_filtered4.0.csv"

# Function to log GPU stats in real-time
gpu_stats = []
training_running = True  # Flag to control logging
start_time = time.time()  # Start time for training

def monitor_gpu():
    global gpu_stats
    save_interval = 300  # Save logs every 5 minutes (300 seconds)
    last_save_time = time.time()
    
    while training_running:
        utilization = nvmlDeviceGetUtilizationRates(handle).gpu  # GPU usage in %
        memory_used = nvmlDeviceGetMemoryInfo(handle).used / 1024**3  # Convert to GB
        current_time = time.time() - start_time
        gpu_stats.append((current_time, utilization, memory_used))

        # Save logs every 5 minutes
        if time.time() - last_save_time >= save_interval:
            save_logs()
            last_save_time = time.time()

        time.sleep(5)  # Log every 5 seconds

def save_logs():
    """Save GPU usage logs periodically to a CSV file."""
    df = pd.DataFrame(gpu_stats, columns=["Time (s)", "GPU Utilization (%)", "Memory Used (GB)"])
    df.to_csv(log_file, index=False)
    print(f"Logs saved to {log_file}")

# Start GPU monitoring in a separate thread
gpu_thread = threading.Thread(target=monitor_gpu)
gpu_thread.start()

# Start training
peft_trainer.train()

# Stop GPU monitoring
training_running = False
gpu_thread.join()

# Final save of logs
save_logs()

# Compute total time and peak memory usage
end_time = time.time()
total_time = end_time - start_time
peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB

# Print Results
print(f"Total Training Time: {total_time:.2f} seconds")
print(f"Peak GPU Memory Usage: {peak_memory:.2f} GB")

# Shutdown NVIDIA library
nvmlShutdown()
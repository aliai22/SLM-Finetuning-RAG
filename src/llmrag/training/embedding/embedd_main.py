from data_loading import preprocessing_QAs, loading_dataset
from evaluation import generate_evaluator
from embedd_finetuning import load_model, define_loss, prepare_trainer
import torch

import time
import torch
import threading
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, nvmlShutdown
import pandas as pd

dataset_path = "dataset.jsonl"
output_path = "./finetuning_embeddModel/filtered_qa_pairs.jsonl"

preprocessing_QAs(input_file=dataset_path,
                  output_file=output_path)

train_dataset, test_dataset, corpus, queries, relevant_docs = loading_dataset(test_path="./finetuning_embeddModel/test_dataset.json",
                                                                            train_path="./finetuning_embeddModel/train_dataset.json")

evaluator = generate_evaluator(queries=queries,
                   corpus=corpus,
                   relevant_docs=relevant_docs)

base_model_id = "BAAI/bge-base-en-v1.5"
base_model = load_model(model_id=base_model_id)

trainer_loss = define_loss(model=base_model)

output_dir = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"

trainer = prepare_trainer(model=base_model,
                train_dataset=train_dataset,
                train_loss=trainer_loss,
                evaluator=evaluator,
                output_dir=output_dir)

import gc
gc.collect()
torch.cuda.empty_cache()

print("Training Started!")

# Initialize NVIDIA Management Library
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)  # Assuming you are using GPU 0

# Log file
log_file = "./finetuning_embeddModel/gpu_training_log.csv"

# Function to log GPU stats in real-time
gpu_stats = []
training_running = True  # Flag to control logging
start_time = time.time()  # Start time for training

def monitor_gpu():
    global gpu_stats
    save_interval = 60  # Save logs every 5 minutes (300 seconds)
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
trainer.train(resume_from_checkpoint="./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0/checkpoint-5000")
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

# save the best model
trainer.save_model()
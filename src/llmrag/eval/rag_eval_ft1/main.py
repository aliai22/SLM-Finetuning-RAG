# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print(sys.path)

import csv
import os
from finetuning_embeddModel.embedd_finetuning import load_model
from FT_1RAG.rag_main import run_rag_pipeline
from .eval_dataset import prepare_rag_evaluation_data
from .evaluator import evaluate_rag

eval_dataset_path = "synthetic_QAs.json"
csv_log_path = "rag_eval_results.csv"

# Load evaluation dataset
processed_dataset = prepare_rag_evaluation_data(file_path=eval_dataset_path)
print(f"Total QA Pairs: {len(processed_dataset)}")

# Load embedding model
embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

# Prepare CSV logging
csv_fields = [
    "question", "ground_truth", "generated_answer", "normalized_prediction",
    "em", "f1", "bleu", "rouge1", "rougeL", "embedding_similarity"
]

write_header = not os.path.exists(csv_log_path)
with open(csv_log_path, mode="a", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()

    results_list = []
    for i, qa_pair in enumerate(processed_dataset, start=1):
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]
        print(f"\n🔍 Q{i}: {question}\n🔑 Ground Truth: {ground_truth}")

        generated_answer, _ = run_rag_pipeline(query=question)

        eval_results = evaluate_rag(
            question=question,
            ground_truth=ground_truth,
            rag_response=generated_answer,
            similarity_model=embedd_model
        )

        results_list.append(eval_results)
        writer.writerow(eval_results)
        print(f"✅ Evaluation {i}:", eval_results)

        if i == 100:
            break  # Remove or modify if full evaluation is needed

# Print aggregated metrics
all_metrics = ["em", "f1", "bleu", "rouge1", "rougeL", "embedding_similarity"]
averages = {metric: sum(d[metric] for d in results_list) / len(results_list) for metric in all_metrics}

print("\n=== 📊 Final Evaluation Summary ===")
for metric, avg in averages.items():
    print(f"{metric.upper()}: {avg:.4f}")


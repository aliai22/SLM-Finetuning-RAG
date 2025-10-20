# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# print(sys.path)

import csv
import os
import json
from finetuning_embeddModel.embedd_finetuning import load_model
from FT_2RAG.rag_main import run_rag_pipeline
from .eval_dataset import prepare_rag_evaluation_data
from .evaluator import evaluate_rag
import re
from sklearn.metrics import ndcg_score

# ---- load gold-document IDs built earlier -----------------
with open("gold_doc_ids.json", "r", encoding="utf-8") as f:
    GOLD_MAP = json.load(f)        # keys are "0", "1", ...

eval_dataset_path = "synthetic_QAs.json"
csv_log_path = "rag_KW(2)_ablation_ON.csv"

KW_DB_PATH = "keywords_database.json"

with open(KW_DB_PATH, "r", encoding="utf-8") as f:
    keyword_db = json.load(f) 

def normalize(text):
    text = re.sub(r'\W+', ' ', text.lower()).strip()
    return re.sub(r'\s+', ' ', text)

def contains_answer(doc_text, answer):
    return normalize(answer) in normalize(doc_text)

K = 5                    # cut-off for P@K, R@K, nDCG@K

# ------------------------------------------------------------------
#  Helper: build binary relevance vector
# ------------------------------------------------------------------
def relevance_vector(retrieved_ids, gold_ids, k=K):
    """Return [1/0] list length k indicating relevance of each retrieved doc."""
    return [1 if rid in gold_ids else 0 for rid in retrieved_ids[:k]]

# Load evaluation dataset
processed_dataset = prepare_rag_evaluation_data(file_path=eval_dataset_path)
print(f"Total QA Pairs: {len(processed_dataset)}")

# Load embedding model
embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

# Prepare CSV logging
csv_fields = [
    "question", "ground_truth", "generated_answer", "normalized_prediction",
    "em", "f1", "bleu", "rouge1", "rougeL", "embedding_similarity", "precision@5", "recall@5", "mrr", "ndcg@5", "map", "avg_rank"
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

        generated_answer, filtered_docs = run_rag_pipeline(query=question)
        # filtered_docs_only = [d for d, _ in filtered_docs]
        # 1) Collect retrieved IDs
        retrieved_ids = [d.metadata["doc_id"] if isinstance(d, tuple) else d.metadata["doc_id"]
                        for d in filtered_docs]
        
        # ------------------------------------------------------------------
        #  Retrieve gold IDs for this sample  (index used as string key)
        # ------------------------------------------------------------------
        sample_id = str(i-1)                      # because enumerate starts at 1
        gold_ids  = set(GOLD_MAP.get(sample_id, []))   # may be empty set
        print(f"CURRENT GOLD_ID: {gold_ids}")
        # ------------------------------------------------------------------
        #  Build binary relevance vector & metrics
        # ------------------------------------------------------------------
        retrieved_ids = [d.metadata["doc_id"] for d in filtered_docs]
        rel_vec       = relevance_vector(retrieved_ids, gold_ids, k=K)

        precision_k = sum(rel_vec) / K
        recall_k    = int(bool(sum(rel_vec)))
        try:
            first_pos = rel_vec.index(1) + 1
            mrr       = 1 / first_pos
            avg_rank  = first_pos
        except ValueError:
            mrr = 0.0
            avg_rank = K + 1

        ndcg = ndcg_score([rel_vec], [rel_vec])

        hits, precisions = 0, []
        for idx, rel in enumerate(rel_vec, 1):
            if rel:
                hits += 1
                precisions.append(hits / idx)
        ap = sum(precisions) / (len(gold_ids) if gold_ids else 1)

        # retrieved_ids   = []
        # retrieval_hit   = 0

        # for rank, doc in enumerate(filtered_docs, 1):
        #     retrieved_ids.append(doc.metadata["doc_id"])
        #     if contains_answer(doc.page_content, ground_truth):
        #         retrieval_hit = 1
        #         reciprocal_rank = 1 / rank
        #         break
        # else:
        #     reciprocal_rank = 0

        eval_results = evaluate_rag(
            question=question,
            ground_truth=ground_truth,
            rag_response=generated_answer,
            similarity_model=embedd_model
        )

        eval_results.update({
            "precision@5": precision_k,
            "recall@5":    recall_k,
            "mrr":         mrr,
            "ndcg@5":      ndcg,
            "map":         ap,
            "avg_rank":    avg_rank
        })

        results_list.append(eval_results)
        writer.writerow(eval_results)
        print(f"✅ Evaluation {i}:", eval_results)
        break
        if i == 100:
            break  # Remove or modify if full evaluation is needed

# Print aggregated metrics
all_metrics = ["em", "f1", "bleu", "rouge1", "rougeL", "embedding_similarity", "precision@5", "recall@5", "mrr", "ndcg@5", "map", "avg_rank"]
averages = {metric: sum(d[metric] for d in results_list) / len(results_list) for metric in all_metrics}

print("\n=== 📊 Final Evaluation Summary ===")
for metric, avg in averages.items():
    print(f"{metric.upper()}: {avg:.4f}")
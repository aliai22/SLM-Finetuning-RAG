from __future__ import annotations
import os, json, csv, re
from typing import Dict, List
from sklearn.metrics import ndcg_score

# Light imports at module level; defer heavy ones to run-time in real run
from llmrag.eval.common.datasets import prepare_rag_evaluation_data
from llmrag.eval.common.runners import evaluate_rag

# -------------------- Real run --------------------
def run_kw_ablation(
    eval_path: str = ".data/finetuning/qa/synthetic_QAs.json",
    gold_map_path: str = ".data/resources/gold_doc_ids.json",
    kw_db_path: str = ".data/resources/keywords_database.json",
    out_csv: str = ".artifacts/logs/rag_KW(2)_ablation_ON.csv",
    k: int = 5,
    limit: int | None = 100
):
    """
    Real evaluation using your FT_2RAG pipeline and files. Writes CSV and returns a summary.
    """
    # Defer heavy imports to keep module import cheap
    from llmrag.training.embedding.embedd_finetuning import load_model
    from llmrag.pipelines.ft_2rag.rag_main import run_rag_pipeline

    with open(gold_map_path, "r", encoding="utf-8") as f:
        GOLD_MAP: Dict[str, List[str]] = json.load(f)

    processed_dataset = prepare_rag_evaluation_data(file_path=eval_path)
    if limit:
        processed_dataset = processed_dataset[:limit]
    print(f"Total QA Pairs (limit={limit}): {len(processed_dataset)}")

    embedd_model = load_model(model_id=".models/emb_model", eval=True)

    csv_fields = [
        "question", "ground_truth", "generated_answer", "normalized_prediction",
        "em", "f1", "bleu", "rouge1", "rougeL", "embedding_similarity",
        "precision@5", "recall@5", "mrr", "ndcg@5", "map", "avg_rank"
    ]
    write_header = not os.path.exists(out_csv)
    results_list = []

    with open(out_csv, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()

        for i, qa_pair in enumerate(processed_dataset, start=1):
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            print(f"\n🔍 Q{i}: {question}\n🔑 Ground Truth: {ground_truth}")

            generated_answer, filtered_docs = run_rag_pipeline(query=question)

            retrieved_ids = [d.metadata.get("doc_id") for d in filtered_docs]
            sample_id = str(i - 1)
            gold_ids = set(GOLD_MAP.get(sample_id, []))

            # relevance vector for @k
            rel_vec = [(1 if rid in gold_ids else 0) for rid in (retrieved_ids[:k] if retrieved_ids else [])]
            rel_vec += [0] * max(0, k - len(rel_vec))

            precision_k = sum(rel_vec) / max(1, k)
            recall_k    = 1.0 if sum(rel_vec) > 0 else 0.0
            try:
                first_pos = rel_vec.index(1) + 1
                mrr       = 1.0 / first_pos
                avg_rank  = float(first_pos)
            except ValueError:
                mrr = 0.0
                avg_rank = float(k + 1)

            ndcg = float(ndcg_score([rel_vec], [rel_vec]))

            # Answer quality metrics
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
                "map":         _avg_precision(rel_vec),
                "avg_rank":    avg_rank
            })

            results_list.append(eval_results)
            writer.writerow(eval_results)

    # Aggregate summary
    all_metrics = ["em", "f1", "bleu", "rouge1", "rougeL", "embedding_similarity",
                   "precision@5", "recall@5", "mrr", "ndcg@5", "map", "avg_rank"]
    averages = {m: _safe_avg([d[m] for d in results_list]) for m in all_metrics}

    return {"ok": True, "out_csv": out_csv, "n": len(results_list), "averages": averages}

# -------------------- Smoke run --------------------
def smoke_kw_ablation():
    """
    File-free smoke: tiny similarity + simple retrieval metrics.
    Ensures code paths and dependencies are callable.
    """
    from sentence_transformers import SentenceTransformer
    sim = SentenceTransformer("all-MiniLM-L6-v2")

    q = "What is overfitting?"
    gold = "Overfitting is poor generalization."
    pred = "Overfitting happens when a model memorizes training data."

    ev = evaluate_rag(q, gold, pred, sim)

    # fake retrieval ids
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
    gold_ids  = {"doc_3", "doc_42"}
    k = 5
    rel_vec = [(1 if rid in gold_ids else 0) for rid in retrieved[:k]]
    precision_k = sum(rel_vec) / k
    recall_k    = 1.0 if sum(rel_vec) > 0 else 0.0

    try:
        first_pos = rel_vec.index(1) + 1
        mrr = 1.0 / first_pos
        avg_rank = float(first_pos)
    except ValueError:
        mrr = 0.0
        avg_rank = float(k + 1)

    return {
        "ok": True,
        "em": ev["em"],
        "f1": ev["f1"],
        "precision@5": precision_k,
        "recall@5": recall_k,
        "mrr": mrr,
        "avg_rank": avg_rank
    }

# -------------------- helpers --------------------
def _avg_precision(rel_vec: List[int]) -> float:
    hits = 0
    precisions = []
    for i, rel in enumerate(rel_vec, 1):
        if rel:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    # divide by number of relevant docs in the *gold* set if you want classic AP,
    # here we approximate with number of hits at k.
    return sum(precisions) / max(1, hits)

def _safe_avg(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / max(1, len(xs))

if __name__ == "__main__":
    print(run_kw_ablation())

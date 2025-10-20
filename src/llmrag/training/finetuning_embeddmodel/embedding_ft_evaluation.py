import json
from embedd_finetuning import load_model
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

# import matplotlib.pyplot as plt
# import seaborn as sns


# COMPARITIVE BASELINE WITH all-MiniLM-L6-v2

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import json

# # Load the baseline pre-trained model
# baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
# print("Baseline model loaded!")

# # Load the same questions and answers
# questions, answers = [], []
# with open("uniqueQA_dataset0.7.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line)
#         questions.append(data["question"])
#         answers.append(data["answer"])

# # Get embeddings using baseline model
# q_emb_baseline = baseline_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
# a_emb_baseline = baseline_model.encode(answers, convert_to_numpy=True, normalize_embeddings=True)

# # Compute cosine similarity matrix
# sim_matrix_baseline = cosine_similarity(q_emb_baseline, a_emb_baseline)

# # Diagonal = correct pairs; compute average similarity
# avg_diag_baseline = np.diag(sim_matrix_baseline).mean()
# print(f"\n🔹 Baseline Model Average Q-A Cosine Similarity: {avg_diag_baseline:.4f}")

# # Optional: Recall@1 for baseline
# top1 = np.argmax(sim_matrix_baseline, axis=1)
# recall_at_1_baseline = np.mean(top1 == np.arange(len(questions)))
# print(f"🔹 Baseline Model Recall@1: {recall_at_1_baseline:.4f}")



# Step 1: Load and structure the QA data

questions, answers = [], []

with open("uniqueQA_dataset0.7.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        questions.append(obj["question"])
        answers.append(obj["answer"])

# Step 2: Load finetuned embedding model

ft_model = load_model(model_id="finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0",
                      eval=True)

base_model = load_model(model_id="BAAI/bge-base-en-v1.5",
                      eval=True)

# Step 3: Compute Embeddings for Questions and Answers

print("Encoding questions...")
ft_question_embeddings = ft_model.encode(
    answers,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True,  # Optional but useful for cosine sim
)

print("Encoding questions...")
pt_question_embeddings = base_model.encode(
    answers,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True,  # Optional but useful for cosine sim
)

# print("Encoding answers...")
# answer_embeddings = ft_model.encode(
#     answers,
#     batch_size=64,
#     show_progress_bar=True,
#     convert_to_tensor=True,
#     normalize_embeddings=True,
# )

# # Step 1: Compute cosine similarity between questions and answers
# sim_matrix = cosine_similarity(question_embeddings.cpu(), answer_embeddings.cpu())

# # Step 2: Sort indices of answers by similarity for each question
# sorted_indices = np.argsort(-sim_matrix, axis=1)  # Descending order

# # # Step 3: Define helper metrics
# # def recall_at_k(sorted_indices, k):
# #     hits = 0
# #     for i, row in enumerate(sorted_indices):
# #         if i in row[:k]:
# #             hits += 1
# #     return hits / len(sorted_indices)

# # def mean_reciprocal_rank(sorted_indices):
# #     rr_total = 0
# #     for i, row in enumerate(sorted_indices):
# #         if i in row:
# #             rank = np.where(row == i)[0][0] + 1
# #             rr_total += 1 / rank
# #     return rr_total / len(sorted_indices)

# # def ndcg_at_k(sorted_indices, k):
# #     ndcg_total = 0
# #     for i, row in enumerate(sorted_indices):
# #         if i in row[:k]:
# #             rank = np.where(row[:k] == i)[0][0] + 1
# #             ndcg_total += 1 / np.log2(rank + 1)
# #     return ndcg_total / len(sorted_indices)

# # # Step 4: Report metrics
# # print("\n--- Embedding Evaluation Metrics ---")
# # for k in [1, 3, 5]:
# #     recall = recall_at_k(sorted_indices, k)
# #     ndcg = ndcg_at_k(sorted_indices, k)
# #     print(f"Recall@{k}: {recall:.4f}")
# #     print(f"NDCG@{k} : {ndcg:.4f}")

# # mrr = mean_reciprocal_rank(sorted_indices)
# # print(f"MRR      : {mrr:.4f}")

# # Convert similarity matrix to numpy if it's still on GPU
# # sim_matrix_np = sim_matrix
# # subset = sim_matrix_np[:100, :100]  # Only top 100 Q/A

# # plt.figure(figsize=(10, 8))
# # sns.heatmap(subset, cmap='viridis', xticklabels=False, yticklabels=False)
# # plt.title("Cosine Similarity Matrix (Questions vs Answers)")
# # plt.xlabel("Answers")
# # plt.ylabel("Questions")
# # plt.show()

# # === Diagonal similarity: similarity of correct (question, answer) pairs ===
# diagonal_scores = np.diag(sim_matrix)  # Gets diagonal elements from numpy array
# avg_diag_sim = diagonal_scores.mean()
# print(f"Average Cosine Similarity (Q-A pairs): {avg_diag_sim:.4f}")

# # === Get top-1 predicted answer index for each question ===
# top_pred_indices = np.argmax(sim_matrix, axis=1)

# # # === Load your dataset ===
# # qa_pairs = []
# # with open("uniqueQA_dataset0.7.jsonl", "r", encoding="utf-8") as f:
# #     for line in f:
# #         data = json.loads(line)
# #         qa_pairs.append((data["question"], data["answer"]))

# # # === Inspect a few mismatched samples ===
# # print("\n--- Misaligned Pairs ---")
# # for i in range(len(qa_pairs)):
# #     if top_pred_indices[i] != i:
# #         q, gt_a = qa_pairs[i]
# #         pred_a = qa_pairs[top_pred_indices[i]][1]
# #         print(f"\nQuestion {i+1}: {q}")
# #         print(f"→ Ground Truth Answer: {gt_a}")
# #         print(f"→ Predicted Top Answer: {pred_a}")
# #         print(f"→ Cosine Similarity (GT): {sim_matrix[i, i]:.4f}")
# #         print(f"→ Cosine Similarity (Pred): {sim_matrix[i, top_pred_indices[i]]:.4f}")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def plot_embeddings(embeddings, labels=None, title="Embedding Visualization", method="pca"):
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    else:
        raise ValueError("Choose 'pca' or 'tsne'.")

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.get_cmap('tab10', len(unique_labels))
        for i, label in enumerate(unique_labels):
            idx = [j for j, l in enumerate(labels) if l == label]
            plt.scatter(reduced[idx, 0], reduced[idx, 1], label=label, alpha=0.6, color=colors(i))
        plt.legend()
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# embeddings_pretrained: np.array of shape (N, D)
# embeddings_finetuned: np.array of shape (N, D)
# domain_labels: optional list like ["AI", "RL", "AI", "CV", ...]

ft_question_embeddings_np = ft_question_embeddings.cpu().numpy()
pt_question_embeddings_np = pt_question_embeddings.cpu().numpy()

plot_embeddings(pt_question_embeddings_np, title="Pretrained Embeddings (PCA)", method="pca")
plot_embeddings(ft_question_embeddings_np, title="Fine-tuned Embeddings (PCA)", method="pca")

# You can also try t-SNE (takes longer to compute)
plot_embeddings(pt_question_embeddings_np, title="Pretrained Embeddings (t-SNE)", method="tsne")
plot_embeddings(ft_question_embeddings_np, title="Fine-tuned Embeddings (t-SNE)", method="tsne")

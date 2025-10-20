import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .eval_dataset import prepare_rag_evaluation_data
from finetuning_embeddModel.embedd_finetuning import load_model
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from FT_2RAG.vectorstore import generate_embeddings, LocalEmbeddingFunction, create_vecdb, similarity_score

import logging
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

eval_dataset_path = "synthetic_QAs.json"
csv_log_path = "rag_eval_results.csv"

# Load evaluation dataset
processed_dataset = prepare_rag_evaluation_data(file_path=eval_dataset_path)
print(f"Total QA Pairs: {len(processed_dataset)}")

# Load embedding model
embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
model_id = "BAAI/bge-base-en-v1.5"
embedd_model = load_model(model_id=model_id, eval=True)

emf = LocalEmbeddingFunction(embedd_model=embedd_model,
                            )

# Load vector DB from existing directory
db_path = "./FT_1RAG/vecDB/textbooks_v2"  # Replace with your actual path
vectorstore = Chroma(persist_directory=db_path, embedding_function=emf)

def compute_semantic_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def evaluate_retrieval(qa_pairs, vectorstore, embedding_model, top_k_list=[1, 3, 5]):
    results = {k: {"recall": [], "mrr": []} for k in top_k_list}

    for qa in tqdm(qa_pairs, desc="Evaluating retrieval"):
        query = qa["question"]
        ground_truth_answer = qa["answer"]
        print(f"Question: {query}")
        print(f"Answer: {ground_truth_answer}")

        # Embed the ground truth answer
        gt_embedding = np.array(embedding_model.encode([ground_truth_answer])[0])

        for k in top_k_list:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(query)
            print(f"Retrieved Context@{k}: {retrieved_docs}")
            found = False
            reciprocal_rank = 0

            for idx, doc in enumerate(retrieved_docs):
                doc_embedding = np.array(embedding_model.encode([doc.page_content])[0])
                sim = compute_semantic_similarity(gt_embedding, doc_embedding)

                if sim > 0.6 and not found:  # Adjustable threshold
                    results[k]["recall"].append(1)
                    reciprocal_rank = 1 / (idx + 1)
                    found = True
                    break

            if not found:
                results[k]["recall"].append(0)

            results[k]["mrr"].append(reciprocal_rank)

            # print(f"Results @{k}: {results}")

    final_scores = {
        k: {
            "Recall@{}".format(k): np.mean(results[k]["recall"]),
            "MRR@{}".format(k): np.mean(results[k]["mrr"])
        } for k in top_k_list
    }

    print(final_scores)

evaluate_retrieval(qa_pairs=processed_dataset,
                   vectorstore=vectorstore,
                   embedding_model=embedd_model,)

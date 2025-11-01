from evaluation import generate_evaluator
from data_loading import loading_dataset
from embedd_finetuning import load_model

train_dataset, test_dataset, queries, corpus, relevant_docs = loading_dataset(test_path="./finetuning_embeddModel/test_dataset.json",
                train_path="./finetuning_embeddModel/train_dataset.json",
                )

evaluator = generate_evaluator(queries=queries,
                   corpus=corpus,
                   relevant_docs=relevant_docs,
                   )

model_id = "BAAI/bge-base-en-v1.5"

base_model = load_model(model_id=model_id, eval=True)

baseline_results = evaluator(base_model)
print("Baseline Model")
print(f"dim_512_cosine_ndcg@10: ", baseline_results["dim_512_cosine_ndcg@10"])

ft_model = load_model(model_id="finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0",
                      eval=True)

ft_results = evaluator(ft_model)
print("Finetuned Model")
print(f"dim_512_cosine_ndcg@10: ", ft_results["dim_512_cosine_ndcg@10"])
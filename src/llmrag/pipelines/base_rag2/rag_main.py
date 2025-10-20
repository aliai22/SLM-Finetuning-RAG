from LLM_Model import load_local_llm
from .finetuned_model import load_ft_model
from .dataset import load_dataset, preprocess_dataset, batchify
from .vectorstore import generate_embeddings, LocalEmbeddingFunction, create_vecdb, similarity_score
from .chatbot import rag_chatbot, generate_response
from finetuning_embeddModel.embedd_finetuning import load_model

# from langchain.evaluation import load_evaluator
# from sentence_transformers import SentenceTransformer

import os

def run_rag_pipeline(query=None):

    """
    Executes RAG pipeline using finetuned LLM and QA-based vector database.
    Args:
        query (str): User question to run through RAG. If None, uses default.
    Returns:
        response (str): Generated answer.
        context (list): Retrieved documents.
    """

    llm_path = "./LLMs"

    base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)
    print("Local Base Model Loaded Successfully!")

    # ft_ckpt = "./finetuning_textbooks/finetuning_checkpoints/final_checkpoint/checkpoint-10000"


    # ft_model = load_ft_model(base_model=base_model,
    #                         ft_ckpt=ft_ckpt)
    # print("Finetuned Model Loaded Successfully!")

    embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
    embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    dataset_path = "uniqueQA_dataset0.7.jsonl"
    dataset = load_dataset(dataset_path)
    proc_dataset = preprocess_dataset(data=dataset)

    print(f"Length of Dataset: {len(proc_dataset)}")

    # batch_size = 64  # Choose a suitable batch size
    # all_embeddings = []

    # for batch in batchify(dataset, batch_size):
    #     embeddings = generate_embeddings(ft_model, eval_tokenizer, batch)
    #     all_embeddings.extend(embeddings)  # Append embeddings from the batch

    # create_new=True

    db_path = "./FT_2RAG/vecDB/QAs_v2"
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    emf = LocalEmbeddingFunction(embedd_model=embedd_model
                                )

    vec_DB = create_vecdb(path=db_path,
                dataset=proc_dataset,
                embedding_function=emf,
                create_new=False)

    if query == None:
        query = "what is machine learning?"

    # context = [
    #     'Q: What were the differences in the type of care delineated in the reviewed studies?\nA: Variations in the type of care were observed in the reviewed studies, including predicting drug response, diagnosing neoplasms, and personalizing treatment regimens based on genomic and functional data.',
    #     'Q: What variations characterized the type of care across the studies reviewed?\nA: Variability in the type of care was evident across the studies reviewed, involving predicting drug response, accurate neoplasm diagnosis, and personalized treatment based on genomic and functional data.',
    #     'Q: How did the type of care differ in the studies included in the review?\nA: Variability in the type of care was evident across the reviewed studies, ranging from predicting drug response, diagnosing neoplasms, to customizing treatments based on genomic and functional data.'
    #     ]

    # model = SentenceTransformer("all-MiniLM-L6-v2")

    # for cont in context:
    #     embeddings1=model.encode(query)
    #     embeddings2=model.encode(cont)
    #     sim_score = model.similarity(embeddings1, embeddings2)
    #     print(query)
    #     print(cont)
    #     print(sim_score)

    # for cont in context:
    #     sim_score = similarity_score(text1=query,
    #                      text2=cont,
    #                      model=ft_model,
    #                      tokenizer=eval_tokenizer)
    #     print(query)
    #     print(cont)
    #     print(sim_score)
    #     # if sim_score

    # evaluator = load_evaluator("embedding_distance",
    #                            emebeddings=emf)

    # for cont in context:
    #     print(evaluator.evaluate_strings(prediction=cont, reference=query))

    print(f"\nUser:\n{query}")

    response, context = rag_chatbot(user_query=query,
                vec_db=vec_DB,
                model=base_model,
                tokenizer=eval_tokenizer)

    print(f"\nChatbot:\n{response}")

    return response, context

if __name__ == "__main__":
    response, context = run_rag_pipeline(query=None)
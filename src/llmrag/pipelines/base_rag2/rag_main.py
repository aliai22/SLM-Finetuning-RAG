from llmrag.training.slm.qa_pairs.LLM_Model import load_local_llm
from .finetuned_model import load_ft_model
from .dataset import load_dataset, preprocess_dataset, batchify
from .vectorstore import generate_embeddings, LocalEmbeddingFunction, create_vecdb, similarity_score
from .chatbot import rag_chatbot
from llmrag.training.embedding.embedd_finetuning import load_model
import os

def run_rag_pipeline(
    query=None,
    vectorstore=None,   # NEW (optional) – injected in smoke
    embedder=None,      # NEW (optional) – injected in smoke
    generator=None,     # NEW (optional) – injected (model, tokenizer)
    top_k=3             # NEW (optional) – plumb to chatbot if you add k later
):
    """
    Executes RAG pipeline using finetuned LLM and QA-based vector database.
    In normal runs, nothing changes. In smokes, injected components skip heavy setup.
    """
    llm_path = ".models/base"

    # generator: (model, tokenizer) injected or load local
    if generator is not None:
        base_model, tokenizer = generator
        eval_tokenizer = tokenizer
        print("Injected generator loaded (smoke mode).")
    else:
        base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path, base_model_id="microsoft/phi-2")
        print("Local Base Model Loaded Successfully!")

    # embedder: injected or load local checkpoint
    if embedder is not None:
        embedd_model = embedder
        print("Injected embedder loaded (smoke mode).")
    else:
        embedd_ftmodel_ckpt = ".models/emb_model"
        embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    # vector store: injected or build/load persisted Chroma
    if vectorstore is not None:
        vec_DB = vectorstore
        print("Injected vector store loaded (smoke mode).")
    else:
        dataset_path = ".data/finetuning/qa/uniqueQA_dataset0.7.jsonl"
        dataset = load_dataset(dataset_path)
        proc_dataset = preprocess_dataset(data=dataset)
        print(f"Length of Dataset: {len(proc_dataset)}")

        db_path = "./FT_2RAG/vecDB/QAs_v2"
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        emf = LocalEmbeddingFunction(embedd_model=embedd_model)
        vec_DB = create_vecdb(
            path=db_path,
            dataset=proc_dataset,
            embedding_function=emf,
            create_new=False
        )

    if query is None:
        query = "what is machine learning?"

    print(f"\nUser:\n{query}")

    response, context = rag_chatbot(
        user_query=query,
        vec_db=vec_DB,
        model=base_model,
        tokenizer=eval_tokenizer
    )

    print(f"\nChatbot:\n{response}")
    return response, context

if __name__ == "__main__":
    response, context = run_rag_pipeline(query=None)

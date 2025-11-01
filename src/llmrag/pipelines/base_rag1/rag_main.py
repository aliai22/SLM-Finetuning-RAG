from llmrag.training.slm.qa_pairs.LLM_Model import load_local_llm
from .finetuned_model import load_ft_model
from .dataset import load_dataset, preprocess_dataset, batchify
from .vectorstore import generate_embeddings, LocalEmbeddingFunction, create_vecdb, similarity_score, process_pdf_to_vecdb, extract_text_by_page
from .chatbot import rag_chatbot, generate_response
from llmrag.training.embedding.embedd_finetuning import load_model
import json
from glob import glob
import os  # <-- ensure this is imported

def run_rag_pipeline(
    query=None,
    vectorstore=None,          # NEW (optional): inject a ready-made vector store
    embedder=None,             # NEW (optional): inject a ready-made embedding model
    generator=None,            # NEW (optional): inject (base_model, tokenizer)
):
    llm_path = ".models/base"

    # --- (A) Generator: use injected (model, tokenizer) if provided, else your original loader
    if generator is not None:
        base_model, tokenizer = generator  # eval_tokenizer not needed for chat
        eval_tokenizer = tokenizer
        print("Injected generator loaded (smoke mode).")
    else:
        base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path, base_model_id="microsoft/phi-2")
        print("Local Base Model Loaded Successfully!")

    # --- (B) Embedding model: use injected if provided, else your original loader
    if embedder is not None:
        embedd_model = embedder
        print("Injected embedder loaded (smoke mode).")
    else:
        embedd_ftmodel_ckpt = ".models/emb_model"
        embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    # --- (C) Vector store: use injected if provided, else your original path
    if vectorstore is not None:
        vec_DB = vectorstore
        print("Injected vector store loaded (smoke mode).")
    else:
        # Original heavy path (left intact)
        with open(".configs/ingest/pdfs_config.json") as f:
            config = json.load(f)

        db_path = "./FT_1RAG/vecDB/textbooks_v2"
        if not os.path.exists(db_path):
            os.makedirs(db_path)

        emf = LocalEmbeddingFunction(embedd_model=embedd_model)

        create_new = False
        all_pages_text = []
        if create_new:
            data_files = glob("AIbooks_dataset/AI-books/*")
            for i, file_path in enumerate(sorted(data_files)):
                extracted = extract_text_by_page(pdf_path=file_path, pdf_config=config, config_index=i)
                all_pages_text.extend(extracted)
        
        vec_DB = process_pdf_to_vecdb(
            pdf_text_by_page=all_pages_text,
            db_path=db_path,
            embedding_function=emf,
            batch_size=32,
            create_new=create_new,
        )

    if query is None:
        query = "Explain the difference between classificaition and regression."

    print(f"\nUser:\n{query}")
    
    response, context = rag_chatbot(
        user_query=query,
        vec_db=vec_DB,
        model=base_model,
        tokenizer=tokenizer
    )

    print(f"\nChatbot:\n{response}")
    return response, context

if __name__ == "__main__":
    response, context = run_rag_pipeline(query=None)

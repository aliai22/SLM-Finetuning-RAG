from LLM_Model import load_local_llm
from FT_1RAG.finetuned_model import load_ft_model
from FT_1RAG.dataset import load_dataset, preprocess_dataset, batchify
from FT_1RAG.vectorstore import generate_embeddings, LocalEmbeddingFunction, create_vecdb, similarity_score, process_pdf_to_vecdb, extract_text_by_page
from FT_1RAG.chatbot import rag_chatbot, generate_response
from finetuning_embeddModel.embedd_finetuning import load_model
import json
from glob import glob
from sentence_transformers import SentenceTransformer

# from langchain.evaluation import load_evaluator
# from sentence_transformers import SentenceTransformer

import os

# NEW: import prompt templates for sensitivity mode
from rag_eval_FT1.sensitivity.prompts import PROMPTS, DEFAULT_PROMPT_ID

KW_MODEL = SentenceTransformer("all-MiniLM-L12-v2")    # fast model for extractor
TOP_K_DENSE   = 20
TOP_K_CONTEXT = 5

def extract_KW_from_query(
    kw_model,
    doc:str,
    top_n:int,
    mmr_diversity:int
):
    keyphrases = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        use_mmr=True,
        diversity=mmr_diversity,
        top_n=top_n
    )
    
    
    # print(keyphrases)
    filtered_keywords = [kw for kw, score in keyphrases if score > 0.6]
    return filtered_keywords

KW_DB_PATH = "keywords_database.json"

with open(KW_DB_PATH, "r", encoding="utf-8") as f:
    keyword_db = json.load(f)          # {doc_id: [kw1, kw2, ...]}

# def run_rag_pipeline(query=None):

#     llm_path = "./LLMs"

#     base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)
#     print("Local Base Model Loaded Successfully!")

#     ft_ckpt = "./Finetuning_Checkpoints_filtered/final-checkpoint/checkpoint-9750"

#     ft_model = load_ft_model(base_model=base_model,
#                             ft_ckpt=ft_ckpt)
#     # print("Finetuned Model Loaded Successfully!")

#     embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
#     embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

#     # dataset_path = "./AIbooks_dataset/text_finetuningData.jsonl"
#     # dataset, metadata = load_dataset(dataset_path)
#     # proc_dataset = preprocess_dataset(data=dataset, metadata_list=metadata)
#     # print(f"Sample Dataset: {proc_dataset[0]}")
#     # print(f"Length of Dataset: {len(proc_dataset)}")

#     # batch_size = 64  # Choose a suitable batch size
#     # all_embeddings = []

#     # for batch in batchify(dataset, batch_size):
#     #     embeddings = generate_embeddings(ft_model, eval_tokenizer, batch)
#     #     all_embeddings.extend(embeddings)  # Append embeddings from the batch

#     # create_new=True

#     with open("AIbooks_dataset/pdfs_config.json") as f:
#         config = json.load(f)

#     db_path = "./FT_1RAG/vecDB/textbooks_v2"
#     # db_path = "./FT_2RAG/vecDB/QAs_v2"
#     if not os.path.exists(db_path):
#         os.makedirs(db_path)

#     emf = LocalEmbeddingFunction(embedd_model=embedd_model,
#                                 )

#     create_new=False
#     all_pages_text = []
#     if create_new:
#         data_files = glob("AIbooks_dataset/AI-books/*")
#         for i, file_path in enumerate(sorted(data_files)):
#             extracted = extract_text_by_page(pdf_path=file_path, pdf_config=config, config_index=i)
#             all_pages_text.extend(extracted)
    
#     vec_DB = process_pdf_to_vecdb(pdf_text_by_page=all_pages_text,
#                          db_path=db_path,
#                          embedding_function=emf,
#                          batch_size=32,
#                          create_new=create_new)

#     # vec_DB = create_vecdb(path=db_path,
#     #             dataset=proc_dataset,
#     #             embedding_function=emf,
#     #             # create_new=False
#     #             )

#     if query == None:
#         query = "What are conv nets and why they are used?"

#     # query = "What is support vector machine (SVM)?"
#     # context = [
#     #     'Q: What were the differences in the type of care delineated in the reviewed studies?\nA: Variations in the type of care were observed in the reviewed studies, including predicting drug response, diagnosing neoplasms, and personalizing treatment regimens based on genomic and functional data.',
#     #     'Q: What variations characterized the type of care across the studies reviewed?\nA: Variability in the type of care was evident across the studies reviewed, involving predicting drug response, accurate neoplasm diagnosis, and personalized treatment based on genomic and functional data.',
#     #     'Q: How did the type of care differ in the studies included in the review?\nA: Variability in the type of care was evident across the reviewed studies, ranging from predicting drug response, diagnosing neoplasms, to customizing treatments based on genomic and functional data.'
#     #     ]

#     # model = SentenceTransformer("all-MiniLM-L6-v2")

#     # for cont in context:
#     #     embeddings1=model.encode(query)
#     #     embeddings2=model.encode(cont)
#     #     sim_score = model.similarity(embeddings1, embeddings2)
#     #     print(query)
#     #     print(cont)
#     #     print(sim_score)

#     # for cont in context:
#     #     sim_score = similarity_score(text1=query,
#     #                      text2=cont,
#     #                      model=ft_model,
#     #                      tokenizer=eval_tokenizer)
#     #     print(query)
#     #     print(cont)
#     #     print(sim_score)
#     #     # if sim_score

#     # evaluator = load_evaluator("embedding_distance",
#     #                            emebeddings=emf)

#     # for cont in context:
#     #     print(evaluator.evaluate_strings(prediction=cont, reference=query))

#     print(f"\nUser:\n{query}")
 
#     response, context = rag_chatbot(user_query=query,
#                 vec_db=vec_DB,
#                 model=ft_model,
#                 tokenizer=eval_tokenizer)

#     print(f"\nChatbot:\n{response}")

#     return response, context

# NEW SENSITIVITY STUFF

def _ensure_vecdb(embedd_model, db_path="./FT_1RAG/vecDB/textbooks_v2", create_new=False,
                  pdf_glob="AIbooks_dataset/AI-books/*", pdf_cfg="AIbooks_dataset/pdfs_config.json",
                  batch_size=32):
    """
    Builds or loads your Chroma vector DB exactly like before.
    """
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    emf = LocalEmbeddingFunction(embedd_model=embedd_model)

    all_pages_text = []
    if create_new:
        with open(pdf_cfg) as f:
            config = json.load(f)
        data_files = glob(pdf_glob)
        for i, file_path in enumerate(sorted(data_files)):
            # Assumes you have extract_text_by_page somewhere earlier in your project
            extracted = extract_text_by_page(pdf_path=file_path, pdf_config=config, config_index=i)
            all_pages_text.extend(extracted)

    vec_DB = process_pdf_to_vecdb(
        pdf_text_by_page=all_pages_text,
        db_path=db_path,
        embedding_function=emf,
        batch_size=batch_size,
        create_new=create_new
    )
    return vec_DB, emf

def _retrieve_context(vec_DB, query: str, k: int = 5, order: str = "as_is"):
    """
    Retrieves top-k docs from your existing Chroma store and returns (ctx_text, ctx_ids, docs).
    """
    retriever = vec_DB.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(query)

    # Optional shuffle (kept for completeness; you said we'll skip retrieval sensitivity)
    if order == "shuffled":
        import random
        random.shuffle(docs)

    # Join page contents into one context block
    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])

    # Try to extract stable IDs if present in metadata; otherwise fall back to index
    ctx_ids = []
    for idx, d in enumerate(docs):
        mid = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            mid = d.metadata.get("source_id") or d.metadata.get("source") or d.metadata.get("id")
        ctx_ids.append(mid if mid is not None else str(idx))
    return context_text, ctx_ids, docs

def _build_prompt(prompt_id: str, context_text: str, question: str) -> str:
    """
    Format the prompt using our sensitivity templates; fallback to DEFAULT if unknown id.
    """
    pid = prompt_id if (prompt_id in PROMPTS) else DEFAULT_PROMPT_ID
    return PROMPTS[pid].format(context=context_text, question=question), pid

def _generate_with_model(model, tokenizer, prompt: str, temperature: float = 0.2, max_new_tokens: int = 256):
    """
    Simple HF-style generation using your already-loaded FT model & tokenizer.
    """
    import torch
    model.eval()
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Do-sample for temperature > 0, greedy when temperature ~ 0
    do_sample = (temperature is not None) and (temperature > 0.0)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature) if temperature is not None else 0.0,
        eos_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated portion if you prefer; here we decode all
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # If your tokenizer includes the prompt in the decode, you can strip it:
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

def run_rag_pipeline(query: str = None,
                     prompt_id: str = None,
                     top_k: int = 5,
                     ctx_order: str = "as_is",
                     temperature: float = 0.2):
    """
    Backward compatible RAG entry-point:
    - Old behavior: call rag_chatbot(...) if prompt_id is None (no sensitivity controls).
    - Sensitivity mode: if prompt_id is provided (or you want explicit control), perform:
        load LLM -> load embedder -> ensure vecDB -> retrieve k -> build PROMPT -> generate.
    Returns: (answer_text, meta_dict)
    """
    # --- Defaults and safety ---
    if query is None:
        query = "What are conv nets and why they are used?"

    # --- Load local base+FT model and embedder (as before) ---
    llm_path = "./LLMs"
    base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)
    print("Local Base Model Loaded Successfully!")

    ft_ckpt = "./Finetuning_Checkpoints_filtered/final-checkpoint/checkpoint-9750"
    ft_model = load_ft_model(base_model=base_model, ft_ckpt=ft_ckpt)

    embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
    embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    # --- Build/load vecDB (same path you use) ---
    db_path = "./FT_1RAG/vecDB/textbooks_v2"
    vec_DB, _ = _ensure_vecdb(embedd_model=embedd_model, db_path=db_path, create_new=False)

    # === BRANCH 1: Backward-compatible path (no prompt_id provided) ===
    if prompt_id is None:
        print(f"\nUser:\n{query}")
        response, context = rag_chatbot(
            user_query=query,
            vec_db=vec_DB,
            model=ft_model,
            tokenizer=eval_tokenizer
        )
        print(f"\nChatbot:\n{response}")
        # context may already contain IDs; if not, standardize meta
        meta = context if isinstance(context, dict) else {}
        meta.setdefault("prompt_id", "legacy_default")
        meta.setdefault("top_k", None)
        meta.setdefault("ctx_order", "as_is")
        return response, meta

    # === BRANCH 2: Sensitivity path (prompt-aware, controlled) ===
    # 1) Retrieve context @ top_k
    context_text, ctx_ids, _docs = _retrieve_context(vec_DB, query, k=top_k, order=ctx_order)

    # 2) Build prompt from templates
    prompt, resolved_pid = _build_prompt(prompt_id, context_text, query)

    # 3) Generate with explicit temperature
    answer_text = _generate_with_model(ft_model, eval_tokenizer, prompt, temperature=temperature)

    # 4) Meta for Evidence Stability (if you later want it) and logging
    meta = {
        "ctx_ids": ctx_ids,
        "prompt_id": resolved_pid,
        "top_k": top_k,
        "ctx_order": ctx_order
    }

    return answer_text, meta

if __name__ == "__main__":
    response, context = run_rag_pipeline(query=None)
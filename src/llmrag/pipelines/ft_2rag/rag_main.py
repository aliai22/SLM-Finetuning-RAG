from LLM_Model import load_local_llm
from FT_2RAG.finetuned_model import load_ft_model
from FT_2RAG.dataset import load_dataset, preprocess_dataset
from FT_2RAG.vectorstore import LocalEmbeddingFunction, create_vecdb
from FT_2RAG.chatbot import rag_chatbot, generate_response
from finetuning_embeddModel.embedd_finetuning import load_model

import os, json
from rag_eval_FT2.sensitivity.prompts import PROMPTS, DEFAULT_PROMPT_ID

def _ensure_vecdb_QA(embedd_model,
                     db_path:str="./FT_2RAG/vecDB/QAs_v3",
                     dataset_path:str="uniqueQA_dataset0.7.jsonl",
                     create_new:bool=False):
    """
    Builds/loads the QA-based vector DB exactly as your current FT2 pipeline does.
    """
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    dataset = load_dataset(dataset_path)
    proc_dataset = preprocess_dataset(data=dataset)

    emf = LocalEmbeddingFunction(embedd_model=embedd_model)
    vec_DB = create_vecdb(
        path=db_path,
        dataset=proc_dataset,
        embedding_function=emf,
        create_new=create_new
    )
    return vec_DB, emf

def _retrieve_context(vec_DB, query:str, k:int=5, order:str="as_is"):
    """
    Retrieve top-k docs from the QA vector store and return (context_text, ctx_ids, docs).
    Works with Chroma/LC retriever used in FT_2RAG.
    """
    retriever = vec_DB.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if order == "shuffled":
        import random
        random.shuffle(docs)

    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])

    # Pull a stable id if present; fall back to index
    ctx_ids = []
    for idx, d in enumerate(docs):
        mid = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            # FT2 QA store commonly uses "doc_id" — keep other fallbacks too
            mid = d.metadata.get("doc_id") or d.metadata.get("source_id") or d.metadata.get("source") or d.metadata.get("id")
        ctx_ids.append(mid if mid is not None else str(idx))

    return context_text, ctx_ids, docs

def _build_prompt(prompt_id:str, context_text:str, question:str):
    pid = prompt_id if (prompt_id in PROMPTS) else DEFAULT_PROMPT_ID
    return PROMPTS[pid].format(context=context_text, question=question), pid

def run_rag_pipeline(query:str=None,
                     prompt_id:str=None,   # <-- pass a value to enable sensitivity mode
                     top_k:int=5,
                     ctx_order:str="as_is",
                     temperature:float=0.1,
                     max_new_tokens:int=256):
    """
    FT2 RAG with QA vecDB.
    - Legacy mode (prompt_id=None): uses your original rag_chatbot(...) and returns (response, context_docs).
    - Sensitivity mode (prompt_id provided): retrieves top_k QA docs, formats template, and
      generates with your finetuned model using the given temperature. Returns (response, docs).

    Returns:
        response (str), context_docs (list of retrieved docs)
    """
    if query is None:
        query = "what is machine learning?"

    # Load LLMs (same as your current code)
    llm_path = "./LLMs"
    base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)
    print("Local Base Model Loaded Successfully!")

    ft_ckpt = "./Finetuning_Checkpoints_filtered/final-checkpoint/checkpoint-9750"
    ft_model = load_ft_model(base_model=base_model, ft_ckpt=ft_ckpt)

    # Embedding model for vecDB
    embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
    embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    # Ensure QA-based vecDB (no PDFs)
    db_path = "./FT_2RAG/vecDB/QAs_v3"
    vec_DB, _ = _ensure_vecdb_QA(embedd_model=embedd_model,
                                 db_path=db_path,
                                 dataset_path="uniqueQA_dataset0.7.jsonl",
                                 create_new=False)

    # === Legacy path (backward-compatible): no prompt control ===
    if prompt_id is None:
        print(f"\nUser:\n{query}")
        # Keep your original behavior exactly (note: you previously used base_model here)
        response, context_docs = rag_chatbot(
            user_query=query,
            vec_db=vec_DB,
            model=base_model,          # unchanged to avoid breaking any heuristics in rag_chatbot
            tokenizer=eval_tokenizer
        )
        print(f"\nChatbot:\n{response}")
        return response, context_docs

    # === Sensitivity path (prompt-aware, controlled) ===
    # 1) Retrieve QA context
    context_text, ctx_ids, docs = _retrieve_context(vec_DB, query, k=top_k, order=ctx_order)

    # 2) Build prompt from templates
    prompt, resolved_pid = _build_prompt(prompt_id, context_text, query)

    # 3) Generate with your existing helper (preferred) or fallback to a simple generate
    try:
        # If your generate_response supports temperature & max_new_tokens, pass them; else it will ignore extras.
        response = generate_response(
            model=ft_model,
            tokenizer=eval_tokenizer,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
    except TypeError:
        # Fallback: minimal HF-style generation if your generate_response signature differs
        import torch
        ft_model.eval()
        device = next(ft_model.parameters()).device if hasattr(ft_model, "parameters") else "cpu"
        inputs = eval_tokenizer(prompt, return_tensors="pt"); inputs = {k:v.to(device) for k,v in inputs.items()}
        do_sample = (temperature is not None) and (float(temperature) > 0.0)
        out_ids = ft_model.generate(**inputs,
                                    do_sample=do_sample,
                                    temperature=float(temperature) if temperature else 0.0,
                                    max_new_tokens=max_new_tokens,
                                    eos_token_id=eval_tokenizer.eos_token_id)
        response = eval_tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if response.startswith(prompt): response = response[len(prompt):].strip()

    # 4) Return answer and the retrieved docs list (keeps KW_ablation-friendly shape if you ever reuse it)
    return response, docs

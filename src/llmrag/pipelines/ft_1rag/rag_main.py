from llmrag.training.slm.qa_pairs.LLM_Model import load_local_llm
from llmrag.pipelines.ft_1rag.finetuned_model import load_ft_model
from llmrag.pipelines.ft_1rag.vectorstore import (
    LocalEmbeddingFunction, create_vecdb, process_pdf_to_vecdb, extract_text_by_page
)
from llmrag.pipelines.ft_1rag.chatbot import rag_chatbot
from llmrag.training.embedding.embedd_finetuning import load_model

import json
from glob import glob
import os

# NEW: prompt sensitivity templates (kept from your code)
from llmrag.eval.common.sensitivity.prompts import PROMPTS, DEFAULT_PROMPT_ID

def _ensure_vecdb(
    embedd_model,
    db_path="./FT_1RAG/vecDB/textbooks_v2",
    create_new=False,
    pdf_glob="AIbooks_dataset/AI-books/*",
    pdf_cfg=".configs/ingest/pdfs_config.json",
    batch_size=32,
):
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    emf = LocalEmbeddingFunction(embedd_model=embedd_model)
    all_pages_text = []
    if create_new:
        with open(pdf_cfg) as f:
            config = json.load(f)
        data_files = glob(pdf_glob)
        for i, file_path in enumerate(sorted(data_files)):
            extracted = extract_text_by_page(pdf_path=file_path, pdf_config=config, config_index=i)
            all_pages_text.extend(extracted)
    vec_DB = process_pdf_to_vecdb(
        pdf_text_by_page=all_pages_text,
        db_path=db_path,
        embedding_function=emf,
        batch_size=batch_size,
        create_new=create_new,
    )
    return vec_DB, emf

def _retrieve_context(vec_DB, query: str, k: int = 5, order: str = "as_is"):
    if hasattr(vec_DB, "as_retriever"):
        retriever = vec_DB.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(query)
    elif hasattr(vec_DB, "similarity_search"):
        docs = vec_DB.similarity_search(query, k=k)
    else:
        # last resort
        docs = []

    if order == "shuffled" and docs:
        import random
        random.shuffle(docs)

    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
    ctx_ids = []
    for idx, d in enumerate(docs):
        mid = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            mid = d.metadata.get("source_id") or d.metadata.get("source") or d.metadata.get("id")
        ctx_ids.append(mid if mid is not None else str(idx))
    return context_text, ctx_ids, docs

def _build_prompt(prompt_id: str, context_text: str, question: str) -> str:
    pid = prompt_id if (prompt_id in PROMPTS) else DEFAULT_PROMPT_ID
    return PROMPTS[pid].format(context=context_text, question=question), pid

def _generate_with_model(model, tokenizer, prompt: str, temperature: float = 0.2, max_new_tokens: int = 256):
    import torch
    model.eval()
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = (temperature is not None) and (temperature > 0.0)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=float(temperature) if temperature is not None else 0.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

def run_rag_pipeline(
    query: str = None,
    prompt_id: str = None,
    top_k: int = 5,
    ctx_order: str = "as_is",
    temperature: float = 0.2,
    *,
    # ---- NEW optional injections for smoke ----
    vectorstore=None,
    embedder=None,
    generator=None
):
    """
    Backward compatible RAG entry-point.
    - Normal use: loads your models/DB as before.
    - Smoke use: accepts injected (vectorstore, embedder, generator=(model, tokenizer)).
    Returns: (answer_text, meta_dict) in sensitivity branch; (response, context) in legacy path.
    """
    if query is None:
        query = "What are conv nets and why they are used?"

    # --- Load model + tokenizer (as before or injected) ---
    if generator is not None:
        base_model, eval_tokenizer = generator
        print("Injected generator loaded (smoke mode).")
    else:
        llm_path = "./LLMs"
        base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)
        print("Local Base Model Loaded Successfully!")

    # --- Finetuned adapter (PEFT) ---
    # Use FT only in real runs (no injected generator) and when not disabled
    use_ft = (generator is None) and (os.getenv("LLMRAG_DISABLE_FT", "0") != "1")

    if use_ft:
        ft_ckpt = ".models/gen_model/qa_pairs_finetuning"
        try:
            ft_model = load_ft_model(base_model=base_model, ft_ckpt=ft_ckpt)
            print("Finetuned adapter loaded.")
        except Exception as e:
            print(f"[FT WARNING] Could not load adapter from '{ft_ckpt}': {e}")
            print("[FT WARNING] Falling back to base model.")
            ft_model = base_model
    else:
        print("Skipping FT adapter (smoke mode or disabled).")
        ft_model = base_model

    # Embedder (injected or load)
    if embedder is not None:
        embedd_model = embedder
        print("Injected embedder loaded (smoke mode).")
    else:
        embedd_ftmodel_ckpt = ".models/emb_model"
        embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    # Vector store (injected or ensure on disk)
    if vectorstore is not None:
        vec_DB = vectorstore
        print("Injected vector store loaded (smoke mode).")
    else:
        db_path = "./FT_1RAG/vecDB/textbooks_v2"
        vec_DB, _ = _ensure_vecdb(embedd_model=embedd_model, db_path=db_path, create_new=False)

    # === Legacy path (no prompt_id) keeps old behavior ===
    if prompt_id is None:
        print(f"\nUser:\n{query}")
        response, context = rag_chatbot(
            user_query=query,
            vec_db=vec_DB,
            model=ft_model,
            tokenizer=eval_tokenizer
        )
        print(f"\nChatbot:\n{response}")
        return response, context

    # === Sensitivity path (prompt-aware) ===
    context_text, ctx_ids, _docs = _retrieve_context(vec_DB, query, k=top_k, order=ctx_order)
    prompt, resolved_pid = _build_prompt(prompt_id, context_text, query)
    answer_text = _generate_with_model(ft_model, eval_tokenizer, prompt, temperature=temperature)

    meta = {"ctx_ids": ctx_ids, "prompt_id": resolved_pid, "top_k": top_k, "ctx_order": ctx_order}
    return answer_text, meta

if __name__ == "__main__":
    response, context = run_rag_pipeline(query=None)

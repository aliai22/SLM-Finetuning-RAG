from llmrag.training.slm.qa_pairs.LLM_Model import load_local_llm
from llmrag.pipelines.ft_2rag.finetuned_model import load_ft_model
from llmrag.pipelines.ft_2rag.dataset import load_dataset, preprocess_dataset
from llmrag.pipelines.ft_2rag.vectorstore import LocalEmbeddingFunction, create_vecdb
from llmrag.pipelines.ft_2rag.chatbot import rag_chatbot
from llmrag.training.embedding.embedd_finetuning import load_model

import os, json, torch
from llmrag.eval.common.sensitivity.prompts import PROMPTS, DEFAULT_PROMPT_ID

def _ensure_vecdb_QA(embedd_model,
                     db_path:str="./FT_2RAG/vecDB/QAs_v3",
                     dataset_path:str=".data/finetuningq/qa/uniqueQA_dataset0.7.jsonl",
                     create_new:bool=False):
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
    if hasattr(vec_DB, "as_retriever"):
        retriever = vec_DB.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(query)
    elif hasattr(vec_DB, "similarity_search"):
        docs = vec_DB.similarity_search(query, k=k)
    else:
        docs = []
    if order == "shuffled" and docs:
        import random
        random.shuffle(docs)
    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
    ctx_ids = []
    for idx, d in enumerate(docs):
        mid = None
        if hasattr(d, "metadata") and isinstance(d.metadata, dict):
            mid = d.metadata.get("doc_id") or d.metadata.get("source_id") or d.metadata.get("source") or d.metadata.get("id")
        ctx_ids.append(mid if mid is not None else str(idx))
    return context_text, ctx_ids, docs

def _build_prompt(pid: str, context_text: str, question: str):
    pid = pid if (pid in PROMPTS) else DEFAULT_PROMPT_ID
    return PROMPTS[pid].format(context=context_text, question=question), pid

def _generate_with_model(model, tokenizer, prompt: str, temperature: float = 0.1, max_new_tokens: int = 256):
    model.eval()
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = (temperature is not None) and (float(temperature) > 0.0)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=float(temperature) if temperature else 0.0,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

def run_rag_pipeline(query:str=None,
                     prompt_id:str=None,
                     top_k:int=5,
                     ctx_order:str="as_is",
                     temperature:float=0.1,
                     max_new_tokens:int=256,
                     *,
                     # ---- NEW: optional smoke injections ----
                     vectorstore=None,
                     embedder=None,
                     generator=None):
    """
    FT2 RAG with QA vecDB.
    - Legacy mode (prompt_id=None): calls rag_chatbot(...) and returns (response, context_docs).
    - Sensitivity mode (prompt_id provided): retrieves top_k QA docs and generates with finetuned/base model.
    - Smoke mode: if (generator/embedder/vectorstore) are injected, skip the heavy parts.
    """
    if query is None:
        query = "what is machine learning?"

    # --- base model (injected or local) ---
    if generator is not None:
        base_model, eval_tokenizer = generator
        print("Injected generator loaded (smoke mode).")
    else:
        llm_path = "./LLMs"
        base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)
        print("Local Base Model Loaded Successfully!")

    # --- finetuned adapter: use only when not injected and not disabled ---
    use_ft = (generator is None) and (os.getenv("LLMRAG_DISABLE_FT", "0") != "1")
    if use_ft:
        ft_ckpt = "./Finetuning_Checkpoints_filtered/final-checkpoint/checkpoint-9750"
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

    # --- embedder (injected or load) ---
    if embedder is not None:
        embedd_model = embedder
        print("Injected embedder loaded (smoke mode).")
    else:
        embedd_ftmodel_ckpt = ".models/emb_model/finetuning_embeddModel"
        embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    # --- vectorstore (injected or ensure QA vecDB) ---
    if vectorstore is not None:
        vec_DB = vectorstore
        print("Injected vector store loaded (smoke mode).")
    else:
        db_path = "./FT_2RAG/vecDB/QAs_v3"
        vec_DB, _ = _ensure_vecdb_QA(
            embedd_model=embedd_model,
            db_path=db_path,
            dataset_path=".data/finetuningq/qa/uniqueQA_dataset0.7.jsonl",
            create_new=False
        )

    # === Legacy path (no prompt control) ===
    if prompt_id is None:
        print(f"\nUser:\n{query}")
        response, context_docs = rag_chatbot(
            user_query=query,
            vec_db=vec_DB,
            model=base_model,          # unchanged to match your original rag_chatbot usage
            tokenizer=eval_tokenizer
        )
        print(f"\nChatbot:\n{response}")
        return response, context_docs

    # === Sensitivity path ===
    context_text, ctx_ids, docs = _retrieve_context(vec_DB, query, k=top_k, order=ctx_order)
    prompt, resolved_pid = _build_prompt(prompt_id, context_text, query)
    answer_text = _generate_with_model(ft_model, eval_tokenizer, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
    return answer_text, docs

if __name__ == "__main__":
    response, context = run_rag_pipeline(query=None)

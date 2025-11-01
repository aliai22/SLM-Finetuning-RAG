from llmrag.pipelines.ft_1rag.vectorstore import query_vecdb

import re
import os
import torch
from langchain_core.prompts import ChatPromptTemplate

def _pick_device(model):
    """
    Choose a sensible device:
    - If the model has .device (common when loaded with accelerate/device_map), use it.
    - Else, use CUDA if available, otherwise CPU.
    - Allow override via env LLMRAG_DEVICE (e.g., 'cpu', 'cuda', 'cuda:0').
    """
    dev_env = os.getenv("LLMRAG_DEVICE")
    if dev_env:
        return torch.device(dev_env)

    # Some HF models expose model.device; Accelerate puts modules on multiple devices,
    # but .device is still valid for the wrapper.
    if hasattr(model, "device") and model.device is not None:
        return model.device

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _summarize_docs(docs, tokenizer, k=2, per_doc_tokens=80):
    """Keep top-k docs; trim each doc to ~per_doc_tokens tokens."""
    kept = []
    for d in docs[:k]:
        text = getattr(d, "page_content", str(d))
        # Tokenize and keep only tail tokens (often most specific)
        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) > per_doc_tokens:
            toks = toks[-per_doc_tokens:]
            text = tokenizer.decode(toks, skip_special_tokens=True)
        kept.append(text)
    return kept

def _to_device(batch, device):
    # Move only tensors; keep others untouched
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

def _max_ctx_len(model):
    # Try common HF config fields; fall back conservatively.
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "max_position_embeddings"):
        return int(cfg.max_position_embeddings)
    if cfg is not None and hasattr(cfg, "n_positions"):
        return int(cfg.n_positions)
    return 512

def generate_response(model, tokenizer, text):
    # previous behavior preserved: same signature & sampling knobs
    device = _pick_device(model)

    # --- context-fit guard (prevents "index out of range") ---
    max_ctx = _max_ctx_len(model)
    # keep some space for generation; cap at 1/4 of context or 1024 (whichever is smaller)
    budget = max(16, min(1024, max_ctx // 4))
    max_input_len = max(8, max_ctx - budget)

    # Tokenize with truncation to fit within model context
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    )
    # Hard-clip as a backstop in case tokenizer config ignores max_length
    if tokens["input_ids"].shape[-1] > max_input_len:
        tokens["input_ids"] = tokens["input_ids"][:, -max_input_len:]
        if "attention_mask" in tokens:
            tokens["attention_mask"] = tokens["attention_mask"][:, -max_input_len:]

    # Place model/tensors on the chosen device (no-op on CPU)
    try:
        if not hasattr(model, "device") or str(model.device) != str(device):
            model.to(device)
    except Exception:
        pass
    tokens = _to_device(tokens, device)

    # Generate with a safe max_new_tokens = budget (prevents overflow)
    res = model.generate(
        **tokens,
        max_new_tokens=budget,          # was 1024; now safely bounded by context
        num_return_sequences=1,
        temperature=0.01,
        num_beams=1,
        top_p=0.95,
        do_sample=True
    ).to("cpu")  # keep your original .to('cpu') for decoding

    return tokenizer.batch_decode(res, skip_special_tokens=True)

def _count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def _auto_limits(user_query, docs, tokenizer, model, want_k_default=3):
    """
    Decide k and per_doc_tokens automatically to fit model context
    without any env vars.
    """
    max_ctx = _max_ctx_len(model)

    # reserve some tokens for generation; at most 1/4 of context, but <= 256
    gen_budget = max(16, min(256, max_ctx // 4))

    # rough prompt overhead: headings + boilerplate
    overhead = 40

    # tokens available to feed as input (prompt)
    max_input = max(64, max_ctx - gen_budget)

    q_tokens = _count_tokens(tokenizer, user_query)
    remaining = max_input - q_tokens - overhead

    if remaining <= 48:
        # really tiny room: keep 1 short snippet
        return 1, max(24, remaining)

    # start with a small k and adjust so each doc gets a decent slice
    k = min(want_k_default, len(docs))
    per_doc = max(48, remaining // max(1, k))

    # if each doc slice is too tiny, reduce k until it’s reasonable
    while k > 1 and per_doc < 32:
        k -= 1
        per_doc = remaining // k

    # clip per_doc to something sane
    per_doc = int(max(24, min(per_doc, 256)))
    return k, per_doc

def _summarize_docs_auto(docs, tokenizer, k, per_doc_tokens):
    kept = []
    for d in docs[:k]:
        text = getattr(d, "page_content", str(d))
        toks = tokenizer.encode(text, add_special_tokens=False)
        if len(toks) > per_doc_tokens:
            toks = toks[-per_doc_tokens:]
            text = tokenizer.decode(toks, skip_special_tokens=True)
        kept.append(text)
    return kept

def rag_chatbot(user_query: str, vec_db, model, tokenizer):
    PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions clearly and accurately based only on the given context below.

Only use the provided context below to answer the question. If the context does not contain a complete answer, respond with "I don’t know based on the provided context."
You must not use your own knowledge or assumptions.

---

Context:
{context}

Question:
{query}

Answer:
"""

    # retrieve as before
    retrieved_context = query_vecdb(query=user_query, vectorstore=vec_db)
    if not retrieved_context:
        return "I don’t know based on the provided context.", None

    # --- AUTO limits (env vars still respected if set, but not required) ---
    k_env = os.getenv("LLMRAG_RAG_TOPK")
    per_doc_env = os.getenv("LLMRAG_RAG_PERDOC")

    if k_env is not None and per_doc_env is not None:
        k = int(k_env)
        per_doc = int(per_doc_env)
    else:
        # pick sensible k and per-doc tokens automatically
        k, per_doc = _auto_limits(user_query, retrieved_context, tokenizer, model, want_k_default=3)

    trimmed = _summarize_docs_auto(retrieved_context, tokenizer, k=k, per_doc_tokens=per_doc)
    context_text = "\n\n".join(trimmed)

    prompt = PROMPT_TEMPLATE.format(context=context_text, query=user_query)

    response = generate_response(model=model, tokenizer=tokenizer, text=prompt)

    out_text = response[0] if isinstance(response, list) else str(response)
    if "Answer:" in out_text:
        after = out_text.split("Answer:", 1)[1]
        first_line = next((ln.strip() for ln in after.splitlines() if ln.strip()), "")
        output = first_line or after.strip()
    else:
        output = out_text.strip()

    return output, retrieved_context
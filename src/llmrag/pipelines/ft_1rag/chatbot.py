from llmrag.pipelines.ft_1rag.vectorstore import query_vecdb
import os
import torch

# ---------------- device + context helpers ----------------
def _pick_device(model):
    dev_env = os.getenv("LLMRAG_DEVICE")
    if dev_env:
        return torch.device(dev_env)
    if hasattr(model, "device") and model.device is not None:
        return model.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}

def _max_ctx_len(model):
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "max_position_embeddings"):
        return int(cfg.max_position_embeddings)
    if cfg is not None and hasattr(cfg, "n_positions"):
        return int(cfg.n_positions)
    return 512  # conservative fallback

def _count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def _auto_limits(user_query, docs, tokenizer, model, want_k_default=3):
    max_ctx = _max_ctx_len(model)
    gen_budget = max(16, min(256, max_ctx // 4))
    overhead = 40
    max_input = max(64, max_ctx - gen_budget)
    q_tokens = _count_tokens(tokenizer, user_query)
    remaining = max_input - q_tokens - overhead
    if remaining <= 48:
        return 1, max(24, remaining)
    k = min(want_k_default, len(docs))
    per_doc = max(48, remaining // max(1, k))
    while k > 1 and per_doc < 32:
        k -= 1
        per_doc = max(24, remaining // max(1, k))
    return k, int(min(per_doc, 256))

def _trim_text(tokenizer, text, per_doc_tokens):
    toks = tokenizer.encode(text, add_special_tokens=False)
    if len(toks) > per_doc_tokens:
        toks = toks[-per_doc_tokens:]
        text = tokenizer.decode(toks, skip_special_tokens=True)
    return text
# -----------------------------------------------------------

def generate_response(model, tokenizer, text):
    # previous signature retained; now device/context safe
    device = _pick_device(model)
    max_ctx = _max_ctx_len(model)
    gen_budget = max(16, min(256, max_ctx // 4))
    max_input_len = max(8, max_ctx - gen_budget)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    )
    # backstop hard-clip
    if tokens["input_ids"].shape[-1] > max_input_len:
        tokens["input_ids"] = tokens["input_ids"][:, -max_input_len:]
        if "attention_mask" in tokens:
            tokens["attention_mask"] = tokens["attention_mask"][:, -max_input_len:]

    try:
        if not hasattr(model, "device") or str(model.device) != str(device):
            model.to(device)
    except Exception:
        pass
    tokens = _to_device(tokens, device)

    res = model.generate(
        **tokens,
        max_new_tokens=gen_budget,
        num_return_sequences=1,
        temperature=0.01,
        num_beams=1,
        top_p=0.95,
        do_sample=True
    ).to("cpu")

    return tokenizer.batch_decode(res, skip_special_tokens=True)

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

    # 1) Retrieve
    retrieved_context = query_vecdb(query=user_query, vectorstore=vec_db)
    print(f"\nDense Retrieval Context:\n{retrieved_context}")

    if not retrieved_context:
        return "I don't know. I couldn't find the relevant information in the provided context.", None

    # 2) Auto-trim to fit model context
    k, per_doc = _auto_limits(user_query, retrieved_context, tokenizer, model, want_k_default=3)
    trimmed_snippets = []
    for d in retrieved_context[:k]:
        txt = getattr(d, "page_content", str(d))
        trimmed_snippets.append(_trim_text(tokenizer, txt, per_doc))
    text = "\n\n".join(trimmed_snippets)

    # 3) Build prompt + generate
    prompt = PROMPT_TEMPLATE.format(context=text, query=user_query)
    response = generate_response(model=model, tokenizer=tokenizer, text=prompt)

    out_text = response[0] if isinstance(response, list) else str(response)
    if "Answer:\n" in out_text:
        after = out_text.split("Answer:\n", 1)[1]
        first_line = next((ln.strip() for ln in after.splitlines() if ln.strip()), "")
        output = first_line or after.strip()
    else:
        output = out_text.strip()
    return output, retrieved_context

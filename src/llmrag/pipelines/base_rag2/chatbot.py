from .vectorstore import query_vecdb
import re
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

def _trim_doc(tokenizer, text, per_doc_tokens):
    toks = tokenizer.encode(text, add_special_tokens=False)
    if len(toks) > per_doc_tokens:
        toks = toks[-per_doc_tokens:]
        text = tokenizer.decode(toks, skip_special_tokens=True)
    return text
# -----------------------------------------------------------

def generate_response(model, tokenizer, text):
    # keep your output format; just make it device/context safe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    device = _pick_device(model)
    max_ctx = _max_ctx_len(model)
    gen_budget = max(16, min(256, max_ctx // 4))
    max_input_len = max(8, max_ctx - gen_budget)

    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_input_len
    )
    # hard-clip as backstop
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

    output = model.generate(
        **tokens,
        max_new_tokens=gen_budget,
        temperature=0.1,
        num_beams=1,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    ).to("cpu")

    return tokenizer.batch_decode(output, skip_special_tokens=True)

def format_qa_context(docs):
    formatted_qas = []
    for idx, doc in enumerate(docs, 1):
        match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", doc.page_content, re.DOTALL)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            qa_formatted = f"QA{idx}:\nQuestion: {question}\nAnswer: {answer} <|endoftext|>"
            formatted_qas.append(qa_formatted)
    return "\n\n".join(formatted_qas)

def rag_chatbot(user_query:str, vec_db, model, tokenizer):
    PROMPT_TEMPLATE = """
You are a helpful AI assistant. You will be given some QA pairs retrieved from a textbook-based knowledge source.

Each QA pair is labeled as QA1, QA2, etc. Every answer ends with the special token <|endoftext|>. Based on the provided context, answer the user query **factually** and **only based on the context**.

Use a clear and concise explanation. Your answer must also end with the token <|endoftext|>.

Context:
{context}

Now, answer the following question:

User Query:
{question}

Answer:
"""

    # retrieve
    retrieved_context = query_vecdb(query=user_query, vectorstore=vec_db)
    if not retrieved_context:
        return "I don't know. I couldn't find the relevant information in the provided context."

    # auto decide k + per-doc tokens to fit model context
    k, per_doc = _auto_limits(user_query, retrieved_context, tokenizer, model, want_k_default=3)

    # trim each QA chunk before formatting
    trimmed_docs = []
    for d in retrieved_context[:k]:
        text = getattr(d, "page_content", str(d))
        trimmed = _trim_doc(tokenizer, text, per_doc_tokens=per_doc)
        # build a shallow clone with trimmed content for formatter
        trimmed_docs.append(type("Doc", (), {"page_content": trimmed, "metadata": getattr(d, "metadata", {})}))

    formatted_context = format_qa_context(trimmed_docs)
    prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=user_query)

    response = generate_response(model=model, tokenizer=tokenizer, text=prompt)
    out_text = response[0] if isinstance(response, list) else str(response)
    # robust extraction
    if "Answer:" in out_text:
        after = out_text.split("Answer:", 1)[1]
        output = after.strip()
    else:
        output = out_text.strip()
    return output, retrieved_context

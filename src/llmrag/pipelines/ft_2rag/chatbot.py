from llmrag.pipelines.ft_2rag.vectorstore import query_vecdb
# from llmrag.training.embedding.embedd_finetuning import load_model
import json, os, re, torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

# ====== constants (unchanged) ======
KW_MODEL = KeyBERT(model='all-MiniLM-L12-v2')
TOP_K_DENSE   = 20
TOP_K_CONTEXT = 5

KW_DB_PATH = ".data/resources/keywords_database.json"
keyword_db = None  # lazy-loaded only if enabled and present

def _maybe_load_keyword_db():
    global keyword_db
    if keyword_db is not None:
        return keyword_db
    try:
        with open(KW_DB_PATH, "r", encoding="utf-8") as f:
            keyword_db = json.load(f)  # {doc_id: [kw1, kw2, ...]}
            print(f"[KW] Loaded keyword DB: {KW_DB_PATH}")
    except FileNotFoundError:
        print(f"[KW] {KW_DB_PATH} not found — keyword reranking disabled.")
        keyword_db = None
    except Exception as e:
        print(f"[KW] Failed to load {KW_DB_PATH}: {e} — keyword reranking disabled.")
        keyword_db = None
    return keyword_db

KW_SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
SIM_THRESHOLD = 0.6
ALPHA = 0.7
BETA  = 0.3

# ====== helpers (device + context) ======
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
    return 512  # fallback

def _count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))

def _auto_limits(user_query, docs, tokenizer, model, want_k_default=5):
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

# ====== your original keyword helpers (unchanged logic) ======
def extract_KW_from_query(kw_model, doc: str, top_n: int, mmr_diversity: float):
    keyphrases = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        use_mmr=True,
        diversity=mmr_diversity,
        top_n=top_n
    )
    filtered_keywords = [kw for kw, score in keyphrases if score > 0.6]
    return filtered_keywords

def keyword_weighted_rerank(query, candidate_docs, keyword_db, top_k=5):
    query_kws = extract_KW_from_query(KW_MODEL, doc=query, top_n=5, mmr_diversity=0.7)
    q_kw_embs = KW_SIM_MODEL.encode(query_kws, convert_to_tensor=True, normalize_embeddings=True)
    new_scored = []
    for doc, dense_sim in candidate_docs:
        doc_id = doc.metadata.get("doc_id")
        doc_kws = [kw for kw, _ in keyword_db.get(doc_id, [])]
        if not doc_kws:
            kw_score = 0.0
        else:
            d_kw_embs = KW_SIM_MODEL.encode(doc_kws, convert_to_tensor=True, normalize_embeddings=True)
            sim_mat  = util.cos_sim(q_kw_embs, d_kw_embs)
            per_qmax = sim_mat.max(dim=1).values
            kw_score = per_qmax.mean().item()
        final_score = ALPHA * dense_sim + BETA * kw_score
        new_scored.append((doc, final_score))
    reranked = sorted(new_scored, key=lambda x: x[1], reverse=True)
    return [d for d, _ in reranked[:top_k]]

def kw_overlap_semantic(query_kws, doc_kws) -> bool:
    if not query_kws or not doc_kws:
        return False
    q_embs = KW_SIM_MODEL.encode(query_kws, convert_to_tensor=True, normalize_embeddings=True)
    d_embs = KW_SIM_MODEL.encode(doc_kws,   convert_to_tensor=True, normalize_embeddings=True)
    cos = util.cos_sim(q_embs, d_embs)
    return bool((cos >= SIM_THRESHOLD).any())

def kw_semantic_rerank(docs, query_keywords, keyword_db, embedder, top_k=5):
    q_embs = embedder.encode(query_keywords, convert_to_tensor=True, normalize_embeddings=True)
    scored = []
    docs_only = [d for d, _ in docs]  # docs is [(Document, dense_score), ...]
    for doc in docs_only:
        raw_kws = keyword_db.get(doc.metadata.get("doc_id"), [])
        if raw_kws and isinstance(raw_kws[0], tuple):
            d_kws = [kw for kw, _ in raw_kws]
        else:
            d_kws = [kw for kw in raw_kws if isinstance(kw, str)]
        if not d_kws:
            scored.append((doc, 0.0))
            continue
        d_embs = embedder.encode(d_kws, convert_to_tensor=True, normalize_embeddings=True)
        d_embs = d_embs.to(q_embs.device)
        cos    = util.cos_sim(q_embs, d_embs)
        score  = cos.max(dim=1).values.mean().item()
        scored.append((doc, score))
    reranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [d for d, _ in reranked[:top_k]]

# keep this global as in your original
# embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
# embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

# ====== generation (device-agnostic, context-safe) ======
def generate_response(model, tokenizer, text):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    device = _pick_device(model)
    max_ctx = _max_ctx_len(model)
    gen_budget = max(32, min(256, max_ctx // 4))
    max_input_len = max(64, max_ctx - gen_budget)

    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_input_len
    )

    # hard-clip to be extra safe
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
        temperature=0.3,
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
            qa_formatted = f"QA{idx}:\nQuestion: {question}\nAnswer: {answer}"
            formatted_qas.append(qa_formatted)
    return "\n\n".join(formatted_qas)

def rag_chatbot(user_query: str, vec_db, model, tokenizer):
    PROMPT_TEMPLATE = """
You are a helpful AI assistant whose job is to answer user queries. You will be given some example QA pairs relevant to the asked query.

Each QA pair is labeled as QA1, QA2, etc. Based on the provided context, answer the user query factually and only based on the context.

Use a clear and concise explanation.

Context:
{context}

Now, answer the following question:

User Query:
{question}

Answer:
"""

    # 1) Dense retrieval (get a pool with scores)
    retrieved_context = query_vecdb(query=user_query, vectorstore=vec_db, top_k=TOP_K_DENSE)
    print(f"\nContext:\n{retrieved_context}")

    if not retrieved_context:
        return "I don't know. I couldn't find the relevant information in the provided context.", None

    # 2) Keyword extraction + semantic rerank (uses your DB keywords)
    # Normalize: we may have [(Document, score), ...] or [Document, ...]
    if retrieved_context and isinstance(retrieved_context[0], tuple):
        candidate_pairs = retrieved_context
    else:
        candidate_pairs = [(d, 1.0) for d in retrieved_context]

    # Feature flag (default OFF). Set LLMRAG_ENABLE_KW=1 to enable keyword reranking.
    kw_enabled = os.getenv("LLMRAG_ENABLE_KW", "0") == "1"
    db = _maybe_load_keyword_db() if kw_enabled else None

    if kw_enabled and db:
        # Keyword extraction + semantic rerank
        query_kws = extract_KW_from_query(KW_MODEL, doc=user_query, top_n=5, mmr_diversity=0.7)
        contexts = kw_semantic_rerank(
            docs=candidate_pairs,
            query_keywords=query_kws,
            keyword_db=db,
            embedder=KW_SIM_MODEL
        )
        print(f"\nKeyword Filtered Context:\n{contexts}")
    else:
        # Dense-only fallback (top-k)
        # TOP_K_CONTEXT = 5  # reuse your constant or keep same value here
        contexts = [d for d, _ in candidate_pairs[:TOP_K_CONTEXT]]
        if not kw_enabled:
            print("[KW] Disabled via env (LLMRAG_ENABLE_KW!=1). Using dense-only top-k.")
        elif not db:
            print("[KW] DB missing/unloadable. Using dense-only top-k.")

    # 3) Auto-trim selected contexts to fit model context window
    want_k = min(TOP_K_CONTEXT, len(contexts))
    k_auto, per_doc = _auto_limits(user_query, contexts, tokenizer, model, want_k_default=want_k)
    trimmed = []
    for d in contexts[:k_auto]:
        txt = getattr(d, "page_content", str(d))
        trimmed.append(_trim_text(tokenizer, txt, per_doc))
    formatted_context = format_qa_context(contexts[:k_auto]) if trimmed else ""

    prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=user_query)
    response = generate_response(model=model, tokenizer=tokenizer, text=prompt)

    out_text = response[0] if isinstance(response, list) else str(response)
    if "Answer:\n" in out_text:
        after = out_text.split("Answer:\n", 1)[1]
        first_line = next((ln.strip() for ln in after.splitlines() if ln.strip()), "")
        output = first_line or after.strip()
    else:
        output = out_text.strip()

    return output, contexts

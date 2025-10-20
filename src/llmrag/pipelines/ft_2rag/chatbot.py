from FT_2RAG.vectorstore import query_vecdb
from finetuning_embeddModel.embedd_finetuning import load_model
import json
from keybert import KeyBERT

import re
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer, util

KW_MODEL = KeyBERT(model='all-MiniLM-L12-v2')    # fast model for extractor
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

KW_SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
SIM_THRESHOLD = 0.6          # tune 0.4–0.6

# KW_SIM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # light encoder
ALPHA = 0.7   # weight on dense cosine similarity
BETA  = 0.3   # weight on keyword score  (α+β=1 is convenient)

def keyword_weighted_rerank(query, candidate_docs, keyword_db, top_k=5):
    """
    candidate_docs : List[ (Document, dense_similarity_score) ]
    Returns same docs re-ordered by α·cos + β·kw_score.
    """
    # --- pre-compute query-keyword set
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
            # cosine sim (q_kw × d_kw) -> take max per query_kw then mean
            sim_mat  = util.cos_sim(q_kw_embs, d_kw_embs)         # (|QKW|, |DKW|)
            per_qmax = sim_mat.max(dim=1).values                  # len == |QKW|
            kw_score = per_qmax.mean().item()                    # 0–1

        final_score = ALPHA * dense_sim + BETA * kw_score
        new_scored.append((doc, final_score))

    # sort descending by fused score
    reranked = sorted(new_scored, key=lambda x: x[1], reverse=True)
    print(f"RERANKED: {reranked}")
    return [d for d, _ in reranked[:top_k]]

def kw_overlap_semantic(query_kws, doc_kws) -> bool:
    """
    Returns True if *any* query keyword has *semantic* cosine ≥ threshold
    with *any* doc keyword.
    """
    if not query_kws or not doc_kws:
        return False

    q_embs = KW_SIM_MODEL.encode(query_kws, convert_to_tensor=True, normalize_embeddings=True)
    d_embs = KW_SIM_MODEL.encode(doc_kws,   convert_to_tensor=True, normalize_embeddings=True)

    cos = util.cos_sim(q_embs, d_embs)          # shape (len(query_kws), len(doc_kws))
    print(f"SOCRE: {cos}")
    return bool((cos >= SIM_THRESHOLD).any())

def kw_semantic_rerank(
        docs, query_keywords, keyword_db, embedder, top_k=5,):
    """
    Uses cosine(queryKW, docKW) max-pooling like your function,
    but vectorised and returns top_k docs.
    """
    # encode query keywords once
    q_embs = embedder.encode(query_keywords, convert_to_tensor=True, normalize_embeddings=True)

    scored = []
    docs_only = [d for d, _ in docs]
    for doc in docs_only:
        # ------------------------------------------------------------
        # 1) Fetch keyword list from the DB
        # ------------------------------------------------------------
        raw_kws = keyword_db.get(doc.metadata["doc_id"], [])

        # --- Unify to list[str] ------------------------------------------
        if raw_kws and isinstance(raw_kws[0], tuple):
            # format: [(kw, score), ...]
            d_kws = [kw for kw, _ in raw_kws]
        else:
            # format: [kw1, score1, kw2, ...] or [kw1, kw2, ...]
            d_kws = [kw for kw in raw_kws if isinstance(kw, str)]

        # skip if still empty
        if not d_kws:
            scored.append((doc, 0.0))
            continue

        # ------------------------------------------------------------
        # 3) Compute semantic keyword score
        # ------------------------------------------------------------
        d_embs = embedder.encode(
            d_kws,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # ensure same device as query‐keyword embeddings
        d_embs = d_embs.to(q_embs.device)

        cos    = util.cos_sim(q_embs, d_embs)            # shape: |QKW| × |DKW|
        score  = cos.max(dim=1).values.mean().item()     # pooling: mean(max_i)

        scored.append((doc, score))

    reranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [d for d,_ in reranked[:top_k]]

embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

def generate_response(model, tokenizer, text):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Fallback
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    print(tokenizer.pad_token)
    print(tokenizer.eos_token_id)

    tokens = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True
    )

    # END_TOKEN = tokenizer.eos_token
    # print(tokenizer.tokenize(END_TOKEN))
    # print(tokenizer.convert_tokens_to_ids(END_TOKEN))

    # print(f"END_TOKEN: {END_TOKEN}")

    input_ids = tokens['input_ids'].to("cuda")
    attention_mask = tokens['attention_mask'].to("cuda")

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
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
        # Extract Q and A
        match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", doc.page_content, re.DOTALL)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            # Format with <END> token
            qa_formatted = f"QA{idx}:\nQuestion: {question}\nAnswer: {answer}"
            formatted_qas.append(qa_formatted)
    return "\n\n".join(formatted_qas)

def rag_chatbot(user_query:str, vec_db, model, tokenizer):
    # query_vector = create_embeddings(text=user_query,
    #                   model=SentenceTransformer('thenlper/gte-small', device="cuda"))

#     PROMPT_TEMPLATE = """
# You are a helpful AI assistant. You will be given some QA pairs retrieved from a textbook-based knowledge source.

# Each QA pair is labeled as QA1, QA2, etc. Every answer ends with the special token <|endoftext|>. Based on the provided context, answer the user query **factually** and **only based on the context**.

# Use a clear and concise explanation. Your answer must also end with the token <|endoftext|>.

# Context:
# {context}

# Now, answer the following question:

# User Query:
# {question}

# Answer:
# """

    PROMPT_TEMPLATE = """
You are a helpful AI assistant whose job is to answer user queries. You will be given some example QA pairs relevant to the asked query.

Each QA pair is labeled as QA1, QA2, etc. Based on the provided context, answer the user query **factually** and **only based on the context**.

Use a clear and concise explanation.

Context:
{context}

Now, answer the following question:

User Query:
{question}

Answer:
"""
    # 1) Dense retrieval  ------------------------------------------------
    # q_emb = embedd_model.encode(user_query, convert_to_tensor=True, normalize_embeddings=True)
    
    # Searching the Vector Database

    retrieved_context = query_vecdb(query=user_query,
                               vectorstore=vec_db,
                               top_k=TOP_K_DENSE)
    print(f"\nContext:\n{retrieved_context}")
    # retrieved_qa = semantic_chunk_retriever.invoke(user_query)

    if not retrieved_context:
        return "I don't know. I couldn't find the relevant information in the provided context."
    
    # # 2) Keyword extraction & filter  ------------------------------------
    query_kws = extract_KW_from_query(KW_MODEL, doc=user_query, top_n=5, mmr_diversity=0.7)
    # print(f"QUERY KKKEYWORDS: {query_kws}")
    # filtered  = []
    # for doc in retrieved_context:
    #     print(f"DOCUMENT: {doc}")
    #     doc_id = doc.metadata.get("doc_id")
    #     print(f"DOC ID: {doc_id}")
    #     kws = [k for k, _ in keyword_db.get(doc_id, [])]
    #     print(f"KWS: {kws}")
    #     # if any(k in kws for k in query_kws):
    #     #     filtered.append(doc)
    #     if kw_overlap_semantic(query_kws, kws):
    #         filtered.append(doc)

    # # if nothing survives, fall back to dense top-k
    # if not filtered:
    #     filtered = retrieved_context[:TOP_K_CONTEXT]
    # else:
    #     filtered = filtered[:TOP_K_CONTEXT]

    # contexts = [d for d in filtered]
    
    # contexts = keyword_weighted_rerank(query=user_query,
    #                         candidate_docs=retrieved_context,
    #                         keyword_db=keyword_db,
    #                         )
    
    contexts = kw_semantic_rerank(docs=retrieved_context,
                                  query_keywords=query_kws,
                                  keyword_db=keyword_db,
                                  embedder=KW_SIM_MODEL)
    
    print(f"\nDense Retrieval Context:\n{retrieved_context}")
    # retrieved_qa = semantic_chunk_retriever.invoke(user_query)
    print(f"\nKeyword Filtered Context:\n{contexts}")

    # Extract answers from the QA-format chunks (optional, based on your DB format)
    # raw_text = "\n".join(doc.page_content for doc in retrieved_context)
    # raw_text = format_qa_context([d for d, _ in retrieved_context])
    raw_text = format_qa_context(contexts)
    
    # Optional: If context includes `Q:` and `A:` blocks
    # context_text = re.findall(r"(Q:.*?A:.*?)(?=\nQ:|\Z)", raw_text, re.DOTALL)
    # formatted_context = "\n\n".join(qapair.strip() for qapair in context_text)

    # Or, if the chunks are clean already
    formatted_context = raw_text

    prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=user_query)
    print(f"PROMPT: {prompt}")

    
    # print(context)
    
    # history = []
    # retrieval = semantic_chunk_retriever.invoke(user_query)
    # for doc in retrieval:
    #     history.append(doc.page_content)
    # retrieval = test_collection.query(query_embeddings=query_vector, n_results=5)
    # documents = retrieval["documents"]
    # history.append(context)
    # history.append(user_query)
    # context = " ".join(history)
    # print(context)

    # message = [
    #     {"role":"system", "content":f"You are an AI assistant that answers a user's query accurately from the given context. If you can not find relevant information in context, please respond with 'I don't know'."},
    #     {"role":"user", "content": user_query}
    # ]

    response = generate_response(model=model,
                                 tokenizer=tokenizer,
                                 text=prompt)
    # print(response[0].split("Output:\n")[1].split("### End of Output", 1)[0].strip())
    # output = response[0].split("Output:\n")[1].split("### End of Output", 1)[0].strip()
    # output = output.split("\n")[0]
    output = response[0].split("Answer:\n")[1].split("\n")[0].strip()
    return output, contexts
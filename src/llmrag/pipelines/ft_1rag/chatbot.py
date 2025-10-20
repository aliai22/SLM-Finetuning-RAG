from FT_1RAG.vectorstore import query_vecdb
# from finetuning_embeddModel.embedd_finetuning import load_model

import re
from langchain.prompts import ChatPromptTemplate
# from sentence_transformers import SentenceTransformer
# import json

# KW_MODEL = SentenceTransformer("all-MiniLM-L12-v2")    # fast model for extractor
# TOP_K_DENSE   = 20
# TOP_K_CONTEXT = 5

# def extract_KW_from_query(
#     kw_model,
#     doc:str,
#     top_n:int,
#     mmr_diversity:int
# ):
#     keyphrases = kw_model.extract_keywords(
#         doc,
#         keyphrase_ngram_range=(1, 3),
#         stop_words='english',
#         use_mmr=True,
#         diversity=mmr_diversity,
#         top_n=top_n
#     )
    
    
#     # print(keyphrases)
#     filtered_keywords = [kw for kw, score in keyphrases if score > 0.6]
#     return filtered_keywords

# KW_DB_PATH = "keywords_database.json"

# with open(KW_DB_PATH, "r", encoding="utf-8") as f:
#     keyword_db = json.load(f)          # {doc_id: [kw1, kw2, ...]}

# embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
# embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

def generate_response(model, tokenizer, text):
    # prompt = f"User Query:\n{query}\n\nContext:\n{retrieved_qa}"

    # formatted_prompt = """
    # You are an AI assistant who answers a user's query based on the given context. You can only make conversations based on the provided context. If the context is not provided, politely say you don’t have knowledge about that topic.\nNow, answer the following question using the provided context.\n\n{prompt}\nOutput:\n"
    # """
    # message_text = f"Instruct: {prompt[0]['content']}\nUser's Query:{prompt[1]['content']}\nContext: {context}\nOutput:\n"
    # print(message_text)

    # print(formatted_prompt)
    
    tokens = tokenizer(text, return_tensors="pt")
    res = model.generate(**tokens.to("cuda"),
                         max_new_tokens=1024,
                         num_return_sequences=1,
                         temperature=0.01,
                         num_beams=1,
                         top_p=0.95,
                         do_sample=True
                        ).to('cpu')
    
    return tokenizer.batch_decode(res,skip_special_tokens=True)

def rag_chatbot(user_query:str, vec_db, model, tokenizer):
    # query_vector = create_embeddings(text=user_query,
    #                   model=SentenceTransformer('thenlper/gte-small', device="cuda"))

    PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions **clearly and accurately** based **only on the given context below**.

Only use the provided context below to answer the question. If the context does not contain a complete answer, respond with "I don’t know based on the provided context."

You must not use your own knowledge or assumptions.

---

Context:
{context}

Question:
{query}

Answer:
"""

    # # 1) Dense retrieval  ------------------------------------------------
    # q_emb = embedd_model.encode(user_query, convert_to_tensor=True, normalize_embeddings=True)
    # dense_hits = vec_DB.similarity_search_with_score(q_emb, k=TOP_K_DENSE)   # [(doc, score), ...]

    # Searching the Vector Database

    retrieved_context = query_vecdb(query=user_query,
                               vectorstore=vec_db,
                               )

    # # 2) Keyword extraction & filter  ------------------------------------
    # query_kws = extract_KW_from_query(KW_MODEL, doc=user_query, top_n=5, mmr_diversity=0.7)
    # filtered  = []
    # for doc, score in retrieved_context:
    #     doc_id = doc.metadata.get("doc_id")
    #     kws = [k for k, _ in keyword_db.get(doc_id, [])]
    #     if any(k in kws for k in query_kws):
    #         filtered.append((doc, score))

    # # if nothing survives, fall back to dense top-k
    # if not filtered:
    #     filtered = retrieved_context[:TOP_K_CONTEXT]
    # else:
    #     filtered = filtered[:TOP_K_CONTEXT]

    # contexts = [d.page_content for d, _ in filtered]
    
    
    print(f"\nDense Retrieval Context:\n{retrieved_context}")
    # retrieved_qa = semantic_chunk_retriever.invoke(user_query)
    # print(f"\nKeyword Filtered Context:\n{retrieved_context}")

    if not retrieved_context:
        return "I don't know. I couldn't find the relevant information in the provided context.", None
    
    # text = "\n\n".join(retrieved_context)

    text = "\n\n".join(item.page_content for item in retrieved_context)
    
    # # Use regex to find all content after "A:"
    # answers = re.findall(r'A:\s*(.*?)(?=\s*Q:|$)', text, re.DOTALL)
    
    # # Combine all answers into a single string
    # context_text = " ".join(answer.strip() for answer in answers)

    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_template = PROMPT_TEMPLATE
    prompt = prompt_template.format(context=text, query=user_query)
    # print(prompt)

    
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
    output = response[0].split("Answer:\n")[1].split("\n")[0]
    return output, retrieved_context
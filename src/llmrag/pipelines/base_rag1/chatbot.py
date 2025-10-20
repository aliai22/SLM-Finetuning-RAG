from FT_1RAG.vectorstore import query_vecdb

import re
from langchain.prompts import ChatPromptTemplate

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

    
    # Searching the Vector Database

    retrieved_context = query_vecdb(query=user_query,
                               vectorstore=vec_db)
    print(f"\nContext:\n{retrieved_context}")
    # retrieved_qa = semantic_chunk_retriever.invoke(user_query)

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
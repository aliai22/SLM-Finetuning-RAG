from .vectorstore import query_vecdb

import re
from langchain.prompts import ChatPromptTemplate

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
        # Extract Q and A
        match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", doc.page_content, re.DOTALL)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            # Format with <END> token
            qa_formatted = f"QA{idx}:\nQuestion: {question}\nAnswer: {answer} <|endoftext|>"
            formatted_qas.append(qa_formatted)
    return "\n\n".join(formatted_qas)

def rag_chatbot(user_query:str, vec_db, model, tokenizer):
    # query_vector = create_embeddings(text=user_query,
    #                   model=SentenceTransformer('thenlper/gte-small', device="cuda"))

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

    
    # Searching the Vector Database

    retrieved_context = query_vecdb(query=user_query,
                               vectorstore=vec_db)
    print(f"\nContext:\n{retrieved_context}")
    # retrieved_qa = semantic_chunk_retriever.invoke(user_query)

    if not retrieved_context:
        return "I don't know. I couldn't find the relevant information in the provided context."
    
    # Extract answers from the QA-format chunks (optional, based on your DB format)
    # raw_text = "\n".join(doc.page_content for doc in retrieved_context)
    raw_text = format_qa_context(retrieved_context)
    
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
    output = response[0].split("Answer:\n")[1].strip()
    return output, retrieved_context
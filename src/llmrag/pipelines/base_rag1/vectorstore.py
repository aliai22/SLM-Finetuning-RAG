from chromadb import Documents, EmbeddingFunction, Embeddings
from math import ceil
import torch
from langchain_community.vectorstores import Chroma

from FT_1RAG.dataset import batchify

import torch
import gc
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
import os
from typing import List
import os
import re
import unicodedata

from langchain.schema import Document


# import chromadb

# def generate_embeddings(model, tokenizer, text):
#     model.eval()
#     # Ensure input is a batch of strings
#     if isinstance(text, str):
#         text = [text]
#     # print(text)
#     # print(type(text))
#     t_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
#     with torch.no_grad():
#         last_hidden_state = model(**t_input.to("cuda"), output_hidden_states=True).hidden_states[-1]
    
#     # Mean Pooling
#     embeddings = last_hidden_state.mean(dim=1)
#     # print(embeddings.shape)
#     # Normalize embeddings
#     embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
#     # print(len(embeddings.tolist()[0]))
#     return embeddings.tolist()[0]  # Returns a list of embeddings (one per input text)

def generate_embeddings(model, text):
    embeddings = model.encode(text)
    return embeddings.tolist()

class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embedd_model):
        self.model = embedd_model
        # self.eval_tokenizer = tokenizer
 
    def __call__(self, input: Documents) -> Embeddings:
        #Convert the numpy array to a Python list
        return generate_embeddings(model=self.model,
                                   text=input)
    
    def embed_documents(self, texts):
        embeddings = generate_embeddings(model=self.model,
                                         text=texts)
        return embeddings
    
    def embed_query(self, query):
        query_embeddings = generate_embeddings(model=self.model,
                                               text=query)
        return query_embeddings
# def create_db(path:str, collection_name:str, model, tokenizer):
#     client = chromadb.PersistentClient(path)
#     emf = LocalEmbeddingFunction(embedd_model=model,
#                                  tokenizer=tokenizer)
#     collection = client.get_or_create_collection(
#         name=collection_name,
#         embedding_function=emf
#     )
#     print(f"Collection {collection_name} Created Successfully!")
#     return collection

def clean_text(text: str) -> str:
    """
    Cleans extracted text by:
    - Normalizing unicode
    - Removing excessive symbols, junk lines
    - Replacing smart quotes and dashes
    - Stripping extra whitespace
    """
    # Normalize to remove smart quotes, etc.
    text = unicodedata.normalize("NFKD", text)

    # Replace smart quotes and dashes
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\uf0b7": "-",  # Bullet
        "\xa0": " ",  # Non-breaking space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove junk lines: long sequences of symbols or digits
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[\d\W_]{5,}$", line):
            continue  # likely garbage
        if re.search(r"[\ud800-\udfff]", line):
            continue  # remove surrogate junk
        clean_lines.append(line)

    return " ".join(clean_lines)

def extract_text_by_page(pdf_path: str, pdf_config: dict, config_index: int) -> list:
    """
    Extracts cleaned text from a PDF file page-by-page, applying skip and trim rules from config.

    Args:
        pdf_path (str): Path to the PDF file
        pdf_config (dict): Loaded config.json as a dict
        config_index (int): Index of the file in the config array (i in your code)

    Returns:
        List of dicts with 'text', 'source', and 'page'
    """
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    filename = os.path.basename(pdf_path)
    file_key = f"file{config_index + 1}.pdf"

    print(f"\U0001F4C4 Processing: {filename} | Total Pages: {total_pages}")

    skip_start = pdf_config[config_index].get(file_key, {}).get("skip_start_pages", 0)
    skip_end = pdf_config[config_index].get(file_key, {}).get("skip_last_pages", 0)
    header_lines = pdf_config[config_index].get(file_key, {}).get("header_lines", 0)
    footer_lines = pdf_config[config_index].get(file_key, {}).get("footer_lines", 0)

    print(f"  ⏩ Skip first {skip_start} pages, last {skip_end} pages")
    print(f"  \U0001F9FC Trim headers: {header_lines} lines | footers: {footer_lines} lines")

    raw_pages = []

    for page_num in range(skip_start, total_pages - skip_end):
        page = reader.pages[page_num]
        page_text = page.extract_text()

        if not page_text:
            continue

        lines = page_text.splitlines(True)  # keep line endings
        if header_lines > 0:
            lines = lines[header_lines:]
        if footer_lines > 0:
            lines = lines[:-footer_lines]

        joined_text = "".join(lines)
        cleaned = clean_text(joined_text)

        if cleaned:
            raw_pages.append({
                "text": cleaned,
                "source": filename,
                "page": page_num + 1
            })

    return raw_pages

def chunk_documents(dataset: List[dict], chunk_size=512, chunk_overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunked_docs = []

    for item in dataset:
        chunks = splitter.split_text(item["text"])
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": item["source"],
                    "page": item["page"]
                }
            )
            chunked_docs.append(doc)

    return chunked_docs

def clean_text(text):
    # Remove all surrogate characters (unpaired UTF-16 parts)
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

def create_vecdb(path: str, documents: List[Document], embedding_function, batch_size=32, create_new=True):

    if create_new:
        vectorstore = None
        print("Creating a new Vector Database...")

        batches = list(batchify(documents, batch_size))
        total_batches = len(batches)

        for idx, batch in enumerate(batches):
            print(f"Processing batch {idx + 1}/{total_batches}...")
            gc.collect()
            torch.cuda.empty_cache()

            if vectorstore:
                for doc in batch:
                    content = doc.page_content

                    if not isinstance(content, str):
                        print("❌ Not a string")
                        continue

                    content = clean_text(content)

                    if not content.strip():
                        print("⚠️ Empty after cleaning")
                        continue

                    try:
                        embedding = embedding_function(content)
                        vectorstore.add_texts([content], embeddings=[embedding], metadatas=[doc.metadata])
                    except Exception as e:
                        print("🚨 Still failed embedding!")
                        print("Value (repr):", repr(content[:300]))
                        print("Error:", str(e))
                        continue
            else:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_function,
                    persist_directory=path
                )

            vectorstore.persist()
            gc.collect()
            torch.cuda.empty_cache()

        print("✅ Vector DB created and saved at:", path)
        return vectorstore

    else:
        vectorstore = Chroma(
            persist_directory=path,
            embedding_function=embedding_function
        )
        print("Loaded existing Vector DB.")
        return vectorstore

def process_pdf_to_vecdb(
    pdf_text_by_page: List,
    db_path: str,
    embedding_function,
    batch_size: int = 32,
    create_new=True,
):
    chunked_docs = chunk_documents(pdf_text_by_page)
    vector_store = create_vecdb(db_path, chunked_docs, embedding_function, batch_size=batch_size, create_new=create_new)
    return vector_store

    # if create_new:
    #     vectorstore = Chroma.from_documents(
    #         documents=dataset,
    #         embedding=embedding_function,
    #         persist_directory=path
    #     )
    #     print("Vector Database Created Successfully!")
    #     vectorstore.persist()
    #     print("Vector Database Saved Successfully!")
    # else:
    #     vectorstore = Chroma(
    #     embedding=embedding_function,
    #     persist_directory=path
    # )
    # print("Vector Database Retrieved Successfully!")
    # return vectorstore

# def retrieve_vecdb(path:str, embedding_function):
#     vectorstore = Chroma(
#         embedding=embedding_function,
#         persist_directory=path
#     )
#     print("Vector Database Retrieved Successfully!")
#     return vectorstore

# def query_vecdb(query, vectorstore):
#     retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
#                              search_kwargs={"score_threshold": 0.5, "k" : 3}
#                              )
#     documents = retriever.invoke(query)
#     return documents

def query_vecdb(
    query: str,
    vectorstore,
    # embedding_function,
    k: int = 5,
    rerank_top_k: int = 3
) -> List[Document]:
    """
    Retrieves and optionally reranks top-k documents from the vectorstore.

    Args:
        query (str): User query
        vectorstore: Chroma or LangChain-compatible vector DB
        embedding_function: Callable that returns embeddings (1D or batched)
        k (int): Number of initial candidates to retrieve
        rerank_top_k (int): Final number of top documents to return after reranking
    
    Returns:
        List[Document]: Ranked and cleaned top documents
    """

    # Step 1: Initial retrieval with broader top-k
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    retrieved_docs = retriever.invoke(query)

    if not retrieved_docs:
        return []

    # # Step 2: Compute query embedding
    # query_embedding = embedding_function(query)
    # if isinstance(query_embedding, torch.Tensor):
    #     query_embedding = query_embedding.cpu().numpy()
    # if isinstance(query_embedding, list):  # some embedders return [embedding]
    #     query_embedding = np.array(query_embedding)

    # # Step 3: Rerank retrieved docs using cosine similarity
    # doc_embeddings = []
    # for doc in retrieved_docs:
    #     embedding = embedding_function(doc.page_content)
    #     if isinstance(embedding, torch.Tensor):
    #         embedding = embedding.cpu().numpy()
    #     if isinstance(embedding, list):
    #         embedding = np.array(embedding)
    #     doc_embeddings.append(embedding)

    # similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    # ranked_docs = sorted(zip(retrieved_docs, similarities), key=lambda x: x[1], reverse=True)

    # Step 4: Return top reranked documents
    # return [doc for doc, _ in ranked_docs[:rerank_top_k]]
    return [doc for doc in retrieved_docs]

def similarity_score(text1, text2, model, tokenizer):
    embedding_func = LocalEmbeddingFunction(embedd_model=model,
                                            tokenizer=tokenizer)
    embeddings1 = np.array(embedding_func.embed_query(query=text1))
    embeddings2 = np.array(embedding_func.embed_query(query=text2))

    embeddings1 = embeddings1.reshape(1,-1)
    embeddings2 = embeddings2.reshape(1,-1)

    # Calculate cosine similarity
    similarity = cosine_similarity(embeddings1, embeddings1)[0][0]
    return similarity

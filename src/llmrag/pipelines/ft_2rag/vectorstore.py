from chromadb import Documents, EmbeddingFunction, Embeddings
from math import ceil
import torch
import gc
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
from FT_2RAG.dataset import batchify


# ===================== Embedding =====================

def generate_embeddings(model, text):
    # Always return a list of floats (if single string) or list of lists (if list of strings)
    if isinstance(text, str):
        return [model.encode(text).tolist()]  # Single vector in a list
    return model.encode(text).tolist()  # Batch encoding


class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embedd_model):
        self.model = embedd_model

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        return generate_embeddings(model=self.model, text=input)

    def embed_documents(self, texts):
        return generate_embeddings(model=self.model, text=texts)

    def embed_query(self, query):
        return generate_embeddings(model=self.model, text=query)[0]


# ===================== Vector DB Creation =====================

def create_vecdb(path: str, dataset, embedding_function, batch_size=64, create_new=True):
    """
    Creates or updates a Chroma vector database with batch processing.

    Args:
        path (str): Directory path to persist the vector database.
        dataset (list): List of text chunks or QA documents.
        embedding_function: Embedding function object.
        create_new (bool): Whether to create a new vector database.
        batch_size (int): Items per batch.
    """

    if create_new:
        vectorstore = None
        print("🆕 Creating a new Vector Database...")
        total_batches = ceil(len(dataset) / batch_size)

        global_idx = 0
        for batch_idx in range(total_batches):
            batch_raw = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            print(f"🔄 Processing batch {batch_idx + 1}/{total_batches}...")

            gc.collect();  torch.cuda.empty_cache()

            # --- wrap each text into a Document with doc_id metadata ----
            batch_docs = []
            for text in batch_raw:
                doc = Document(
                    page_content=text,
                    metadata={"doc_id": f"doc_{global_idx}"}
                )
                batch_docs.append(doc)
                global_idx += 1

            if not batch_docs:
                print(f"⚠️ Skipping empty batch {batch_idx + 1}")
                continue

            # --- store ---------------------------------------------------
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch_docs,
                    embedding=embedding_function,
                    persist_directory=path
                )
            else:
                # ❶ embed each doc separately  (len == len(batch_docs))
                batch_embs = [embedding_function(d.page_content) for d in batch_docs]

                # ❷ add matching ids so all field-lists have the same length
                vectorstore.add_texts(
                    texts      =[d.page_content            for d in batch_docs],
                    metadatas  =[d.metadata                for d in batch_docs],
                    ids        =[d.metadata["doc_id"]      for d in batch_docs],
                    embeddings =batch_embs
                )


            vectorstore.persist()
            print(f"✅ Batch {batch_idx + 1} saved.")

        print("🎉 Vector Database Created and Saved Successfully!")
        return vectorstore

    else:
        vectorstore = Chroma(
            embedding_function=embedding_function,
            persist_directory=path
        )
        print("📁 Loaded existing Vector Database.")
        return vectorstore


# ===================== Query Vector DB =====================

# def query_vecdb(query, vectorstore, top_k=3, score_threshold=0.7, fallback_k=5):
#     """
#     Retrieves relevant documents from the vectorstore using similarity threshold,
#     with fallback to top-k similarity if no result meets the threshold.

#     Returns:
#         List of LangChain Document objects.
#     """
#     retriever = vectorstore.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={"score_threshold": score_threshold, "k": top_k}
#     )
#     results = retriever.invoke(query)

#     if not results:
#         print(f"⚠️ No results for query: \"{query}\" above threshold {score_threshold}. Falling back to top-{fallback_k} similarity.")
#         fallback_retriever = vectorstore.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": fallback_k}
#         )
#         results = fallback_retriever.invoke(query)

#     return results

def query_vecdb(
    query: str,
    vectorstore,
    top_k: int = 20            # grab a generous pool
) -> list:
    """
    Retrieve the top-K most-similar documents from the vector store
    (no similarity‐score threshold).  Returns a list of LangChain
    Document objects ordered by descending similarity.
    """
    return vectorstore.similarity_search_with_score(query=query, k=20)
    retriever = vectorstore.as_retriever(
        search_type="similarity",         # plain dense search
        search_kwargs={"k": top_k}        # top-K
    )
    results = retriever.invoke(query)     # returns List[Document]
    return results

# ===================== Optional Similarity Score Check =====================

def similarity_score(text1, text2, model, tokenizer=None):
    embedding_func = LocalEmbeddingFunction(embedd_model=model)
    embeddings1 = np.array(embedding_func.embed_query(query=text1)).reshape(1, -1)
    embeddings2 = np.array(embedding_func.embed_query(query=text2)).reshape(1, -1)

    return cosine_similarity(embeddings1, embeddings2)[0][0]

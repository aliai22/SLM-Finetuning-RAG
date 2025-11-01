from chromadb import Documents, EmbeddingFunction, Embeddings
from math import ceil
import torch, gc, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .dataset import batchify

# ---- LangChain v0.1.x vs v0.2+ compatibility shims ----
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain.vectorstores import Chroma  # older LC
try:
    from langchain.docstore.document import Document
except Exception:
    from langchain_core.documents import Document
# -------------------------------------------------------

# ===================== Embedding =====================

def generate_embeddings(model, text):
    if isinstance(text, str):
        return [model.encode(text).tolist()]
    return model.encode(text).tolist()

class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embedd_model):
        self.model = embedd_model
    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        # return a list-of-floats per item, Chroma will handle batching
        return generate_embeddings(model=self.model, text=input)[0]
    def embed_documents(self, texts):
        return generate_embeddings(model=self.model, text=texts)[0]
    def embed_query(self, query):
        return generate_embeddings(model=self.model, text=query)[0]

# ===================== Vector DB Creation =====================

def create_vecdb(path: str, dataset, embedding_function, batch_size=64, create_new=True):
    if create_new:
        vectorstore = None
        print("🆕 Creating a new Vector Database...")
        batches = batchify(dataset, batch_size)
        total_batches = ceil(len(dataset) / batch_size)
        for batch_idx, batch in enumerate(batches):
            print(f"🔄 Processing batch {batch_idx + 1}/{total_batches}...")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            if not all(batch):
                print(f"⚠️ Skipping empty batch {batch_idx + 1}")
                continue

            if vectorstore:
                batch_embeddings = [embedding_function(text) for text in batch]
                vectorstore.add_texts(texts=batch, embeddings=batch_embeddings)
            else:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_function,
                    persist_directory=path
                )

            vectorstore.persist()
            print(f"✅ Batch {batch_idx + 1} saved.")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

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

def query_vecdb(query, vectorstore, top_k=3, score_threshold=0.7, fallback_k=5):
    """
    Try LC retriever with threshold; if unavailable, fall back to similarity_search or generic search.
    Returns a list of Document objects.
    """
    # LC-style retriever with threshold
    if hasattr(vectorstore, "as_retriever"):
        try:
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": top_k}
            )
            results = retriever.invoke(query)
            if results:
                return results
            # fallback to plain similarity
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": fallback_k})
            return retriever.invoke(query)
        except Exception:
            pass

    # LC-like helper
    if hasattr(vectorstore, "similarity_search"):
        try:
            return vectorstore.similarity_search(query, k=fallback_k)
        except Exception:
            pass

    # generic store with .search() returning (id, text, score)
    if hasattr(vectorstore, "search"):
        hits = vectorstore.search(query, top_k=fallback_k)
        return [Document(page_content=t, metadata={"id": i, "score": s}) for i, t, s in hits]

    return []

# ===================== Optional Similarity Score Check =====================

def similarity_score(text1, text2, model, tokenizer=None):
    # tokenizer is unused; keep signature for compatibility
    emb = LocalEmbeddingFunction(embedd_model=model)
    e1 = np.array(emb.embed_query(query=text1)).reshape(1, -1)
    e2 = np.array(emb.embed_query(query=text2)).reshape(1, -1)
    return cosine_similarity(e1, e2)[0][0]

from chromadb import Documents, EmbeddingFunction, Embeddings
from math import ceil
import torch, gc, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llmrag.pipelines.ft_2rag.dataset import batchify

# ---- LangChain 0.1.x / 0.2+ shims ----
try:
    from langchain_community.vectorstores import Chroma
except Exception:
    from langchain.vectorstores import Chroma  # old

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.docstore.document import Document  # old
# --------------------------------------

# ===================== Embedding =====================

def generate_embeddings(model, text):
    # sentence-transformers: handles str or List[str]
    return model.encode(text).tolist()

class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embedd_model):
        self.model = embedd_model
    def __call__(self, input: Documents) -> Embeddings:
        return generate_embeddings(model=self.model, text=input)
    def embed_documents(self, texts):
        return generate_embeddings(model=self.model, text=texts)
    def embed_query(self, query):
        # return a single vector (no extra list)
        vec = generate_embeddings(model=self.model, text=query)
        return vec if isinstance(vec[0], float) else vec[0]

# ===================== Vector DB Creation =====================

def create_vecdb(path: str, dataset, embedding_function, batch_size=64, create_new=True):
    """
    Creates or updates a Chroma vector database with batch processing.
    dataset: list[str] (QA texts) or list[Document]
    """
    if create_new:
        vectorstore = None
        print("🆕 Creating a new Vector Database...")
        total_batches = ceil(len(dataset) / batch_size)

        global_idx = 0
        for batch_idx in range(total_batches):
            batch_raw = dataset[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            print(f"🔄 Processing batch {batch_idx + 1}/{total_batches}...")

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            # ensure Documents with a doc_id
            batch_docs = []
            for text in batch_raw:
                if isinstance(text, Document):
                    doc = text
                    if "doc_id" not in doc.metadata:
                        doc.metadata["doc_id"] = f"doc_{global_idx}"
                else:
                    doc = Document(page_content=text, metadata={"doc_id": f"doc_{global_idx}"})
                batch_docs.append(doc)
                global_idx += 1

            if not batch_docs:
                print(f"⚠️ Skipping empty batch {batch_idx + 1}")
                continue

            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch_docs,
                    embedding=embedding_function,
                    persist_directory=path
                )
            else:
                batch_embs = [embedding_function(d.page_content) for d in batch_docs]
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

def query_vecdb(query: str, vectorstore, top_k: int = 20) -> list:
    """
    Returns a list of (Document, score) pairs when available,
    otherwise falls back to plain Documents with dummy scores.
    """
    # prefer Chroma's similarity_search_with_score if present
    if hasattr(vectorstore, "similarity_search_with_score"):
        try:
            results = vectorstore.similarity_search_with_score(query=query, k=top_k)
            # already List[(Document, score)]
            return results
        except Exception:
            pass

    # LangChain retriever path
    if hasattr(vectorstore, "as_retriever"):
        try:
            ret = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            docs = ret.invoke(query)
            return [(d, 1.0) for d in docs]  # dummy scores
        except Exception:
            pass

    # last resort
    if hasattr(vectorstore, "similarity_search"):
        try:
            docs = vectorstore.similarity_search(query, k=top_k)
            return [(d, 1.0) for d in docs]
        except Exception:
            pass

    return []

# ===================== Optional Similarity Score Check =====================

def similarity_score(text1, text2, model, tokenizer=None):
    embedding_func = LocalEmbeddingFunction(embedd_model=model)
    e1 = np.array(embedding_func.embed_query(query=text1)).reshape(1, -1)
    e2 = np.array(embedding_func.embed_query(query=text2)).reshape(1, -1)
    return cosine_similarity(e1, e2)[0][0]

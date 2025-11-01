from chromadb import Documents, EmbeddingFunction, Embeddings
from math import ceil
import torch, gc, numpy as np, os, re, unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from .dataset import batchify
from typing import List
from PyPDF2 import PdfReader

# ---- LangChain 0.1.x / 0.2+ shims ----
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # old

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
    # sentence-transformers models: .encode handles str or List[str]
    return model.encode(text).tolist()

class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, embedd_model):
        self.model = embedd_model
    def __call__(self, input: Documents) -> Embeddings:
        return generate_embeddings(model=self.model, text=input)
    def embed_documents(self, texts):
        return generate_embeddings(model=self.model, text=texts)
    def embed_query(self, query):
        return generate_embeddings(model=self.model, text=query)

# ===================== PDF / cleaning =====================

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "--",
        "\uf0b7": "-", "\xa0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[\d\W_]{5,}$", line):
            continue
        if re.search(r"[\ud800-\udfff]", line):
            continue
        clean_lines.append(line)
    return " ".join(clean_lines)

def extract_text_by_page(pdf_path: str, pdf_config: dict, config_index: int) -> list:
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
        lines = page_text.splitlines(True)
        if header_lines > 0:
            lines = lines[header_lines:]
        if footer_lines > 0:
            lines = lines[:-footer_lines]
        joined_text = "".join(lines)
        cleaned = clean_text(joined_text)
        if cleaned:
            raw_pages.append({"text": cleaned, "source": filename, "page": page_num + 1})
    return raw_pages

def chunk_documents(dataset: List[dict], chunk_size=512, chunk_overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunked_docs = []
    for item in dataset:
        for chunk in splitter.split_text(item["text"]):
            chunked_docs.append(
                Document(page_content=chunk, metadata={"source": item["source"], "page": item["page"]})
            )
    return chunked_docs

def _clean_utf(text):
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")

# ===================== Vector DB =====================

def create_vecdb(path: str, documents: List[Document], embedding_function, batch_size=32, create_new=True):
    if create_new:
        vectorstore = None
        print("Creating a new Vector Database...")
        batches = list(batchify(documents, batch_size))
        total_batches = len(batches)
        for idx, batch in enumerate(batches):
            print(f"Processing batch {idx + 1}/{total_batches}...")
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            if vectorstore:
                for doc in batch:
                    content = doc.page_content
                    if not isinstance(content, str):
                        continue
                    content = _clean_utf(content)
                    if not content.strip():
                        continue
                    try:
                        emb = embedding_function(content)
                        vectorstore.add_texts([content], embeddings=[emb], metadatas=[doc.metadata])
                    except Exception as e:
                        print("Embedding failed:", str(e))
                        continue
            else:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_function,
                    persist_directory=path
                )

            vectorstore.persist()
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        print("✅ Vector DB created and saved at:", path)
        return vectorstore
    else:
        vectorstore = Chroma(persist_directory=path, embedding_function=embedding_function)
        print("Loaded existing Vector DB.")
        return vectorstore

def process_pdf_to_vecdb(pdf_text_by_page: List, db_path: str, embedding_function, batch_size: int = 32, create_new=True):
    chunked_docs = chunk_documents(pdf_text_by_page)
    vector_store = create_vecdb(db_path, chunked_docs, embedding_function, batch_size=batch_size, create_new=create_new)
    return vector_store

# ===================== Query =====================

def query_vecdb(query: str, vectorstore, k: int = 5, rerank_top_k: int = 3) -> List[Document]:
    # Prefer LC retriever if available
    if hasattr(vectorstore, "as_retriever"):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        try:
            return retriever.invoke(query) or []
        except Exception:
            pass

    # LC-like helper
    if hasattr(vectorstore, "similarity_search"):
        try:
            return vectorstore.similarity_search(query, k=k)
        except Exception:
            pass

    # Fallback: generic interface that returns (id, text, score)
    if hasattr(vectorstore, "search"):
        hits = vectorstore.search(query, top_k=k)
        return [Document(page_content=t, metadata={"id": i, "score": s}) for i, t, s in hits]

    return []

# ===================== Similarity (optional) =====================

def similarity_score(text1, text2, model, tokenizer=None):
    emb = LocalEmbeddingFunction(embedd_model=model)
    e1 = np.array(emb.embed_query(query=text1)).reshape(1, -1)
    e2 = np.array(emb.embed_query(query=text2)).reshape(1, -1)
    return cosine_similarity(e1, e2)[0][0]

# build_gold_doc_ids.py
import json, os
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from finetuning_embeddModel.embedd_finetuning import load_model
from rag_eval_FT2.eval_dataset import prepare_rag_evaluation_data
from FT_2RAG.vectorstore import LocalEmbeddingFunction

# ---------- paths ----------
TEST_FILE       = "synthetic_QAs.json"          # your test set
VECDB_DIR       = "./FT_2RAG/vecDB/QAs_v3"              # Chroma dir with QA docs
EMB_MODEL_PATH  = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
OUT_PATH        = "gold_doc_ids.json"
TOP_N           = 1                        # keep the nearest doc

# ---------- load embedder ----------
# embedder = SentenceTransformer(EMB_MODEL_PATH)
embedd_model = load_model(model_id=EMB_MODEL_PATH, eval=True)

# Load evaluation dataset
processed_dataset = prepare_rag_evaluation_data(file_path=TEST_FILE)
print(f"Total QA Pairs: {len(processed_dataset)}")
# print(processed_dataset[0])
# ---------- connect to Chroma ----------
emf = LocalEmbeddingFunction(embedd_model=embedd_model
                                )
vec_db = Chroma(
    persist_directory=VECDB_DIR,
    embedding_function=emf
)

gold_map = {}
for idx, sample in tqdm(enumerate(processed_dataset), total=len(processed_dataset),
                        desc="Building gold IDs"):
    sid    = str(idx)                  # use list position as ID
    answer = sample["answer"]

    # ans_emb = embedd_model.encode(
    #     [answer], convert_to_tensor=True, normalize_embeddings=True
    # )

    hits = vec_db.similarity_search_with_score(answer, k=TOP_N)
    gold_ids = [doc.metadata["doc_id"] for doc, _ in hits]
    gold_map[sid] = gold_ids

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(gold_map, f, ensure_ascii=False, indent=2)

print(f"✔ Saved {len(gold_map)} gold IDs to {OUT_PATH}")
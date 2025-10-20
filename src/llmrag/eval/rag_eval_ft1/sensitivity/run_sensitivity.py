import os, csv, random, math
from typing import Dict, List, Tuple
from .perturbations import make_variants
from .prompts import PROMPTS, DEFAULT_PROMPT_ID
from ..eval_dataset import prepare_rag_evaluation_data
from ..evaluator import evaluate_rag
from finetuning_embeddModel.embedd_finetuning import load_model
from FT_1RAG.rag_main import run_rag_pipeline  # expects (query, **kwargs)

# ---- Config ----
EVAL_JSON = "synthetic_QAs.json"
OUT_CSV   = "rag_sensitivity_results_FT1.csv"
N_QUEST   = 100          # sample size
TEMP      = 0.1          # fixed for query/prompt sensitivity
SEED      = 42

# Optional knobs (only if your RAG allows them; otherwise left unused)
RETRIEVAL_K = 5
PROMPT_IDS  = ["strict_v1","strict_v2","soft_v1","soft_v2"]

random.seed(SEED)

# ---- load data + embedder ----
dataset = prepare_rag_evaluation_data(EVAL_JSON)
random.shuffle(dataset)
dataset = dataset[:N_QUEST]

embedd_ftmodel_ckpt = "./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0"
embedd_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

# ---- helpers ----

def call_rag(query:str, prompt_id:str=DEFAULT_PROMPT_ID, temperature:float=TEMP,
             k:int=RETRIEVAL_K, ctx_order:str="as_is") -> Tuple[str, Dict]:
    """
    Wrapper so we can pass prompt template & retrieval hints if your pipeline supports them.
    `meta` should at least include 'ctx_ids' if available for Evidence Stability later.
    """
    kwargs = {}
    # If your run_rag_pipeline accepts these keys, they'll be used.
    kwargs.update({
        "prompt_id": prompt_id,        # needs small edit in your rag_main (see note below)
        "temperature": temperature,    # pass through to generator
        "top_k": k,                    # retrieval depth
        "ctx_order": ctx_order,        # 'as_is' or 'shuffled' (optional)
    })
    try:
        answer, meta = run_rag_pipeline(query=query, **kwargs)
    except TypeError:
        # Fallback if your current signature doesn't take our extras:
        answer, meta = run_rag_pipeline(query=query)
    if meta is None: meta = {}
    return answer, meta

def write_header(path: str, fields: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=fields,
                quoting=csv.QUOTE_ALL,   # always quote
                escapechar="\\",         # escape embedded quotes/newlines
                doublequote=True,
                lineterminator="\n",
            )
            w.writeheader()

FIELDS = [
    "q_id","family","variant","severity","prompt_id","temperature",
    "k","ctx_order","question_text","ground_truth",
    "generated_answer","em","f1","bleu","rouge1","rougeL",
    "embedding_similarity","baseline_metric","rpd","cr_vs_baseline",
]

def _scrub_row(row: dict) -> dict:
    cleaned = {}
    for k in FIELDS:
        v = row.get(k, "")
        # normalize NaN/inf
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                v = 0.0
        # stringify non-scalars (lists/dicts/None/etc.)
        if not isinstance(v, (str, int, float)):
            v = str(v)
        cleaned[k] = v
    return cleaned

write_header(OUT_CSV, FIELDS)

# ---- main loops ----
rows = []

for q_id, item in enumerate(dataset, start=1):
    q = item["question"]; gold = item["answer"]
    variants = make_variants(q)

    # --- Baseline (query sensitivity baseline) ---
    base_ans, base_meta = call_rag(q, prompt_id=DEFAULT_PROMPT_ID, temperature=TEMP, k=RETRIEVAL_K)
    base_eval = evaluate_rag(q, gold, base_ans, embedd_model)
    base_score = base_eval["embedding_similarity"]  # choose one baseline metric for RPD (you can swap)
    base_text  = base_eval["generated_answer"]

    # Query sensitivity: vary query with fixed prompt/settings
    for name, v in variants.items():
        ans, meta = call_rag(v["text"], prompt_id=DEFAULT_PROMPT_ID, temperature=TEMP, k=RETRIEVAL_K)
        ev = evaluate_rag(q, gold, ans, embedd_model)
        # RPD vs baseline (use same metric — embedding_similarity here)
        rpd = (base_score - ev["embedding_similarity"]) / (base_score + 1e-8)
        # Consistency Rate at the single-pair level becomes a boolean; aggregate later
        cr = 1.0 if ev["embedding_similarity"] >= 0.85 and \
                    evaluate_rag(q, base_text, ans, embedd_model)["embedding_similarity"] >= 0.85 else 0.0

        rows.append({
            "q_id": q_id, "family": "query", "variant": name,
            "severity": v["severity"], "prompt_id": DEFAULT_PROMPT_ID,
            "temperature": TEMP, "k": RETRIEVAL_K, "ctx_order": "as_is",
            "question_text": v["text"], "ground_truth": gold,
            "generated_answer": ev["generated_answer"], "em": ev["em"], "f1": ev["f1"],
            "bleu": ev["bleu"], "rouge1": ev["rouge1"], "rougeL": ev["rougeL"],
            "embedding_similarity": ev["embedding_similarity"],
            "baseline_metric": base_score, "rpd": float(rpd), "cr_vs_baseline": float(cr),
        })

    # Prompt sensitivity: fix original query, vary prompt template
    for pid in PROMPT_IDS:
        ans, meta = call_rag(q, prompt_id=pid, temperature=TEMP, k=RETRIEVAL_K)
        ev = evaluate_rag(q, gold, ans, embedd_model)
        rpd = (base_score - ev["embedding_similarity"]) / (base_score + 1e-8)
        cr  = 1.0 if evaluate_rag(q, base_text, ans, embedd_model)["embedding_similarity"] >= 0.85 else 0.0

        rows.append({
            "q_id": q_id, "family": "prompt", "variant": pid, "severity": "n/a",
            "prompt_id": pid, "temperature": TEMP, "k": RETRIEVAL_K, "ctx_order": "as_is",
            "question_text": q, "ground_truth": gold,
            "generated_answer": ev["generated_answer"], "em": ev["em"], "f1": ev["f1"],
            "bleu": ev["bleu"], "rouge1": ev["rouge1"], "rougeL": ev["rougeL"],
            "embedding_similarity": ev["embedding_similarity"],
            "baseline_metric": base_score, "rpd": float(rpd), "cr_vs_baseline": float(cr),
        })

    # (Optional) Retrieval sensitivity — only if your pipeline accepts k/order
    # for cfg in [("k3",3,"as_is"),("k5",5,"as_is"),("k7",7,"as_is"),("shuffle",RETRIEVAL_K,"shuffled")]:
    #     tag,k,order = cfg
    #     ans, meta = call_rag(q, prompt_id=DEFAULT_PROMPT_ID, temperature=TEMP, k=k, ctx_order=order)
    #     ev = evaluate_rag(q, gold, ans, embedd_model)
    #     rpd = (base_score - ev["embedding_similarity"]) / (base_score + 1e-8)
    #     cr  = 1.0 if evaluate_rag(q, base_text, ans, embedd_model)["embedding_similarity"] >= 0.85 else 0.0
    #     rows.append({... as above with family='retrieval', variant=tag ...})

# ---- write CSV incrementally (stream-safe for long runs) ----
with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=FIELDS,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        doublequote=True,
        lineterminator="\n",
    )
    for r in rows:
        w.writerow(_scrub_row(r))

# ---- quick aggregate printout ----
def _avg(xs): return sum(xs)/max(1,len(xs))
def _group(rows, key): 
    d={}; 
    for r in rows: d.setdefault(r[key], []).append(r); 
    return d

print("\n=== Sensitivity Summary ===")
for fam, fam_rows in _group(rows, "family").items():
    by_var = _group(fam_rows, "variant")
    print(f"\n[{fam}]")
    for vname, rr in by_var.items():
        mean_rpd = _avg([r["rpd"] for r in rr])
        cr_rate  = _avg([r["cr_vs_baseline"] for r in rr])
        print(f"  {vname:10s}  Mean RPD: {mean_rpd:.3f}  CR: {cr_rate:.3f}  (n={len(rr)})")

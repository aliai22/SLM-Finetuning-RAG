import os, csv, random, math
from typing import Dict, List, Tuple, Iterable, Any

from .perturbations import make_variants
from .prompts import PROMPTS, DEFAULT_PROMPT_ID
from llmrag.eval.common.datasets import prepare_rag_evaluation_data
from llmrag.eval.common.runners import evaluate_rag
from llmrag.training.embedding.embedd_finetuning import load_model
# Default pipeline for full runs (still FT-1 RAG by default)
from llmrag.pipelines.ft_1rag.rag_main import run_rag_pipeline as _default_run_rag

# ---- Config (kept as you had) ----
EVAL_JSON = ".data/finetuning/qa/synthetic_QAs.json"
OUT_CSV   = ".artifacts/logs/rag_sensitivity_results_FT1.csv"
N_QUEST   = 100
TEMP      = 0.1
SEED      = 42

# Optional knobs (only used if your pipeline supports them)
RETRIEVAL_K = 5
PROMPT_IDS  = ["strict_v1","strict_v2","soft_v1","soft_v2"]

random.seed(SEED)

FIELDS = [
    "q_id","family","variant","severity","prompt_id","temperature",
    "k","ctx_order","question_text","ground_truth",
    "generated_answer","em","f1","bleu","rouge1","rougeL",
    "embedding_similarity","baseline_metric","rpd","cr_vs_baseline",
]

def write_header(path: str, fields: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=fields,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                doublequote=True,
                lineterminator="\n",
            )
            w.writeheader()

def _scrub_row(row: dict) -> dict:
    cleaned = {}
    for k in FIELDS:
        v = row.get(k, "")
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            v = 0.0
        if not isinstance(v, (str, int, float)):
            v = str(v)
        cleaned[k] = v
    return cleaned

def _avg(xs): return sum(xs)/max(1,len(xs))
def _group(rows, key):
    d={}
    for r in rows: d.setdefault(r[key], []).append(r)
    return d

def _as_qa_dict(x: Any) -> Dict[str,str]:
    """Accept dict/list/tuple/namedtuple-ish and normalize to {'question','answer'} strings."""
    if isinstance(x, dict) and "question" in x and "answer" in x:
        return {"question": str(x["question"]), "answer": str(x["answer"])}
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return {"question": str(x[0]), "answer": str(x[1])}
    if hasattr(x, "question") and hasattr(x, "answer"):
        return {"question": str(x.question), "answer": str(x.answer)}
    raise TypeError(f"Unsupported dataset row format: {x!r}")

# ---------------------------
# Smoke-friendly entry point
# ---------------------------
def smoke_entry(
    dataset_rows: Iterable[Any],
    call_rag_func,
    similarity_model,
    out_csv_path: str,
    n_quest: int = 3,
    prompt_ids: List[str] = None,
    temperature: float = 0.1,
    retrieval_k: int = 5,
    ctx_order: str = "as_is",
):
    """
    Lightweight harness for smoke tests. Inject a tiny dataset and a fake/real call_rag,
    and write a small CSV to confirm the loop wiring works.
    """
    if prompt_ids is None:
        prompt_ids = ["strict_v1"]

    rows: List[dict] = []
    write_header(out_csv_path, FIELDS)

    ds = list(dataset_rows)[:n_quest]

    for q_id, raw in enumerate(ds, start=1):
        item = _as_qa_dict(raw)
        q = item["question"]; gold = item["answer"]

        # Baseline
        base_ans, _ = call_rag_func(q, prompt_id=DEFAULT_PROMPT_ID, temperature=temperature, k=retrieval_k, ctx_order=ctx_order)
        base_eval = evaluate_rag(q, gold, base_ans, similarity_model)
        base_score = base_eval["embedding_similarity"]
        base_text  = base_eval["generated_answer"]

        # Query variants
        for name, v in make_variants(q).items():
            ans, _ = call_rag_func(v["text"], prompt_id=DEFAULT_PROMPT_ID, temperature=temperature, k=retrieval_k, ctx_order=ctx_order)
            ev = evaluate_rag(q, gold, ans, similarity_model)
            rpd = (base_score - ev["embedding_similarity"]) / (base_score + 1e-8)
            cr  = 1.0 if evaluate_rag(q, base_text, ans, similarity_model)["embedding_similarity"] >= 0.85 else 0.0

            rows.append({
                "q_id": q_id, "family": "query", "variant": name,
                "severity": v["severity"], "prompt_id": DEFAULT_PROMPT_ID,
                "temperature": temperature, "k": retrieval_k, "ctx_order": ctx_order,
                "question_text": v["text"], "ground_truth": gold,
                "generated_answer": ev["generated_answer"], "em": ev["em"], "f1": ev["f1"],
                "bleu": ev["bleu"], "rouge1": ev["rouge1"], "rougeL": ev["rougeL"],
                "embedding_similarity": ev["embedding_similarity"],
                "baseline_metric": base_score, "rpd": float(rpd), "cr_vs_baseline": float(cr),
            })

        # Prompt variants
        for pid in prompt_ids:
            ans, _ = call_rag_func(q, prompt_id=pid, temperature=temperature, k=retrieval_k, ctx_order=ctx_order)
            ev = evaluate_rag(q, gold, ans, similarity_model)
            rpd = (base_score - ev["embedding_similarity"]) / (base_score + 1e-8)
            cr  = 1.0 if evaluate_rag(q, base_text, ans, similarity_model)["embedding_similarity"] >= 0.85 else 0.0

            rows.append({
                "q_id": q_id, "family": "prompt", "variant": pid, "severity": "n/a",
                "prompt_id": pid, "temperature": temperature, "k": retrieval_k, "ctx_order": ctx_order,
                "question_text": q, "ground_truth": gold,
                "generated_answer": ev["generated_answer"], "em": ev["em"], "f1": ev["f1"],
                "bleu": ev["bleu"], "rouge1": ev["rouge1"], "rougeL": ev["rougeL"],
                "embedding_similarity": ev["embedding_similarity"],
                "baseline_metric": base_score, "rpd": float(rpd), "cr_vs_baseline": float(cr),
            })

    # write CSV
    with open(out_csv_path, "a", newline="", encoding="utf-8") as f:
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

    print(f"[SMOKE] Wrote {len(rows)} rows to {out_csv_path}")
    return rows

# ---------------------------
# Full run entry point (CLI)
# ---------------------------
def main():
    """
    Original behavior, now inside a function so importing this module won’t execute.
    Uses FT-1 RAG by default; change _default_run_rag import above to target another pipeline.
    """
    # Load data
    dataset = prepare_rag_evaluation_data(EVAL_JSON)
    random.shuffle(dataset)
    dataset = dataset[:N_QUEST]

    # Similarity model for metrics (your original FT embedder)
    embedd_ftmodel_ckpt = ".models/emb_model"
    similarity_model = load_model(model_id=embedd_ftmodel_ckpt, eval=True)

    # Runner wrapper (defaults to FT-1 pipeline but keeps your kwargs passthrough)
    def call_rag(query: str,
                 prompt_id: str = DEFAULT_PROMPT_ID,
                 temperature: float = TEMP,
                 k: int = RETRIEVAL_K,
                 ctx_order: str = "as_is") -> Tuple[str, Dict]:
        kwargs = {
            "prompt_id": prompt_id,
            "temperature": temperature,
            "top_k": k,
            "ctx_order": ctx_order,
        }
        try:
            answer, meta = _default_run_rag(query=query, **kwargs)
        except TypeError:
            answer, meta = _default_run_rag(query=query)
        return answer, (meta or {})

    write_header(OUT_CSV, FIELDS)
    rows: List[dict] = []

    for q_id, item in enumerate(dataset, start=1):
        qa = _as_qa_dict(item)
        q = qa["question"]; gold = qa["answer"]
        variants = make_variants(q)

        # Baseline
        base_ans, base_meta = call_rag(q, prompt_id=DEFAULT_PROMPT_ID, temperature=TEMP, k=RETRIEVAL_K)
        base_eval = evaluate_rag(q, gold, base_ans, similarity_model)
        base_score = base_eval["embedding_similarity"]
        base_text  = base_eval["generated_answer"]

        # Query sensitivity
        for name, v in variants.items():
            ans, meta = call_rag(v["text"], prompt_id=DEFAULT_PROMPT_ID, temperature=TEMP, k=RETRIEVAL_K)
            ev = evaluate_rag(q, gold, ans, similarity_model)
            rpd = (base_score - ev["embedding_similarity"]) / (base_score + 1e-8)
            cr  = 1.0 if ev["embedding_similarity"] >= 0.85 and \
                        evaluate_rag(q, base_text, ans, similarity_model)["embedding_similarity"] >= 0.85 else 0.0

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

        # Prompt sensitivity
        for pid in PROMPT_IDS:
            ans, meta = call_rag(q, prompt_id=pid, temperature=TEMP, k=RETRIEVAL_K)
            ev = evaluate_rag(q, gold, ans, similarity_model)
            rpd = (base_score - ev["embedding_similarity"]) / (base_score + 1e-8)
            cr  = 1.0 if evaluate_rag(q, base_text, ans, similarity_model)["embedding_similarity"] >= 0.85 else 0.0

            rows.append({
                "q_id": q_id, "family": "prompt", "variant": pid, "severity": "n/a",
                "prompt_id": pid, "temperature": TEMP, "k": RETRIEVAL_K, "ctx_order": "as_is",
                "question_text": q, "ground_truth": gold,
                "generated_answer": ev["generated_answer"], "em": ev["em"], "f1": ev["f1"],
                "bleu": ev["bleu"], "rouge1": ev["rouge1"], "rougeL": ev["rougeL"],
                "embedding_similarity": ev["embedding_similarity"],
                "baseline_metric": base_score, "rpd": float(rpd), "cr_vs_baseline": float(cr),
            })

    # Write CSV
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

    # Summary
    print("\n=== Sensitivity Summary ===")
    for fam, fam_rows in _group(rows, "family").items():
        by_var = _group(fam_rows, "variant")
        print(f"\n[{fam}]")
        for vname, rr in by_var.items():
            mean_rpd = _avg([r["rpd"] for r in rr])
            cr_rate  = _avg([r["cr_vs_baseline"] for r in rr])
            print(f"  {vname:10s}  Mean RPD: {mean_rpd:.3f}  CR: {cr_rate:.3f}  (n={len(rr)})")

if __name__ == "__main__":
    main()

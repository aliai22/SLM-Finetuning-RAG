#!/usr/bin/env python3
import os, json, time, argparse, statistics as stats
import psutil
import torch

# ---- Project imports (FT1) ----
from LLM_Model import load_local_llm
from FT_2RAG.finetuned_model import load_ft_model
from finetuning_embeddModel.embedd_finetuning import load_model as load_embedder
from FT_2RAG.chatbot import rag_chatbot
from FT_1RAG.vectorstore import LocalEmbeddingFunction
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer

# ----------------- helpers -----------------
def human(n):
    if n is None: return "unknown"
    if n >= 1e12: return f"{n/1e12:.2f}T"
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.2f}K"
    return str(n)

def count_jsonl_tokens(jsonl_path, tokenizer, q_keys=("question","input"), a_keys=("answer","output")):
    total = 0; seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            q = next((j[k] for k in q_keys if isinstance(j.get(k), str)), "")
            a = next((j[k] for k in a_keys if isinstance(j.get(k), str)), "")
            txt = (q + "\n" + a).strip()
            if not txt: continue
            ids = tokenizer(txt, add_special_tokens=False, truncation=False).input_ids
            total += len(ids); seen += 1
    return total, seen

def model_param_count(model):
    try: return sum(p.numel() for p in model.parameters())
    except Exception: return None

def compute_train_flops(n_params, n_tokens):
    # Rule of thumb for dense Transformers (train): ~6 * params * tokens
    return 6 * n_params * n_tokens if n_params and n_tokens else None

def bench(fn, runs=10, warmup=3):
    for _ in range(warmup): fn()
    ts=[]
    for _ in range(runs):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000.0)
    med = stats.median(ts)
    mad = stats.median([abs(x-med) for x in ts]) if len(ts)>1 else 0.0
    qps = 1000.0/med if med>0 else 0.0
    return med, mad, qps

def gpu_info():
    if not torch.cuda.is_available(): return {"available": False}
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    name = torch.cuda.get_device_name(i)
    total_gb = props.total_memory/(1024**3)
    try:
        free, total = torch.cuda.mem_get_info()
        free_gb = free/(1024**3); used_gb = total_gb - free_gb
    except Exception:
        free_gb = used_gb = None
    return {"available": True, "name": name, "total_gb": total_gb, "free_gb": free_gb, "used_gb": used_gb}

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slm_tokenizer_path", type=str, default="./LLMs")
    ap.add_argument("--slm_ft_ckpt", type=str, default="./Finetuning_Checkpoints_filtered/final-checkpoint/checkpoint-9750")
    ap.add_argument("--embed_model_id", type=str, default="./finetuning_embeddModel/bge-base-en-v1.5-matryoshka2.0")
    ap.add_argument("--slm_jsonl", type=str, default="uniqueQA_dataset0.7.jsonl")
    ap.add_argument("--embed_jsonl", type=str, default="uniqueQA_dataset0.7.jsonl")
    ap.add_argument("--ft1_db_path", type=str, default="./FT_1RAG/vecDB/textbooks_v2")
    ap.add_argument("--query", type=str, default="What is a convolutional neural network?")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    # Optional knobs passed to generator (if your prompt wiring uses them):
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.95)
    args = ap.parse_args()

    # Hardware banner
    gi = gpu_info()
    if gi["available"]:
        print(f"GPU: {gi['name']} | Total {gi['total_gb']:.1f} GB | Used ~{(gi['used_gb'] or 0):.1f} GB | Free ~{(gi['free_gb'] or 0):.1f} GB")
    else:
        print("GPU: not available (CPU run)")

    # ---- Load SLM & tokenizer (once) ----
    print("\n== Loading SLM & tokenizer ==")
    base_model, tokenizer, eval_tokenizer = load_local_llm(args.slm_tokenizer_path)
    ft_model = load_ft_model(base_model=base_model, ft_ckpt=args.slm_ft_ckpt)
    # Move to GPU if possible
    if torch.cuda.is_available():
        ft_model.to("cuda")
    n_slm_params = model_param_count(ft_model) or model_param_count(base_model)
    print(f"SLM params (approx): {human(n_slm_params)}")

    # ---- Load embedder & vector DB (once) ----
    print("\n== Loading embedding model & FT1 vector DB ==")
    embed_model = load_embedder(model_id=args.embed_model_id, eval=True)
    emf = LocalEmbeddingFunction(embedd_model=embed_model)
    # Reopen existing Chroma DB without rebuilding
    vec_db = Chroma(persist_directory=args.ft1_db_path, embedding_function=emf)

    n_embed_params = model_param_count(embed_model)
    print(f"Embedding params (approx): {human(n_embed_params)}")

    # ---- Token counts (training) ----
    print("\n== Token counts (training) ==")
    slm_tokens, slm_seen = count_jsonl_tokens(args.slm_jsonl, eval_tokenizer)
    emb_tokens, emb_seen = count_jsonl_tokens(args.embed_jsonl, eval_tokenizer)
    print(f"SLM tokens seen: {slm_tokens:,} (records: {slm_seen})")
    print(f"Embedding tokens seen: {emb_tokens:,} (records: {emb_seen})")

    # ---- Compute FLOPs ----
    print("\n== Approx training compute ==")
    slm_flops = compute_train_flops(n_slm_params, slm_tokens)
    emb_flops = compute_train_flops(n_embed_params, emb_tokens)
    if slm_flops is not None:
        print(f"SLM train FLOPs ≈ {slm_flops/1e15:.2f} PFLOPs")
    else:
        print("SLM train FLOPs: unknown")
    if emb_flops is not None:
        unit = "TFLOPs" if emb_flops < 1e15 else "PFLOPs"
        denom = 1e12 if unit == "TFLOPs" else 1e15
        print(f"Embedding train FLOPs ≈ {emb_flops/denom:.2f} {unit}")
    else:
        print("Embedding train FLOPs: unknown")

    # ---- Inference benchmarks (preloaded) ----
    print("\n== Inference benchmarks (preloaded models/DB) ==")

    # RAG: call rag_chatbot directly with preloaded ft_model/eval_tokenizer/vec_db
    def rag_once():
        with torch.inference_mode():
            _resp, _ctx = rag_chatbot(
                user_query=args.query,
                vec_db=vec_db,
                model=ft_model,
                tokenizer=eval_tokenizer
            )

    rag_ms, rag_mad, rag_qps = bench(rag_once, runs=args.runs, warmup=2)
    print(f"RAG (FT1) latency: {rag_ms:.0f} ms  (±{rag_mad:.0f}) | Throughput: {rag_qps:.2f} QPS")

    # Gen-only: bypass retrieval and call HF .generate()
    def gen_only_once():
        prompt = f"You are a helpful assistant.\n\nQuestion: {args.query}\nAnswer:"
        inputs = eval_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            _out = ft_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=eval_tokenizer.eos_token_id
            )

    gen_ms, gen_mad, gen_qps = bench(gen_only_once, runs=args.runs, warmup=2)
    print(f"Gen-only latency: {gen_ms:.0f} ms  (±{gen_mad:.0f}) | Throughput: {gen_qps:.2f} QPS")

    # ---- Memory snapshot ----
    print("\n== Memory footprint ==")
    if torch.cuda.is_available():
        print(f"GPU alloc (MB): {torch.cuda.memory_allocated()/1e6:.1f}")
        print(f"GPU reserved (MB): {torch.cuda.memory_reserved()/1e6:.1f}")
    print(f"CPU RSS (MB): {psutil.Process(os.getpid()).memory_info().rss/1e6:.1f}")

    print("\nDone.")

if __name__ == "__main__":
    main()

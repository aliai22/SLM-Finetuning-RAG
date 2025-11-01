import argparse, csv, os
from llmrag.eval.common.datasets import prepare_rag_evaluation_data
from llmrag.eval.common.runners import evaluate_qa
from llmrag.pipelines.ft_1rag.rag_main import run_rag_pipeline

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", default=".data/finetuning/qa/synthetic_QAs.json")
    p.add_argument("--out-csv", default=".artifacts/logs/rag_eval_results_ft1.csv")
    p.add_argument("--max-examples", type=int, default=None)
    args = p.parse_args(argv)

    qa = prepare_rag_evaluation_data(file_path=args.eval_json)
    print(f"[ft1] Total QA Pairs: {len(qa)}")

    def rag_fn(q):
        ans, _ctx = run_rag_pipeline(query=q)
        return ans, _ctx

    metrics = evaluate_qa(rag_fn, qa, max_examples=args.max_examples)
    print("[ft1] metrics:", metrics)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n","em","f1","rougeL","bleu"])
        w.writeheader(); w.writerow(metrics)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

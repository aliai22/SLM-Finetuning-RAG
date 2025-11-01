from typing import List, Tuple, Dict
import re, string
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import util  # <-- needed for embedding similarity

# ---------- normalization & metrics ----------
def normalize_answer(s: str) -> str:
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space_fix(t): return " ".join(t.split())
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    def lower(t): return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec  = len(common) / len(truth_tokens)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def rouge_l(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    return scorer.score(ref, pred)['rougeL'].fmeasure

def rouge1(pred: str, ref: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    return scorer.score(ref, pred)['rouge1'].fmeasure

def bleu_score(pred: str, ref: str) -> float:
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)

# ---------- single-example evaluator (compat with old API) ----------
def evaluate_rag(question: str, ground_truth: str, rag_response: str, similarity_model) -> Dict:
    """
    Matches the old evaluator.evaluate_rag() contract and keys.
    similarity_model: a SentenceTransformer (or compatible) with .encode()
    """
    pred = rag_response or ""
    # BLEU / ROUGE
    bleu = bleu_score(pred, ground_truth)
    r1   = rouge1(pred, ground_truth)
    rL   = rouge_l(pred, ground_truth)

    # Embedding similarity
    emb1 = similarity_model.encode(ground_truth, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = similarity_model.encode(pred,         convert_to_tensor=True, normalize_embeddings=True)
    emb_sim = util.pytorch_cos_sim(emb1, emb2).item()

    # EM / F1
    em = int(exact_match_score(pred, ground_truth))
    f1 = f1_score(pred, ground_truth)

    return {
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": pred,
        "normalized_prediction": normalize_answer(pred),
        "em": em,
        "f1": f1,
        "bleu": bleu,
        "rouge1": r1,
        "rougeL": rL,
        "embedding_similarity": emb_sim,
    }

# ---------- multi-example runner ----------
def evaluate_qa(rag_fn, qa_pairs: List[Tuple[str, str]], max_examples: int = None) -> Dict:
    ems, f1s, rouges, bleus = [], [], [], []
    n = 0
    for q, gold in qa_pairs:
        if max_examples is not None and n >= max_examples:
            break
        pred, _ctx = rag_fn(q)
        pred = pred or ""
        ems.append(exact_match_score(pred, gold))
        f1s.append(f1_score(pred, gold))
        rouges.append(rouge_l(pred, gold))
        bleus.append(bleu_score(pred, gold))
        n += 1
    def avg(xs): return sum(xs)/len(xs) if xs else 0.0
    return {"n": n, "em": avg(ems), "f1": avg(f1s), "rougeL": avg(rouges), "bleu": avg(bleus)}

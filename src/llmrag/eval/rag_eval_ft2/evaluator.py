import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_em(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_rag(question, ground_truth, rag_response, similarity_model):
    # --- BLEU ---
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([ground_truth.split()], rag_response.split(), smoothing_function=smoothie)

    # --- ROUGE ---
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(ground_truth, rag_response)

    # --- Embedding Similarity ---
    emb1 = similarity_model.encode(ground_truth, convert_to_tensor=True)
    emb2 = similarity_model.encode(rag_response, convert_to_tensor=True)
    embedding_similarity = util.pytorch_cos_sim(emb1, emb2).item()

    # --- EM and F1 ---
    em = compute_em(rag_response, ground_truth)
    f1 = compute_f1(rag_response, ground_truth)

    # Final result dictionary
    results = {
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": rag_response,
        "normalized_prediction": normalize_answer(rag_response),
        "em": em,
        "f1": f1,
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "embedding_similarity": embedding_similarity,
    }

    return results
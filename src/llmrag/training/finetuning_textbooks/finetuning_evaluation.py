from bert_score import BERTScorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import set_seed

from LLM_Model import load_local_llm
from data_processing import dataset_load, dataset_split
from helpers import get_max_len, preprocess_dataset, gen
from .data_loading import loading_dataset
from rag_eval_B1.eval_dataset import prepare_rag_evaluation_data

from tqdm import tqdm
import json

import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from statistics import mean
from nltk.tokenize import word_tokenize

model_name = "microsoft/phi-2"

llm_path = "./LLMs"
base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)

print("Base Model Loaded Successfully from Local Directory!")

dataset = prepare_rag_evaluation_data(file_path="synthetic_QAs.json")

# # dataset_path = "dataset.jsonl"
# data_output_path = "./AIbooks_dataset/text_finetuningData.jsonl"

# train_dataset, test_dataset = loading_dataset(file_path=data_output_path)

print("Dataset Created Successfully!")

def gen_batch_responses(model, prompts, tokenizer, maxlen=300, sample=True):
    toks = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True)
    toks = {k: v.to("cuda") for k, v in toks.items()}

    res = model.generate(
        **toks,
        max_new_tokens=maxlen,
        do_sample=sample,
        num_return_sequences=1,
        temperature=0.1,
        top_p=0.95,
        num_beams=1,
    )

    return tokenizer.batch_decode(res, skip_special_tokens=True)

ft_model = PeftModel.from_pretrained(base_model,
                                     "./finetuning_textbooks/finetuning_checkpoints/final_checkpoint/checkpoint-10000",
                                     torch_dtype=torch.float16,
                                     is_trainable=False)

# print("Finetuned Model Loaded Successfully!")

# --- Prepare prompts in batches ---
test_data = dataset
batch_size = 32
# predictions, references = [], []

predictions = []
references = []

with open("peftModel_inference_text.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        predictions.append(data["prediction"])
        references.append(data["reference"])

# for batch_start in tqdm(range(0, len(test_data), batch_size)):
#     batch_data = test_data[batch_start : batch_start + batch_size]   # slice list

#     # ─── Build prompt & ground-truth lists ──────────────────────────────
#     prompts = [
#         f"Instruct: Answer the following question accurately.\n{item['question']}\nOutput:\n"
#         for item in batch_data
#     ]
#     ground_truths = [item["answer"] for item in batch_data]

#     # ─── Generate model outputs in a batch ──────────────────────────────
#     batch_outputs = gen_batch_responses(base_model, prompts, tokenizer=eval_tokenizer)

#     # ─── Strip prompt prefix & collect refs/preds ───────────────────────
#     for out, gt in zip(batch_outputs, ground_truths):
#         try:
#             # keep only model text after "Output:\n" and before blank line
#             output_text = out.split("Output:\n", 1)[1].strip().split("\n\n", 1)[0].strip()
#         except IndexError:
#             output_text = out.strip()     # fallback if delimiter missing

#         predictions.append(output_text)
#         references.append(gt)

# # Save as JSONL file
# with open("BaseModel_inference_text.jsonl", "w", encoding="utf-8") as f:
#     for pred, ref in zip(predictions, references):
#         line = {"prediction": pred, "reference": ref}
#         f.write(json.dumps(line, ensure_ascii=False) + "\n")

# Tokenize predictions and references
# tokenized_preds = [pred.split() for pred in predictions]
# tokenized_refs = [[ref.split()] for ref in references]  # note the nested list for corpus_bleu

# # BLEU Score
# smooth_fn = SmoothingFunction().method4
# bleu = corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=smooth_fn)
# print(f"BLEU Score: {bleu:.4f}")

# # ROUGE Score (L, 1, 2)
# rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

# for ref, pred in zip(references, predictions):
#     scores = rouge.score(ref, pred)
#     rouge1_scores.append(scores['rouge1'].fmeasure)
#     rouge2_scores.append(scores['rouge2'].fmeasure)
#     rougeL_scores.append(scores['rougeL'].fmeasure)

# print(f"ROUGE-1 F1: {mean(rouge1_scores):.4f}")
# print(f"ROUGE-2 F1: {mean(rouge2_scores):.4f}")
# print(f"ROUGE-L F1: {mean(rougeL_scores):.4f}")

# # METEOR Score (average over all samples)
# meteor_scores = [
#     meteor_score([word_tokenize(ref)], word_tokenize(pred))
#     for ref, pred in zip(references, predictions)
# ]
# print(f"METEOR Score: {mean(meteor_scores):.4f}")

# eos_token_id = tokenizer.eos_token_id
# # base_model.eval()
# ft_model.eval()
# total_loss = 0
# total_tokens = 0

# for i in tqdm(range(0, len(test_data), batch_size)):
#     batch_data = test_data[i:i+batch_size]

#     # Construct prompts and ground truth answers
#     prompts = [
#         f"Instruct: Answer the following question accurately.\n{item['question']}\nOutput:\n"
#         for item in batch_data
#     ]
#     ground_truths = [item["answer"] for item in batch_data]
#     inputs = [prompt + answer for prompt, answer in zip(prompts, ground_truths)]

#     # Tokenize full input: prompt + answer
#     tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     input_ids = tokenized["input_ids"]
#     attention_mask = tokenized["attention_mask"]

#     # Compute prompt lengths for each input
#     prompt_lengths = [
#         len(tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0])
#         for prompt in prompts
#     ]

#     # Prepare labels: mask out the prompt part with -100 (ignored in loss)
#     labels = input_ids.clone()
#     for j, prompt_len in enumerate(prompt_lengths):
#         labels[j, :prompt_len] = -100
#         labels[labels == eos_token_id] = -100  # ignore <eos> tokens as well

#     # Move data to the appropriate device
#     input_ids = input_ids.to(ft_model.device)
#     attention_mask = attention_mask.to(ft_model.device)
#     labels = labels.to(ft_model.device)

#     # Optional: debug print
#     print("========== FULL DECODED INPUT ==========")
#     print(tokenizer.decode(input_ids[0]))
#     print("========== LABEL (ANSWER ONLY) =========")
#     print(tokenizer.decode(labels[0][labels[0] != -100]))

#     # Model forward pass
#     with torch.no_grad():
#         outputs = ft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         logits = outputs.logits

#     # Compute total token-level loss
#     token_count = (labels != -100).sum().item()
#     total_loss += loss.item() * token_count
#     total_tokens += token_count

# # Final validation loss
# final_val_loss = total_loss / total_tokens
# print(f"\nFinal Validation Loss: {final_val_loss:.4f}")

# import random
# i = random.randint(0, len(dataset))
# print(f"Sample Index: {i}")

# question = dataset[153]["question"]
# ground_truth_response = dataset[153]["answer"]

# print(question)
# print(ground_truth_response)

# prompt = f"Instruct: Answer the following question accurately.\n{question}\nOutput:\n"

# peft_model_res = gen(model=base_model,
#                      prompt=prompt,
#                      tokenizer=eval_tokenizer,
#                      )
# peft_model_output = peft_model_res[0].split('Output:\n')[1]
# #print(peft_model_output)
# # prefix, success, result = peft_model_output.partition('#End')

# dash_line = '-'.join('' for x in range(100))
# print(dash_line)
# print(f'INPUT PROMPT:\n{prompt}')
# print(dash_line)
# print(f'BASELINE Answer:\n{ground_truth_response}\n')
# print(dash_line)
# print(f'PEFT MODEL Response:\n{peft_model_output}')

scorer = BERTScorer(model_type='bert-base-uncased')
# P, R, F1 = scorer.score([ground_truth_response], [peft_model_output])
# print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
P, R, F1 = scorer.score(references, predictions)
print(f"\nBERTScore (All Samples) -> Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
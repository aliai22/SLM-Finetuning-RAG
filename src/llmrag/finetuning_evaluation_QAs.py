from bert_score import BERTScorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import set_seed

from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
import numpy as np

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

from LLM_Model import load_local_llm
from data_processing import dataset_load, dataset_split
from helpers import get_max_len, preprocess_dataset, gen

model_name = "microsoft/phi-2"

llm_path = "./LLMs"
base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)

# ✅ Force correct padding and token config
tokenizer.padding_side = "right"

print("Base Model Loaded Successfully from Local Directory!")

dataset_path = "uniqueQA_dataset0.7.jsonl"

dataset = dataset_load(file_path=dataset_path)
print("Data File Loaded Successfully!")

splitted_dataset = dataset_split(dataset=dataset)
print("Data Split Created Successfully!")

print(splitted_dataset['train'])
print(splitted_dataset['test'])

max_length = get_max_len(model=base_model)

seed = 42
set_seed(seed)

print("Dataset Preprocessing .....")

train_dataset = preprocess_dataset(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["train"])
eval_dataset = preprocess_dataset(tokenizer=tokenizer,
                                   max_length=max_length,
                                   seed=seed,
                                   dataset=splitted_dataset["test"])










print(f"Eval Dataaaaaaaaaaaaaaaaaaaaaaaaaaaaset: {eval_dataset}")





print("Dataset Preprocessed!")

# ft_model = PeftModel.from_pretrained(base_model,
#                                      "./Finetuning_Checkpoints_filtered/final-checkpoint/checkpoint-9750",
#                                      torch_dtype=torch.float16,
#                                      is_trainable=False)

ft_model = PeftModel.from_pretrained(base_model,
                                     "./finetuning_textbooks/finetuning_checkpoints/final_checkpoint/checkpoint-10000",
                                     torch_dtype=torch.float16,
                                     is_trainable=False)

# import random
# i = random.randint(0, len(splitted_dataset['test']))
# print(f"Sample Index: {i}")

# question = splitted_dataset['test'][15679]['question']
# ground_truth_response = splitted_dataset['test'][15679]['answer']

test_data = splitted_dataset['test']

# print(test_data[0])

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

# # --- Prepare prompts in batches ---
# batch_size = 64
# test_data = splitted_dataset["test"]
# predictions = []
# references = []

# for i in tqdm(range(0, len(test_data), batch_size)):
#     batch = {
#     "question": test_data["question"][i:i+batch_size],
#     "answer": test_data["answer"][i:i+batch_size]
#     }
#     batch = [{"question": q, "answer": a} for q, a in zip(batch["question"], batch["answer"])]

#     prompts = [f"Instruct: Answer the following question accurately.\n{sample['question']}\nOutput:\n"
#                for sample in batch]
#     ground_truths = [sample["answer"] for sample in batch]

#     batch_outputs = gen_batch_responses(ft_model, prompts, tokenizer=eval_tokenizer)

#     for out, gt in zip(batch_outputs, ground_truths):
#         try:
#             output_text = out.split('Output:\n')[1].strip().split("\n\n")[0].strip()
#         except IndexError:
#             output_text = out.strip()  # fallback

#         predictions.append(output_text)
#         references.append(gt)

# # Save as JSONL file
# with open("peftModel_inference_text.jsonl", "w", encoding="utf-8") as f:
#     for pred, ref in zip(predictions, references):
#         line = {"prediction": pred, "reference": ref}
#         f.write(json.dumps(line, ensure_ascii=False) + "\n")

# # Tokenize predictions and references
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

# # base_model.eval()
# ft_model.eval()
# total_loss = 0
# total_tokens = 0
# # correct_tokens = 0
# # em_correct = 0

predictions = []
references = []

with open("peftModel_inference_text.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        predictions.append(data["prediction"])
        references.append(data["reference"])

# # print("Inference Data Loaded Successfully!")

# # --- Prepare prompts in batches ---
# batch_size = 64
# test_data = splitted_dataset["test"]

# eos_token_id = tokenizer.eos_token_id

# for i in tqdm(range(0, len(test_data), batch_size)):
#     batch = {
#         "question": test_data["question"][i:i+batch_size],
#         "answer": test_data["answer"][i:i+batch_size]
#     }
#     batch = [{"question": q, "answer": a} for q, a in zip(batch["question"], batch["answer"])]

#     prompts = [f"Instruct: Answer the following question accurately.\n{sample['question']}\nOutput:\n"
#                for sample in batch]
#     ground_truths = [sample["answer"] for sample in batch]
#     inputs = [p + a for p, a in zip(prompts, ground_truths)]

#     # Tokenize full input: prompt + answer
#     tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     input_ids = tokenized["input_ids"]
#     attention_mask = tokenized["attention_mask"]

#     # Determine prompt lengths
#     prompt_lengths = []
#     for prompt in prompts:
#         prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"][0]
#         prompt_lengths.append(len(prompt_ids))

#     # Prepare labels: mask out prompt tokens
#     labels = input_ids.clone()
#     for j, prompt_len in enumerate(prompt_lengths):
#         labels[j, :prompt_len] = -100
#         labels[labels == eos_token_id] = -100

#     # Move to device
#     input_ids = input_ids.to(ft_model.device)
#     attention_mask = attention_mask.to(ft_model.device)
#     labels = labels.to(ft_model.device)

#     # Optional: debug print
#     print("========== FULL DECODED INPUT ==========")
#     print(tokenizer.decode(input_ids[0]))
#     print("========== LABEL (ANSWER ONLY) =========")
#     print(tokenizer.decode(labels[0][labels[0] != -100]))

#     # break

#     with torch.no_grad():
#         outputs = ft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         logits = outputs.logits

#     # Compute token-level loss
#     token_count = (labels != -100).sum().item()
#     total_loss += loss.item() * token_count
#     total_tokens += token_count

# # Final validation loss
# final_val_loss = total_loss / total_tokens
# print(f"\nFinal Validation Loss: {final_val_loss:.4f}")

# for sample in test_data:
#     question = sample['question']
#     ground_truth_response = sample['answer']
    
#     prompt = f"Instruct: Answer the following question accurately.\n{question}\nOutput:\n"
    
#     # Model generation
#     peft_model_res = gen(model=ft_model, prompt=prompt, tokenizer=eval_tokenizer)
#     peft_model_output = peft_model_res[0].split('Output:\n')[1].strip().split("\n\n")[0].strip()
    
#     # Collect for BERTScore and EM
#     predictions.append(peft_model_output)
#     references.append(ground_truth_response.strip())
    
#     # Exact Match
#     if peft_model_output.strip() == ground_truth_response.strip():
#         em_correct += 1
    
#     # 1. Combine prompt and expected output
#     full_input = prompt + ground_truth_response

#     # 2. Tokenize full sequence
#     tokenized = eval_tokenizer(full_input, return_tensors="pt", truncation=True, max_length=512)
#     input_ids = tokenized["input_ids"]

#     # 3. Find prompt token count
#     prompt_ids = eval_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
#     prompt_length = prompt_ids.shape[1]

#     # 4. Create labels: mask prompt tokens with -100, keep rest
#     labels = input_ids.clone()
#     labels[:, :prompt_length] = -100  # ignore prompt tokens for loss

#     # 5. Move to device
#     input_ids = input_ids.to(ft_model.device)
#     labels = labels.to(ft_model.device)
#     attention_mask = tokenized["attention_mask"].to(ft_model.device)
    
#     with torch.no_grad():
#         outputs = ft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         logits = outputs.logits


#     total_loss += loss.item()

#     preds = logits.argmax(dim=-1)
#     mask = labels != -100  # standard ignore index
#     correct = (preds == labels) & mask
#     correct_tokens += correct.sum().item()
#     total_tokens += mask.sum().item()

# # prompt = f"Instruct: Answer the following question accurately.\n{question}\nOutput:\n"

# # peft_model_res = gen(model=ft_model,
# #                      prompt=prompt,
# #                      tokenizer=eval_tokenizer,
# #                      )
# # peft_model_output = peft_model_res[0].split('Output:\n')[1]
# # print(peft_model_output)
# # prefix, success, result = peft_model_output.partition('#End')

# Final Loss
# avg_loss = total_loss / len(test_data)
# print(f"\nFinal Validation Loss: {avg_loss:.4f}")

# # Token-level accuracy
# token_accuracy = correct_tokens / total_tokens
# print(f"Token-Level Accuracy: {token_accuracy:.4f}")

# # Exact match
# em_score = em_correct / len(test_data)
# print(f"Exact Match Accuracy: {em_score:.4f}")

# # dash_line = '-'.join('' for x in range(100))
# # print(dash_line)
# # print(f'INPUT PROMPT:\n{prompt}')
# # print(dash_line)
# # print(f'BASELINE Answer:\n{ground_truth_response}\n')
# # print(dash_line)
# # print(f'PEFT MODEL Response:\n{peft_model_output}')

scorer = BERTScorer(model_type='bert-base-uncased')
# P, R, F1 = scorer.score([ground_truth_response], [peft_model_output])
# print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
P, R, F1 = scorer.score(references, predictions)
print(f"\nBERTScore (All Samples) -> Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
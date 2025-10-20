import os
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"

# from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
device_map = {"":0}

# model_name='microsoft/phi-2'

# class LLMService:

#     def __init__(self, base_model_id: str):
#         self.model_id = base_model_id
#         self.tokenizer = None
#         self.model = None

def load_llm(base_model_id):
    original_model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                                        device_map=device_map,
                                                        quantization_config=bnb_config,
                                                        trust_remote_code=True,
                                                        )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id,
                                            trust_remote_code=True,
                                            padding_side='left',
                                            add_eos_toke=True,
                                            add_bos_token=True,
                                            use_fast=False
                                            )
    tokenizer.pad_token = tokenizer.eos_token

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id,
                                                add_bos_token=True,
                                                trust_remote_code=True,
                                                use_fast=False
                                                )
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
    
    return original_model, tokenizer, eval_tokenizer



# saving model locally

def save_llm_locally(model, tokenizer, eval_tokenizer, local_dir):
    model.save_pretrained(f"{local_dir}/llm/")
    tokenizer.save_pretrained(f"{local_dir}/tokenizer/")
    eval_tokenizer.save_pretrained(f"{local_dir}/evaltokenizer/")



# loading locally saved model

def load_local_llm(local_dir, base_model_id=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(f"{local_dir}/llm/")
    model.to(device)

    # Load the tokenizer
    if base_model_id:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"{local_dir}/tokenizer/", trust_remote_code=True, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        eval_tokenizer = AutoTokenizer.from_pretrained(f"{local_dir}/evaltokenizer/", trust_remote_code=True, use_fast=False)
        eval_tokenizer.pad_token = eval_tokenizer.eos_token

    return model, tokenizer, eval_tokenizer
# from LLM_Model import load_local_llm
from peft import PeftModel
import torch
# import os

# llm_path = "./LLMs"
# # print(os.listdir(llm_path))
# base_model, tokenizer, eval_tokenizer = load_local_llm(llm_path)
# print("Local Base Model Accessed!")

def load_ft_model(base_model, ft_ckpt:str):
    ft_model = PeftModel.from_pretrained(base_model,
                                         ft_ckpt,
                                         torch_dtype=torch.float16,
                                         is_trainable=False)
    print("Finetuned Model Loaded Successfully!")
    return ft_model
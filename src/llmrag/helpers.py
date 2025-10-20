from functools import partial
from transformers import AutoTokenizer

def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction','output')
    Then concatenate them using two newline characters 
    :param sample: Sample dictionnary
    """
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruct: You are an AI assistant that answers a user's query accurately."
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"
    
    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"{sample['question']}" if sample["question"] else None
    response = f"{RESPONSE_KEY}\n{sample['answer']}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt

    return sample

def get_max_len(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max Length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using Default Max Length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    # print(batch)
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int,seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats,)#, batched=True)
    
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        batch_size=32,
        remove_columns=['question', 'answer'],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def gen(model,prompt, tokenizer, maxlen=300, sample=True):
    toks = tokenizer(prompt, return_tensors="pt")
    res = model.generate(**toks.to("cuda"),
                         max_new_tokens=maxlen,
                         do_sample=sample,num_return_sequences=1,
                         temperature=0.1,
                         num_beams=1,
                         top_p=0.95,
                        ).to('cpu')
    return tokenizer.batch_decode(res,skip_special_tokens=True)
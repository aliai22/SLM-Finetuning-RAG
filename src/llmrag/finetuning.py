from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
import torch

config = LoraConfig(
    r=32, #Rank
    lora_alpha=32,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense'
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

def prepare_peft_model(model, configuration=config):

    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    peft_model = get_peft_model(model, configuration)

    return peft_model

def prepare_peft_trainer(output_dir, peft_model, training_dataset, evaluation_dataset, tokenizer):
    peft_training_args = transformers.TrainingArguments(
        output_dir = output_dir,
        warmup_steps=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        max_steps=10000,
        # num_train_epochs=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir = 'True',
        group_by_length=True,
        save_total_limit=True,
        load_best_model_at_end=True
        # remove_unused_columns=False
    )

    device="cuda" if torch.cuda.is_available() else "cpu"
    peft_training_args.device

    peft_model.config.use_cache = False

    peft_trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=training_dataset,
        eval_dataset=evaluation_dataset,
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return peft_trainer
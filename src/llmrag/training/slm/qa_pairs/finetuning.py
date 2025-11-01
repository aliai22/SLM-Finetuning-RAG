from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
import torch
import os
from packaging.version import parse as _parse_version

# ---- Transformers v4/v5 TrainingArguments compatibility shim ----
def _ta_kwargs(**kwargs):
    """
    Accepts either evaluation_strategy (v4) or eval_strategy (v5) and maps to the
    right name for the installed transformers version. All other kwargs are
    passed through unchanged.
    """
    v = _parse_version(transformers.__version__)
    if v.major >= 5:
        # v5 expects eval_strategy
        if "evaluation_strategy" in kwargs and "eval_strategy" not in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    else:
        # v4 expects evaluation_strategy
        if "eval_strategy" in kwargs and "evaluation_strategy" not in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return kwargs
# -----------------------------------------------------------------

config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

def prepare_peft_model(model, configuration=config):
    """
    Your original behavior:
      1) enable gradient checkpointing
      2) prepare model for k-bit training
      3) wrap with LoRA (get_peft_model)

    Added (non-breaking):
      - If env LLMRAG_DISABLE_PEFT=1 is set, return the base model unchanged.
        This is only to simplify smoke tests on Windows/CPU and does not alter
        defaults unless the env var is set.
    """
    if os.getenv("LLMRAG_DISABLE_PEFT", "0") == "1":
        print("[LLMRAG] PEFT disabled via LLMRAG_DISABLE_PEFT=1")
        return model

    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # 3 - Apply LoRA with your provided configuration
    peft_model = get_peft_model(model, configuration)
    return peft_model

def prepare_peft_trainer(
    output_dir,
    peft_model,
    training_dataset,
    evaluation_dataset,
    tokenizer,
    training_args=None,   # NEW: allow caller to provide external TrainingArguments
):
    """
    Your original behavior and defaults are preserved. If 'training_args' is not
    provided, we build the same TrainingArguments you had before. The only change
    is we pass them through the _ta_kwargs() shim so it works on both transformers v4 and v5.
    """
    if training_args is None:
        peft_training_args = transformers.TrainingArguments(
            **_ta_kwargs(
                output_dir=output_dir,
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
                evaluation_strategy="steps",  # will map to eval_strategy on v5
                eval_steps=50,
                do_eval=True,
                gradient_checkpointing=True,
                report_to="none",
                overwrite_output_dir="True",
                group_by_length=True,
                save_total_limit=True,
                load_best_model_at_end=True,
                # remove_unused_columns=False
            )
        )
    else:
        # Use the caller-provided TrainingArguments as-is
        peft_training_args = training_args

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # preserve your original line (no-op) to avoid changing behavior
    peft_training_args.device  # noqa: F841

    peft_model.config.use_cache = False

    peft_trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=training_dataset,
        eval_dataset=evaluation_dataset,
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    return peft_trainer

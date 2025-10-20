from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData

def load_model(model_id, eval=False):
    if not eval:
        model = SentenceTransformer(
            model_id,
            # device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"attn_implementation": "sdpa"},
            model_card_data=SentenceTransformerModelCardData(
                language="en",
                license="apache-2.0",
                model_name="BGE base Financial Matryoshka",
            ),
        )
    else:
        model = SentenceTransformer(
            model_id,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )
    # model.tokenizer.pad_token = model.tokenizer.eos_token  # Assign EOS token as PAD token
    print("Model Loaded Successfully!")
    return model

def define_loss(model, dim=512):
    inner_train_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model,
        inner_train_loss,
        matryoshka_dims=[dim]
    )
    return train_loss

def prepare_trainer(model, train_dataset, train_loss, evaluator, output_dir):
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir, # output directory
        # num_train_epochs=4,                         # number of epochs
        max_steps = 5000,
        per_device_train_batch_size=16,             # train batch size
        gradient_accumulation_steps=8,             # for a global batch size of 512
        per_device_eval_batch_size=8,              # evaluation batch size
        # warmup_ratio=0.1,                           # warmup ratio
        warmup_steps=1,
        learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
        lr_scheduler_type="cosine",                 # use constant learning rate scheduler
        optim="adamw_torch_fused",                  # use fused adamw optimizer
        # tf32=True,                                  # use tf32 precision
        # bf16=True,                                  # use bf16 precision
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        eval_strategy="steps",                      # evaluate after each step
        save_strategy="steps",                      # save after each step
        logging_steps=50,                           # log every 50 steps
        eval_steps = 50,
        save_total_limit=3,                         # save only the last 3 models
        gradient_checkpointing=True,
        load_best_model_at_end=True,                # load the best model when training ends
        metric_for_best_model="eval_dim_512_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
    )
    device="cuda" if torch.cuda.is_available() else "cpu"
    training_args.device
    trainer = SentenceTransformerTrainer(
        model=model, # bg-base-en-v1
        args=training_args,  # training arguments
        train_dataset=train_dataset.select_columns(
            ["positive", "anchor"]
        ),  # training dataset
        loss=train_loss,
        evaluator=evaluator,
    )
    return trainer
---
language:
- en
license: apache-2.0
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:324171
- loss:MatryoshkaLoss
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-base-en-v1.5
widget:
- source_sentence: In the presence of extreme values, researchers should interpret
    Pearson correlation coefficient cautiously, recognizing potential influences on
    the results. Sensitivity analyses, robust correlation measures, or data transformations
    can help address concerns related to extreme values.
  sentences:
  - Can you describe the architecture of Temporal Convolutional Networks (TCNs) used
    for action segmentation?
  - What role does 3D computer vision play in the development of autonomous drones
    for monitoring and preserving historical gardens and botanical collections?
  - How does the interpretation of Pearson correlation coefficient change when analyzing
    data with extreme values?
- source_sentence: The integration of multi-omics data is essential in modern biology
    as it enables researchers to unravel complex biological processes and understand
    the interconnectedness of molecular data.
  sentences:
  - Can spaCy's word vectors handle polysemy effectively?
  - Why is the integration of multi-omics data essential in modern biology?
  - In what ways do symbols facilitate the transfer of knowledge between different
    domains in Neuro Symbolic AI, promoting adaptability and generalization?
- source_sentence: Yes, sequence-to-sequence models can be applied to collaborative
    filtering by representing user-item interactions as sequential data and predicting
    future interactions.
  sentences:
  - How do MAS enable autonomous fault detection and localization in self-healing
    networks?
  - How do variational autoencoders facilitate a better grasp of uncertainty and variability
    within the latent space compared to standard autoencoders?
  - Can sequence-to-sequence models be adapted for tasks involving collaborative filtering
    in recommendation systems?
- source_sentence: Reinforcement learning integrates data collection during agent
    operation, eliminating separate data collection processes and improving efficiency.
  sentences:
  - How does Visual SLAM handle situations with changes in scene appearance caused
    by reflections on wet surfaces in outdoor environments?
  - Can you explain how reinforcement learning eliminates the need for independent
    data collection processes, making it more efficient compared to traditional methods?
  - Can robotic vision contribute to the development of assistive technologies for
    individuals with neurodegenerative disorders by interpreting facial expressions
    and emotions?
- source_sentence: Challenges include domain-specific nuances that might not be adequately
    captured during pre-training, and biases present in the pre-training data may
    impact the model's performance on specific tasks.
  sentences:
  - What are potential drawbacks or challenges associated with relying heavily on
    pre-trained models for NER?
  - How does stemming impact the recognition of sentiment in text data with varying
    degrees of subjectivity?
  - How can regularization techniques be adapted for machine learning models dealing
    with imbalanced classes in predicting equipment failures in the automotive industry?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: BGE base Financial Matryoshka
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: dim 512
      type: dim_512
    metrics:
    - type: cosine_accuracy@1
      value: 0.25750298453593934
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.45345512090840945
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.5580665759737916
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.6801410366750882
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.25750298453593934
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.1511517069694698
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.1116133151947583
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.0680141036675088
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.25750298453593934
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.45345512090840945
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.5580665759737916
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.6801410366750882
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.45387034658339637
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.3831194691338142
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.39462255780930094
      name: Cosine Map@100
---

# BGE base Financial Matryoshka

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on the json dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) <!-- at revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - json
- **Language:** en
- **License:** apache-2.0

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': True}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "Challenges include domain-specific nuances that might not be adequately captured during pre-training, and biases present in the pre-training data may impact the model's performance on specific tasks.",
    'What are potential drawbacks or challenges associated with relying heavily on pre-trained models for NER?',
    'How can regularization techniques be adapted for machine learning models dealing with imbalanced classes in predicting equipment failures in the automotive industry?',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `dim_512`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.2575     |
| cosine_accuracy@3   | 0.4535     |
| cosine_accuracy@5   | 0.5581     |
| cosine_accuracy@10  | 0.6801     |
| cosine_precision@1  | 0.2575     |
| cosine_precision@3  | 0.1512     |
| cosine_precision@5  | 0.1116     |
| cosine_precision@10 | 0.068      |
| cosine_recall@1     | 0.2575     |
| cosine_recall@3     | 0.4535     |
| cosine_recall@5     | 0.5581     |
| cosine_recall@10    | 0.6801     |
| **cosine_ndcg@10**  | **0.4539** |
| cosine_mrr@10       | 0.3831     |
| cosine_map@100      | 0.3946     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### json

* Dataset: json
* Size: 324,171 training samples
* Columns: <code>positive</code> and <code>anchor</code>
* Approximate statistics based on the first 1000 samples:
  |         | positive                                                                          | anchor                                                                             |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             |
  | details | <ul><li>min: 8 tokens</li><li>mean: 34.04 tokens</li><li>max: 89 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 23.59 tokens</li><li>max: 53 tokens</li></ul> |
* Samples:
  | positive                                                                                                                                                                                                  | anchor                                                                                                                                                            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Techniques include using character-level embeddings, incorporating external knowledge bases, and leveraging subword tokenization to handle out-of-vocabulary entities in NER.</code>                | <code>What are some techniques for handling out-of-vocabulary entities in NER?</code>                                                                             |
  | <code>Tagging involves labeling or annotating data to create a reference for training models.</code>                                                                                                      | <code>What is tagging in machine learning?</code>                                                                                                                 |
  | <code>Models incorporating tonal awareness, using tonal-specific embeddings, or training on tonal language-specific data can assist in text classification tasks in languages with tonal features.</code> | <code>What considerations should be taken into account for text classification tasks in languages with tonal features, such as Mandarin Chinese or Yoruba?</code> |
* Loss: [<code>MatryoshkaLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#matryoshkaloss) with these parameters:
  ```json
  {
      "loss": "MultipleNegativesRankingLoss",
      "matryoshka_dims": [
          512
      ],
      "matryoshka_weights": [
          1
      ],
      "n_dims_per_step": -1
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `gradient_accumulation_steps`: 8
- `learning_rate`: 2e-05
- `max_steps`: 5000
- `lr_scheduler_type`: cosine
- `warmup_ratio`: 0.1
- `load_best_model_at_end`: True
- `optim`: adamw_torch_fused
- `gradient_checkpointing`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 8
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 3.0
- `max_steps`: 5000
- `lr_scheduler_type`: cosine
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: True
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | dim_512_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:----------------------:|
| 0.0197 | 50   | 0.1291        | 0.3830                 |
| 0.0395 | 100  | 0.0523        | 0.4055                 |
| 0.0592 | 150  | 0.0263        | 0.4141                 |
| 0.0790 | 200  | 0.0214        | 0.4197                 |
| 0.0987 | 250  | 0.0183        | 0.4226                 |
| 0.1185 | 300  | 0.0156        | 0.4273                 |
| 0.1382 | 350  | 0.0131        | 0.4251                 |
| 0.1579 | 400  | 0.0143        | 0.4306                 |
| 0.1777 | 450  | 0.0114        | 0.4294                 |
| 0.1974 | 500  | 0.0128        | 0.4343                 |
| 0.2172 | 550  | 0.0085        | 0.4325                 |
| 0.2369 | 600  | 0.0082        | 0.4368                 |
| 0.2567 | 650  | 0.0094        | 0.4367                 |
| 0.2764 | 700  | 0.0118        | 0.4372                 |
| 0.2961 | 750  | 0.008         | 0.4361                 |
| 0.3159 | 800  | 0.0077        | 0.4398                 |
| 0.3356 | 850  | 0.0106        | 0.4350                 |
| 0.3554 | 900  | 0.0085        | 0.4397                 |
| 0.3751 | 950  | 0.0079        | 0.4434                 |
| 0.3948 | 1000 | 0.0054        | 0.4391                 |
| 0.4146 | 1050 | 0.007         | 0.4435                 |
| 0.4343 | 1100 | 0.0065        | 0.4442                 |
| 0.4541 | 1150 | 0.0079        | 0.4450                 |
| 0.4738 | 1200 | 0.0068        | 0.4419                 |
| 0.4936 | 1250 | 0.0066        | 0.4419                 |
| 0.5133 | 1300 | 0.0061        | 0.4419                 |
| 0.5330 | 1350 | 0.0064        | 0.4428                 |
| 0.5528 | 1400 | 0.0067        | 0.4450                 |
| 0.5725 | 1450 | 0.0047        | 0.4418                 |
| 0.5923 | 1500 | 0.0046        | 0.4442                 |
| 0.6120 | 1550 | 0.0043        | 0.4467                 |
| 0.6318 | 1600 | 0.0061        | 0.4471                 |
| 0.6515 | 1650 | 0.0066        | 0.4477                 |
| 0.6712 | 1700 | 0.0063        | 0.4485                 |
| 0.6910 | 1750 | 0.0046        | 0.4464                 |
| 0.7107 | 1800 | 0.0068        | 0.4481                 |
| 0.7305 | 1850 | 0.0049        | 0.4474                 |
| 0.7502 | 1900 | 0.0055        | 0.4469                 |
| 0.7700 | 1950 | 0.0064        | 0.4491                 |
| 0.7897 | 2000 | 0.005         | 0.4479                 |
| 0.8094 | 2050 | 0.0072        | 0.4505                 |
| 0.8292 | 2100 | 0.0045        | 0.4459                 |
| 0.8489 | 2150 | 0.0034        | 0.4477                 |
| 0.8687 | 2200 | 0.0045        | 0.4505                 |
| 0.8884 | 2250 | 0.0058        | 0.4525                 |
| 0.9081 | 2300 | 0.005         | 0.4535                 |
| 0.9279 | 2350 | 0.0061        | 0.4485                 |
| 0.9476 | 2400 | 0.0049        | 0.4500                 |
| 0.9674 | 2450 | 0.0046        | 0.4504                 |
| 0.9871 | 2500 | 0.0066        | 0.4502                 |
| 1.0070 | 2550 | 0.0055        | 0.4519                 |
| 1.0267 | 2600 | 0.0042        | 0.4537                 |
| 1.0464 | 2650 | 0.0051        | 0.4524                 |
| 1.0662 | 2700 | 0.0026        | 0.4511                 |
| 1.0859 | 2750 | 0.0043        | 0.4523                 |
| 1.1057 | 2800 | 0.0031        | 0.4517                 |
| 1.1254 | 2850 | 0.0029        | 0.4519                 |
| 1.1452 | 2900 | 0.0027        | 0.4511                 |
| 1.1649 | 2950 | 0.0025        | 0.4513                 |
| 1.1846 | 3000 | 0.0016        | 0.4521                 |
| 1.2044 | 3050 | 0.0026        | 0.4514                 |
| 1.2241 | 3100 | 0.0013        | 0.4518                 |
| 1.2439 | 3150 | 0.0019        | 0.4533                 |
| 1.2636 | 3200 | 0.0024        | 0.4513                 |
| 1.2834 | 3250 | 0.0023        | 0.4523                 |
| 1.3031 | 3300 | 0.0014        | 0.4523                 |
| 1.3228 | 3350 | 0.0025        | 0.4541                 |
| 1.3426 | 3400 | 0.002         | 0.4511                 |
| 1.3623 | 3450 | 0.002         | 0.4524                 |
| 1.3821 | 3500 | 0.0012        | 0.4524                 |
| 1.4018 | 3550 | 0.0011        | 0.4527                 |
| 1.4215 | 3600 | 0.0021        | 0.4512                 |
| 1.4413 | 3650 | 0.0013        | 0.4522                 |
| 1.4610 | 3700 | 0.0017        | 0.4532                 |
| 1.4808 | 3750 | 0.0016        | 0.4514                 |
| 1.5005 | 3800 | 0.0009        | 0.4526                 |
| 1.5203 | 3850 | 0.0014        | 0.4512                 |
| 1.5400 | 3900 | 0.0022        | 0.4509                 |
| 1.5597 | 3950 | 0.0018        | 0.4513                 |
| 1.5795 | 4000 | 0.001         | 0.4527                 |
| 1.5992 | 4050 | 0.0009        | 0.4528                 |
| 1.6190 | 4100 | 0.001         | 0.4525                 |
| 1.6387 | 4150 | 0.0012        | 0.4537                 |
| 1.6585 | 4200 | 0.0015        | 0.4530                 |
| 1.6782 | 4250 | 0.0011        | 0.4535                 |
| 1.6979 | 4300 | 0.0016        | 0.4532                 |
| 1.7177 | 4350 | 0.0016        | 0.4533                 |
| 1.7374 | 4400 | 0.0012        | 0.4556                 |
| 1.7572 | 4450 | 0.0022        | 0.4535                 |
| 1.7769 | 4500 | 0.0014        | 0.4542                 |
| 1.7967 | 4550 | 0.0015        | 0.4538                 |
| 1.8164 | 4600 | 0.0018        | 0.4539                 |
| 1.8361 | 4650 | 0.0013        | 0.4528                 |
| 1.8559 | 4700 | 0.0012        | 0.4540                 |
| 1.8756 | 4750 | 0.0015        | 0.4527                 |
| 1.8954 | 4800 | 0.0015        | 0.4532                 |
| 1.9151 | 4850 | 0.0018        | 0.4542                 |
| 1.9349 | 4900 | 0.0015        | 0.4525                 |
| 1.9546 | 4950 | 0.001         | 0.4525                 |
| 1.9743 | 5000 | 0.0012        | 0.4539                 |


### Framework Versions
- Python: 3.10.0
- Sentence Transformers: 3.3.1
- Transformers: 4.46.3
- PyTorch: 2.5.1+cu124
- Accelerate: 1.2.0
- Datasets: 3.2.0
- Tokenizers: 0.20.3

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MatryoshkaLoss
```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
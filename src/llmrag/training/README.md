# Model Fine-Tuning Workflows

This module contains all scripts related to the **fine-tuning phase** of the study:

> **Domain Adaptation of Small Language Models via Fine-Tuning and RAG:  
> A Case Study in Scientific Question Answering**

The fine-tuning process consists of two complementary parts:

1. **Small Language Model (SLM) fine-tuning** — adapting the base Phi-2 model to domain-specific content.  
2. **Embedding model fine-tuning** — optimizing dense vector representations for retrieval in RAG pipelines.

Both workflows are designed for reproducibility and can be executed independently.

---

## Folder Structure
```bash
training/
├── slm/
│ ├── qa_pairs/         # Fine-tuning Phi-2 on QA pairs
│ ├── text_chunks/      # Fine-tuning Phi-2 on textbook-derived text
│ └── ...
└── embedding/          # Embedding model fine-tuning (BGE-based)
└── ... 
```
---

## 1. SLM Fine-Tuning

The **SLM fine-tuning** adapts the Phi-2 model for domain question answering using two content types:
- **QA pairs** (structured knowledge)
- **Textbook chunks** (unstructured explanatory text)

Training leverages **parameter-efficient fine-tuning (PEFT)** through LoRA adapters.

**Base model:** `microsoft/phi-2`  
**Fine-tuned outputs:** stored under `models/gen_model/qa_pairs_finetuning/` & `models/gen_model/text_chunks_finetuning/`


> **Figure 1. Fine-tuning workflow for the Small Language Model (Phi-2)**  
> ![SLM Fine-Tuning Workflow](https://github.com/aliai22/SLM-Finetuning-RAG/blob/refactor/docs/figures/slm_finetuning-removebg-preview.jpg)

### Key scripts
| File | Purpose |
|------|----------|
| `text_chunks/textFT_main.py` | Fine-tuning Phi-2 on textbook content |
| `qa_pairs/main.py` | Fine-tuning Phi-2 on QA pair data |

### Training Environment
- **Framework:** Hugging Face Transformers + PEFT  
- **Training device:** CUDA-enabled GPU  
- **Batch size:** 16–32  
- **Precision:** Mixed FP16  
- **Loss objective:** Next-token prediction (causal LM loss)

### Output
Each training run produces:
- Model checkpoints under `/models/gen_model/...`
- GPU usage logs (CSV)
- Validation loss and generated inference logs

---

## 2. Embedding Model Fine-Tuning

The **embedding fine-tuning** stage adapts sentence embeddings to better capture semantic similarity in scientific QA contexts.

**Base embedder:** `bge-base-en-v1.5`  
**Fine-tuned checkpoint:**  `models/emb_model/`


**Goal:** Improve retrieval alignment between question and context in RAG pipelines.

> **Figure 2. Embedding Fine-Tuning Workflow**  
> ![Embedding Fine-Tuning Workflow](https://github.com/aliai22/SLM-Finetuning-RAG/blob/refactor/docs/figures/embedding_finetuning-removebg-preview.jpg)

### Key script
| File | Purpose |
|------|----------|
| `embedding/embedd_finetuning.py` | Fine-tuning and evaluation of the embedding model |

### Training Details
- **Dataset:** scientific QA pairs aligned with domain texts  
- **Loss:** Contrastive cosine similarity loss  
- **Optimizer:** AdamW  
- **Evaluation metric:** cosine similarity on held-out pairs  

---

## Notes
- Fine-tuned weights are versioned under `/models/` for reproducibility.  
- The codebase is modular: embedding and language fine-tuning are independent.  
- For exact hyperparameters and metrics, refer to the corresponding section of the paper.

---

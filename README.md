# Domain Adaptation of Small Language Models via Fine-Tuning and RAG
*A Case Study in Scientific Question Answering*

This repository accompanies the research article **“Domain Adaptation of Small Language Models via Fine-Tuning and RAG: A Case Study in Scientific Question Answering.”**  
It provides:

- Training code to **fine-tune** an SLM (Phi-2) on two datasets
- **Embedding model fine-tuning**
- Four **RAG variants** (B1, B2, FT-1, FT-2)
- A modular **evaluation & analysis suite** (sensitivity, contamination audit, keyword ablation)
- Curated **fine-tuned models** and **datasets** needed to reproduce the study

> All datasets needed for reproduction are included in the repo under `data/`. The base model is **microsoft/phi-2** (Hugging Face).

---

## Table of Contents
- [Highlights](#highlights)
- [Environment](#environment)
- [Models & Data](#models--data)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Pipelines](#pipelines)
- [Evaluation Suite](#evaluation-suite)
- [Hardware Notes](#hardware-notes)
- [Citation](#citation)
- [Contact](#contact)

---

## Highlights
- **Focus**: Domain adaptation of a small LLM (Phi-2) via supervised fine-tuning and RAG for scientific Q&A.
- **Four RAG variants**:
  - **B1 / B2**: baseline pipelines
  - **FT-1**: finetuned LLM + textbook chunk DB
  - **FT-2**: finetuned LLM + QA-pairs DB
- **Evaluation**:
  - **Sensitivity** (prompt & query)
  - **Contamination audit** (n-gram overlap + MinHash/LSH)
  - **Keyword ablation** (FT-2 reranking robustness)
- **Reproducibility**:
  - All data required is shipped in `data/`
  - Clear entry points for training, RAG, and evaluation

---

## Environment
- **Python**: 3.10+ recommended  
- **CUDA**: All training and main analyses were run on **GPU**.  
  - CPU runs are possible but slow.  
  - For full runs, use a CUDA-enabled PyTorch build.

Install:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

python -m pip install -U pip
pip install -e .
```
> If using GPU, install the correct CUDA build of PyTorch first (see [pytorch.org](pytorch.org)).

---

## Models & Data
### Base model
- Default: `microsoft/phi-2` via Hugging Face Hub.
### Local model cache (optional)
If running fully offline, you can place a local copy under:
```bash
models/
└── base/                              # optional local cache for base model (Phi-2)
    ├── llm/                    
    ├── tokenizer/                     
    └── evaltokenizer/                 
└── emb_model/                         # finetuned embedding model (bge-base-en-v1.5)
└── gen_model/                         # finetuned generative model (phi-2)
    ├── qa_pairs/
    ├── text_chunks/

```
Otherwise, the code will fetch from HF Hub and cache in local directory on first run.
### Datasets
All datasets used for experiments are included under:
```bash
data/
├── finetuning/
│   ├── text/                         # text-based fine-tuning dataset (used by SLM/text_chunks)
│   │   └── text_finetuningData.jsonl
│   └── qa/                           # question-answer fine-tuning dataset (used by SLM/qa_pairs)
│       └── synthetic_QAs.json        # evaluation or synthetic datasets
│       └── uniqueQA_dataset0.7.jsonl
└── resources/                        # any auxiliary configs or metadata
```
No external downloads are required.

---
## Repository Layout
```bash
src/
└── llmrag/
    ├── pipelines/                     # end-to-end RAG pipelines
    │   ├── b1rag/                     # Baseline RAG 1: standard retrieval-generation
    │   ├── b2rag/                     # Baseline RAG 2: enhanced retrieval strategy
    │   ├── ft_1rag/                   # Finetuned variant using textbook chunks
    │   └── ft_2rag/                   # Finetuned variant using QA pairs
    │
    ├── eval/                          # evaluation and analysis suite
    │   └── common/
    │       ├── datasets.py            # dataset loading & preparation utilities
    │       ├── runners.py             # unified metric computations (EM, F1, BLEU, ROUGE, EmbSim)
    │       │
    │       ├── sensitivity/           # query & prompt sensitivity analysis
    │       │   ├── prompts.py         # pre-defined prompt templates
    │       │   ├── perturbations.py   # query mutation functions
    │       │   └── run.py             # main sensitivity experiment runner
    │       │
    │       ├── contamination.py       # contamination audit (n-gram overlap + MinHash/LSH)
    │       └── kw_ablation.py         # keyword ablation (FT-2 re-ranking analysis)
    │
    ├── training/                      # fine-tuning utilities
    │   ├── slm/                       # SLM (Phi-2) fine-tuning workflows
    │   │   ├── text_chunks/           # fine-tuning on text chunk dataset
    │   │   └── qa_pairs/              # fine-tuning on QA pairs dataset
    │   │
    │   └── embedding/                 # embedding model fine-tuning scripts
    │       └── embedd_finetuning.py
    │
    ├── __init__.py
    └── (other shared utilities)

```

---

## Quickstart
> Below commands use defaults. See each module for extra kwargs (e.g., `top_k`, `temperature`, `prompt_id`, `ctx_order`).
### Run a RAG pipeline
```bash
# FT-1 (finetuned on QA pairs + textbook chunks)
python -m llmrag.pipelines.ft_1rag.rag_main

# FT-2 (finetuned on text chunks + QA pairs)
python -m llmrag.pipelines.ft_2rag.rag_main

# Baselines
python -m llmrag.pipelines.b1rag.rag_main
python -m llmrag.pipelines.b2rag.rag_main

```
### Evaluation
```bash
# Sensitivity (prompt + query)
python -m llmrag.eval.common.sensitivity.run

# Contamination audit
python -m llmrag.eval.common.contamination

# Keyword ablation (FT-2)   
python -m llmrag.eval.common.kw_ablation

```
---

## Pipelines
### RAG Variant Summary

The repository implements four Retrieval-Augmented Generation (RAG) variants designed to analyze how domain-specific fine-tuning and retrieval source affect performance.  
Each variant combines a *generator model* (Phi-2 or fine-tuned Phi-2) with a specific *vector database source*.

| Variant | Generator Fine-tuning Dataset | Vector DB Source | Generator Type | Notes |
|----------|------------------------------|------------------|----------------|-------|
| **B1-RAG** | — *(no fine-tuning)* | **Textbook data (chunks)** | Base Phi-2 | Baseline using textbook-based retrieval |
| **B2-RAG** | — *(no fine-tuning)* | **QA pairs** | Base Phi-2 | Baseline using QA-based retrieval |
| **FT-1-RAG** | **QA pairs** | **Textbook data (chunks)** | Fine-tuned Phi-2 | Domain-adapted generator tested on textbook retrieval |
| **FT-2-RAG** | **Textbook data (chunks)** | **QA pairs** | Fine-tuned Phi-2 | Cross-domain fine-tuning with keyword-based reranking |

**Key Insight:**  
- FT-1 and FT-2 explore *cross-domain generalization* by swapping the generator’s fine-tuning domain and the retrieval database source.  
- B1 and B2 serve as baselines with the non-finetuned Phi-2, isolating the contribution of retrieval and domain adaptation.


Each pipeline exposes a function:
```bash
answer, meta_or_docs = run_rag_pipeline(query="...", **optional_kwargs)
```
#### Common kwargs (when supported):
- `prompt_id`: select a prompt template (see `eval/common/sensitivity/prompts.py`)
- `top_k`: retrieval depth
- `temperature`: generation temperature
- `ctx_order`: `"as_is"` or `"shuffled"`
#### Notes
- **FT-1**: Phi-2 fine-tuned on QA pairs, retrieving from textbook-based vector DB  
- **FT-2**: Phi-2 fine-tuned on textbook data, retrieving from QA-pair vector DB  
- **B1/B2**: baselines using non-finetuned Phi-2, with the same retrieval sources as FT-1 and FT-2 respectively

---

## Evluation Suite
- Sensitivity (`eval/common/sensitivity/run.py`): Measures robustness to prompt variants and query perturbations.
- Contamination audit (`eval/common/contamination.py`): Checks for leakage between evaluation and training/KB corpora using 5-gram overlap and MinHash/LSH.
- Keyword ablation (`eval/common/kw_ablation.py`): Evaluates keyword-aware reranking by comparing retrieval metrics (P@K, nDCG, MRR, MAP) and end-to-end answer quality.

---

## Hardware Notes
- Intended: NVIDIA GPU (CUDA).

    This is how we trained and evaluated in the study (VRAM/time details are in the paper).
- CPU: Feasible for tiny tests only. Expect slow generation and embedding.

---

## Citation

```bash
@article{YourLastName2025SLM-RAG,
  title   = {Domain Adaptation of Small Language Models via Fine-Tuning and RAG: A Case Study in Scientific Question Answering},
  author  = {M. Ali},
  year    = {2025},
  note    = {Code and datasets: https://github.com/aliai22/SLM-Finetuning-RAG}
}
```
---

## Contact

Muhammad Ali

Email: mali.msai22seecs@seecs.edu.pk
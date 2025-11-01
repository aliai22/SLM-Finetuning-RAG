# RAG Pipelines

This module implements **four distinct Retrieval-Augmented Generation (RAG) variants** developed as part of the research study:

> **Domain Adaptation of Small Language Models via Fine-Tuning and RAG:  
> A Case Study in Scientific Question Answering**

Each pipeline combines a **small language model (Phi-2)** with a **retrieval database** (built using Chroma) to analyze the effects of fine-tuning and retrieval source on domain adaptation.

---

## Overview of Variants

| Variant | Generator Fine-tuning Dataset | Vector DB Source | Generator Type | Notes |
|----------|------------------------------|------------------|----------------|-------|
| **B1-RAG** | — *(no fine-tuning)* | **Textbook data (chunks)** | Base Phi-2 | Baseline using textbook-based retrieval |
| **B2-RAG** | — *(no fine-tuning)* | **QA pairs** | Base Phi-2 | Baseline using QA-based retrieval |
| **FT-1-RAG** | **QA pairs** | **Textbook data (chunks)** | Fine-tuned Phi-2 | Cross-domain fine-tuning setup |
| **FT-2-RAG** | **Textbook data (chunks)** | **QA pairs** | Fine-tuned Phi-2 | Includes keyword-based reranking |

Each variant has its own directory:

- **`base_rag1/`** → baseline RAG using textbook retrieval
- **`base_rag2/`** → baseline RAG using QA-based retrieval
- **`ft_1rag/`** → fine-tuned generator (QA pairs) + textbook retrieval
- **`ft_2rag/`** → fine-tuned generator (textbook) + QA retrieval


---

## Core Components per Pipeline

Each RAG variant includes the following core modules:

| File | Description |
|------|--------------|
| `rag_main.py` | Entry point; runs the full RAG pipeline end-to-end |
| `chatbot.py` | Handles prompt construction, context injection, and response generation |
| `vectorstore.py` | Defines Chroma-based retrieval logic and embedding functions |
| `dataset.py` | Prepares dataset for vector DB population (QA or chunk-based) |
| `finetuned_model.py` | Loads the fine-tuned Phi-2 (if applicable) |
| `__init__.py` | Enables relative imports within the pipeline |

Each pipeline exposes a unified interface:
```python
from llmrag.pipelines.ft_1rag.rag_main import run_rag_pipeline

response, context = run_rag_pipeline(
    query="What is gradient descent?",
)
```

---

## Notes
- The baseline pipelines (B1/B2) serve as control experiments for isolating retrieval effects.
- The fine-tuned variants (FT-1/FT-2) analyze cross-domain generalization, swapping generator and retrieval domains.
- All pipeline components are modular and reusable for future domain-adaptation experiments.

![RAG Pipeline Overview](docs/figures/rag_pipelines.png)
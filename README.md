# LLMRAG — Journal Codebase (Clean Repo)

A structured, reproducible release of your RAG pipelines, fine-tuning, and evaluations.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Project layout
```
repo-root/
├─ src/llmrag/              # importable package: pipelines, training, eval, utils
├─ configs/                 # yaml/toml configs (paths, hyperparams)
├─ scripts/                 # CLI entrypoints (train, build index, run RAG, eval)
├─ data/                    # small samples only (no large corpora)
├─ tests/                   # minimal unit tests
├─ resources/               # tokenizer/llm scaffolding (no weights)
├─ model_cards/             # READMEs for released checkpoints
├─ assets/                  # small images/figures
├─ README.md, Makefile, pyproject.toml, .gitignore
```

## Large artifacts
- Model & embedding checkpoints → publish on HF Hub or GitHub Releases.
- Vector DB / FAISS / Chroma indexes → publish separately or via DVC.
- Add links in this README under **Releases**.

## Reproduce headline experiments
```bash
make build-index
make rag
make eval
```

## Citation
See `CITATION.cff`.

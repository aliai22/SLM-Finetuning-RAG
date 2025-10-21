# Domain Adaptation of Small Language Models via Fine-Tuning and RAG: A Case Study in Scientific Question Answering

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
├─ data/                    # small samples only (no large corpora)
├─ resources/               # tokenizer/llm scaffolding (no weights)
├─ README.md, Makefile, pyproject.toml, .gitignore
```

## Citation
See `CITATION.cff`.

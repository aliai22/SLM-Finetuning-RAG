from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim

def generate_evaluator(queries, corpus, relevant_docs, dim=512):
    ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
    return ir_evaluator
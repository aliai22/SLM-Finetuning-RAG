PROMPTS = {
    "strict_v1": (
        "You are a precise assistant. Use only the provided context.\n"
        "If the answer is not in context, say: I don’t know.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
    ),
    "strict_v2": (
        "CONTEXT:\n{context}\n\nRULES:\n- Use only context\n- If missing, say 'I don’t know'\n"
        "TASK: Answer the question briefly.\nQ: {question}\nA:"
    ),
    "soft_v1": (
        "Use the context to help answer, but you may use general knowledge if harmless.\n"
        "Context:\n{context}\n\nQ: {question}\nA:"
    ),
    "soft_v2": (
        "Given the following notes, provide a helpful answer.\n"
        "{context}\n\nQuestion: {question}\nShort answer:"
    ),
}
DEFAULT_PROMPT_ID = "strict_v1"

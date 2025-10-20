# contamination_audit.py
# Robust contamination audit for FT1 RAG
# - Evaluates 5-gram overlap and MinHash (LSH) matches between eval set and:
#   (a) fine-tuning QA corpus, (b) textbook vector DB (or raw chunk dumps)
#
# Usage:
#   python -m rag_eval_FT1.contamination_audit
#
# Notes:
#   - If your QA KB is a different file than the FT QA corpus, point KB_QA_JSONL to it.
#   - If your textbook KB path differs, update TEXTBOOK_CHUNKS_DIR.
#   - CSV is written to contamination_report.csv in this folder.

import os, re, json, glob, csv
from typing import List, Tuple
from datasketch import MinHash, MinHashLSH

# ======================== CONFIG ========================
EVAL_JSON            = "synthetic_QAs.json"                    # eval used in FT1/FT2 sensitivity
FT_QA_JSONL          = "uniqueQA_dataset0.7.jsonl"             # your SFT QA pairs (JSONL)
KB_QA_JSONL          = None                                    # set to a QA KB file if separate; else None -> reuse FT_QA_JSONL
TEXTBOOK_CHUNKS_DIR  = "./FT_2RAG/vecDB/QAs_v3"                # Chroma persist dir or raw chunk dumps
NGRAM_N              = 5
MINHASH_PERM         = 128
LSH_THRESH           = 0.8
OUT_CSV              = "contamination_report_FT2.csv"
# ========================================================


# -------------------- Utilities --------------------
def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def ngrams(s: str, n: int = NGRAM_N):
    toks = norm(s).split()
    if len(toks) < n:
        return set()
    return set(" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1))

def _parse_qa_from_string(s: str) -> Tuple[str, str]:
    """
    Accept strings like: 'Q: ...\nA: ...' (with newlines allowed).
    Returns (q, a) or (None, None).
    """
    if not isinstance(s, str):
        return None, None
    s = s.strip()
    m = re.search(r'Q\s*[:\-]\s*(.+?)\s*[\r\n]+A\s*[:\-]\s*(.+)\Z',
                  s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if "A:" in s:
        head, tail = s.split("A:", 1)
        q = re.sub(r'^\s*Q\s*[:\-]?\s*', '', head, flags=re.IGNORECASE).strip()
        a = tail.strip()
        if q and a:
            return q, a
    return None, None

def _extract_qapairs_from_entry(entry) -> List[Tuple[str, str]]:
    """
    Flexible extractor supporting shapes:
      - {'qa_pairs': [ {'Question':..., 'Answer':...} | 'Q:.. A:..' | ... ], 'input_text': ...}
      - {'question':..., 'answer':...}
      - {'Q':..., 'A':...}
      - {'qa_pairs': {...}} single dict
    """
    pairs = []
    if not isinstance(entry, dict):
        return pairs

    # Direct pair on the entry
    for qk, ak in [('Question', 'Answer'), ('question', 'answer'), ('Q', 'A')]:
        if qk in entry and ak in entry:
            q = str(entry[qk]).strip()
            a = str(entry[ak]).strip()
            if q and a:
                pairs.append((q, a))
                return pairs  # prefer explicit pair

    qa = entry.get('qa_pairs')
    if qa is None:
        return pairs
    if isinstance(qa, dict):
        qa = [qa]

    if isinstance(qa, list):
        for item in qa:
            if isinstance(item, dict):
                for qk, ak in [('Question', 'Answer'), ('question', 'answer'), ('Q', 'A')]:
                    q = str(item.get(qk, '')).strip()
                    a = str(item.get(ak, '')).strip()
                    if q and a:
                        pairs.append((q, a))
                        break
            elif isinstance(item, str):
                q, a = _parse_qa_from_string(item)
                if q and a:
                    pairs.append((q, a))
    return pairs

def load_eval_pairs(file_path: str) -> List[Tuple[str, str]]:
    """
    Robust loader for synthetic_QAs.json; returns list[(question, answer)].
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    if isinstance(data, list):
        for entry in data:
            pairs.extend(_extract_qapairs_from_entry(entry))
        return pairs

    if isinstance(data, dict):
        for key in ["data", "examples", "items"]:
            if key in data and isinstance(data[key], list):
                for entry in data[key]:
                    pairs.extend(_extract_qapairs_from_entry(entry))
                return pairs
        # Try directly
        pairs.extend(_extract_qapairs_from_entry(data))
        return pairs

    return pairs

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def collect_texts_from_QAjsonl(path: str) -> List[str]:
    """
    Collect question + answer strings from your QA JSONL (SFT).
    Accepts keys: question/input, answer/output.
    """
    items = load_jsonl(path)
    texts = []
    for it in items:
        q = it.get("question") or it.get("input") or ""
        a = it.get("answer") or it.get("output") or ""
        if isinstance(q, str) and q.strip():
            texts.append(q.strip())
        if isinstance(a, str) and a.strip():
            texts.append(a.strip())
    return texts

def collect_textbook_texts_from_dir(db_dir: str) -> List[str]:
    """
    Attempts to gather raw 'page_content' or 'documents' strings
    from a Chroma persist dir or any JSON/txt dumps you have there.
    """
    texts = []
    for fp in glob.glob(os.path.join(db_dir, "**", "*.*"), recursive=True):
        try:
            if fp.lower().endswith(".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    j = json.load(f)
                if isinstance(j, dict):
                    # Common Chroma serialized shapes
                    if "documents" in j and isinstance(j["documents"], list):
                        texts.extend([str(d) for d in j["documents"] if isinstance(d, (str, int, float)) or d])
                    elif "metadatas" in j and "embeddings" in j:
                        # ignore embeddings; some dumps have page_content inside metadatas
                        pass
                elif isinstance(j, list):
                    for item in j:
                        if isinstance(item, dict):
                            pc = item.get("page_content")
                            if isinstance(pc, str) and pc.strip():
                                texts.append(pc.strip())
            elif fp.lower().endswith((".txt", ".md")):
                with open(fp, "r", encoding="utf-8") as f:
                    texts.append(f.read())
        except Exception:
            # Skip unreadable/malformed files silently
            pass
    return texts

def make_minhash(sigtext: str) -> MinHash:
    mh = MinHash(num_perm=MINHASH_PERM)
    for tok in set(norm(sigtext).split()):
        mh.update(tok.encode("utf-8"))
    return mh


# -------------------- Load corpora --------------------
print("Loading evaluation pairs...")
eval_pairs = load_eval_pairs(EVAL_JSON)
eval_texts = [q for q, _ in eval_pairs] + [a for _, a in eval_pairs]
print(f"  Eval QA pairs: {len(eval_pairs)}  |  Eval texts: {len(eval_texts)}")

if not eval_pairs:
    print("WARNING: No eval pairs parsed from synthetic_QAs.json. Check file format/keys.")

qa_ft_texts = []
if FT_QA_JSONL and os.path.exists(FT_QA_JSONL):
    print(f"Loading FT QA jsonl: {FT_QA_JSONL}")
    qa_ft_texts = collect_texts_from_QAjsonl(FT_QA_JSONL)
else:
    print("WARNING: FT_QA_JSONL not found or unset; skipping FT QA corpus.")
print(f"  FT QA texts collected: {len(qa_ft_texts)}")

kb_qa_texts = []
if KB_QA_JSONL:
    if os.path.exists(KB_QA_JSONL):
        print(f"Loading KB QA jsonl: {KB_QA_JSONL}")
        kb_qa_texts = collect_texts_from_QAjsonl(KB_QA_JSONL)
    else:
        print("WARNING: KB_QA_JSONL path provided but file not found; skipping KB QA corpus.")
else:
    # Reuse FT QA if separate KB QA not provided
    kb_qa_texts = qa_ft_texts

print("Collecting textbook KB texts (this may take a moment)...")
textbook_texts = collect_textbook_texts_from_dir(TEXTBOOK_CHUNKS_DIR)
print(f"  Textbook KB texts collected: {len(textbook_texts)}")

corpora = {
    "ft_QA": qa_ft_texts,
    "kb_QA": kb_qa_texts,
    "kb_textbook": textbook_texts,
}


# -------------------- N-gram audit --------------------
def ngram_audit(eval_texts: List[str], corpus_texts: List[str], name: str):
    eval_ngrams = [ngrams(t) for t in eval_texts]
    corpus_ngrams = set()
    for t in corpus_texts:
        corpus_ngrams |= ngrams(t)

    hits = []
    for i, s in enumerate(eval_ngrams):
        inter = s & corpus_ngrams
        if inter:
            overlap_ratio = len(inter) / max(1, len(s))
            hits.append((i, len(inter), len(s), round(overlap_ratio, 4)))

    rate = len(hits) / max(1, len(eval_ngrams))
    avg_ratio = round(sum(h[3] for h in hits) / max(1, len(hits)), 4) if hits else 0.0
    res = {
        "name": name,
        "eval_count": len(eval_ngrams),
        "ngram_overlap_hits": len(hits),
        "pct_eval_with_overlap": round(100 * rate, 2),
        "avg_overlap_ratio": avg_ratio
    }
    return res

print("\nRunning 5-gram overlap audit...")
ngram_rows = []
for k, texts in corpora.items():
    row = ngram_audit(eval_texts, texts, f"5gram:{k}")
    print(row)
    ngram_rows.append(row)


# -------------------- MinHash LSH audit --------------------
def lsh_audit(eval_texts: List[str], corpus_texts: List[str], name: str):
    lsh = MinHashLSH(threshold=LSH_THRESH, num_perm=MINHASH_PERM)
    for idx, t in enumerate(corpus_texts):
        lsh.insert(f"c{idx}", make_minhash(t))

    matches = 0
    for t in eval_texts:
        if lsh.query(make_minhash(t)):
            matches += 1

    res = {
        "name": name,
        "eval_count": len(eval_texts),
        "minhash_perm": MINHASH_PERM,
        "minhash_thresh": LSH_THRESH,
        "num_eval_with_match": matches,
        "pct_eval_with_match": round(100 * matches / max(1, len(eval_texts)), 2)
    }
    return res

print("\nRunning MinHash/LSH audit...")
lsh_rows = []
for k, texts in corpora.items():
    row = lsh_audit(eval_texts, texts, f"MinHash:{k}")
    print(row)
    lsh_rows.append(row)


# -------------------- Write CSV --------------------
def _write_csv_safe(path: str, header: List[str], rows: List[dict]):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=header,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            doublequote=True,
            lineterminator="\n"
        )
        if f.tell() == 0:
            w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})

# Start fresh
if os.path.exists(OUT_CSV):
    try:
        os.remove(OUT_CSV)
    except Exception:
        pass

print(f"\nWriting CSV report to: {OUT_CSV}")
if ngram_rows:
    _write_csv_safe(OUT_CSV, list(ngram_rows[0].keys()), ngram_rows)
if lsh_rows:
    # Separate block with its own header for clarity
    with open(OUT_CSV, "a", encoding="utf-8") as f:
        f.write("\n")
    _write_csv_safe(OUT_CSV, list(lsh_rows[0].keys()), lsh_rows)

print("\n=== Summary ===")
print("N-gram rows written:", len(ngram_rows))
print("MinHash rows written:", len(lsh_rows))
print("Done.")

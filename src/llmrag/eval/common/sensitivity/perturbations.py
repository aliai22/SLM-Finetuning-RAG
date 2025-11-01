import random, re, string
from typing import Dict
import nltk
from nltk.corpus import wordnet as wn

# Ensure wordnet is available (safe no-op if already present)
try:
    wn.ensure_loaded()
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

_QW = {"what","why","when","who","whom","which","how","where"}

def _tokenize(t:str): 
    return re.findall(r"\w+|[^\w\s]", t, re.UNICODE)

def _is_content_word(tok:str):
    if not tok.isalpha(): return False
    if tok.lower() in _QW: return False
    return len(tok) > 3

def _synonym(word:str):
    syns = set()
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            w = l.name().replace('_',' ')
            if w.lower()!=word.lower() and w.isalpha():
                syns.add(w)
    # prefer similar-length synonyms
    syns = sorted(syns, key=lambda w: abs(len(w)-len(word)))
    return syns[0] if syns else None

# --- Variants ---

def light_noise(q:str)->str:
    """Random casing + 1-2 keyboard-style typos + punctuation shuffle."""
    toks = _tokenize(q)
    # random cap/lower
    toks = [t.upper() if random.random()<0.1 else t.lower() if random.random()<0.1 else t for t in toks]
    # inject up to 2 simple typos (duplicate or drop char)
    s = "".join(toks)
    for _ in range(random.randint(1,2)):
        if len(s)<4: break
        i = random.randint(1,len(s)-2)
        if s[i].isalpha():
            s = s[:i] + (s[i]*2 if random.random()<0.5 else "") + s[i+1:]
    # occasional punctuation swap (.,?)
    s = s.replace(" .",".").replace(" ,",",")
    return s

def synonym_swap(q:str, max_swaps:int=3)->str:
    toks = _tokenize(q)
    idxs = [i for i,t in enumerate(toks) if _is_content_word(t)]
    random.shuffle(idxs)
    swaps = 0
    for i in idxs:
        if swaps>=max_swaps: break
        syn = _synonym(toks[i])
        if syn:
            # Keep original capitalization
            syn = syn.capitalize() if toks[i][0].isupper() else syn
            toks[i] = syn
            swaps += 1
    return "".join(toks)

def naive_paraphrase(q:str)->str:
    """
    Lightweight, deterministic paraphrase (no external LLM):
    - reorder simple clauses
    - replace common cue words
    """
    s = q.strip()
    replacements = {
        r"\bprovide\b":"give",
        r"\bexplain\b":"describe",
        r"\bimpact\b":"effect",
        r"\busing\b":"with",
        r"\bbased on\b":"according to",
        r"\bcompare\b":"contrast"
    }
    for pat,rep in replacements.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    # move simple preamble like "In brief," to the end
    s = re.sub(r"^(in brief|in short|generally),?\s+", "", s, flags=re.IGNORECASE)
    if "," in s and len(s.split(","))>1:
        first, rest = s.split(",",1)
        s = f"{rest.strip()}, {first.strip()}"
    return s

def make_variants(q:str)->Dict[str, Dict]:
    return {
        "orig": {"text": q, "severity": "base"},
        "noise": {"text": light_noise(q), "severity": "low"},
        "syn": {"text": synonym_swap(q), "severity": "medium"},
        "para": {"text": naive_paraphrase(q), "severity": "high"},
    }

# src/llmrag/utils/runtime.py
import os
import platform
from typing import Optional, Dict, Any

import torch

try:
    from transformers import BitsAndBytesConfig  # optional
    _HAVE_BNB = True
except Exception:
    _HAVE_BNB = False


def has_gpu() -> bool:
    return torch.cuda.is_available()


def prefer_bf16() -> bool:
    """True on modern GPUs with bf16 support (Ampere+), otherwise False."""
    if not has_gpu():
        return False
    # heuristic: bf16 widely OK on Ampere (8.0+) and newer; if unsure, let user override via env
    return os.getenv("LLMRAG_FORCE_FP16", "0") != "1"


def auto_dtype() -> torch.dtype:
    if has_gpu():
        return torch.bfloat16 if prefer_bf16() else torch.float16
    return torch.float32


def maybe_bnb_config() -> Optional[BitsAndBytesConfig]:
    """
    Return a BitsAndBytesConfig if:
      - LLMRAG_USE_4BIT=1
      - bitsandbytes is installed
      - running on Linux (Windows/macOS wheels are fragile)
      - GPU is available
    Otherwise return None.
    Users can force enable on other OS via LLMRAG_ALLOW_BNB_ON_ANY_OS=1.
    """
    use_4bit = os.getenv("LLMRAG_USE_4BIT", "0") == "1"
    if not use_4bit:
        return None
    if not _HAVE_BNB:
        return None
    if not has_gpu():
        return None
    if platform.system() != "Linux" and os.getenv("LLMRAG_ALLOW_BNB_ON_ANY_OS", "0") != "1":
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if prefer_bf16() else torch.float16,
        bnb_4bit_quant_type="nf4",
    )


def from_pretrained_kwargs() -> Dict[str, Any]:
    """
    Consistent kwargs for model.from_pretrained across environments.
    - Device map only when GPU exists.
    - Dtype chosen automatically.
    - Quantization only when safe/explicit.
    """
    kwargs: Dict[str, Any] = {
        "torch_dtype": auto_dtype(),
    }
    if has_gpu():
        kwargs["device_map"] = "auto"
    qcfg = maybe_bnb_config()
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    return kwargs
"""Load HF causal LM + tokenizer, resolve transformer layer modules."""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer from HuggingFace.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or local path.
    dtype : torch.dtype
        Torch dtype for model weights (bfloat16 for inference, float32 for CPU).
    device : str
        Target device ("cuda", "cpu", "mps", etc.).

    Returns
    -------
    tuple[Any, Any]
        (model, tokenizer) ready for inference.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok


def get_transformer_layers(model: Any) -> list[Any]:
    """Return the list of transformer block modules.

    Parameters
    ----------
    model : Any
        A HuggingFace CausalLM model (GPT-NeoX/Pythia, GPT-2, or Llama/Mistral).

    Returns
    -------
    list[Any]
        Transformer block modules in layer order.

    Raises
    ------
    RuntimeError
        If the model architecture is not recognized.
    """
    # GPT-NeoX / Pythia
    if hasattr(model, "gpt_neox"):
        return list(model.gpt_neox.layers)
    # GPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # Llama / Mistral
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise RuntimeError(f"Unknown layer layout for {type(model).__name__}")

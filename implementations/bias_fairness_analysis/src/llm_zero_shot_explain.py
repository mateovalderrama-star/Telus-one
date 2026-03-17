"""LLM zero-shot scoring with Integrated Gradients explanations.

Performs zero-shot scoring on social-problem data using LLMs and computes
Integrated Gradients (IG) for interpretability.

Key features:
- IG score depends on the SAME forward using `inputs_embeds` for gradients.
- Dtype consistency (avoid Float vs Half LayerNorm crash).
- Uses `dtype=` (not deprecated `torch_dtype`).
- Parquet read guard (falls back to CSV if no engine is present).

Example usage:
uv run python scripts/llm_zero_shot_explain.py
  --in data/jigsaw_sample.parquet
  --text_col comment_text
  --task toxicity
  --out outputs/zs_preds.parquet
  --model distilgpt2
  --max_rows 1000
  --ig_rows 25 --ig_steps 32 --save_heatmaps
  --force_float32
  --label_col target
  --id_cols male female black white muslim jewish

"""

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from aieng.llm_interp.utils import get_device
from rich.progress import track
from torch.nn import functional
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = {
    # Binary label pairs; first entry treated as "positive" for score sign
    "toxicity": ["toxic", "non-toxic"],
    "hate": ["hateful", "not hateful"],
    "offense": ["offensive", "not offensive"],
}


def load_llm(model_name: str, device: torch.device, force_float32: bool = False) -> tuple[Any, Any]:
    """
    Load a pre-trained LLM model and its corresponding tokenizer.

    Configures the model for evaluation, moves it to the specified device,
    and sets the appropriate data type. It also ensures a padding token is
    assigned if missing.

    Parameters
    ----------
    model_name : str
        The Hugging Face model identifier to load.
    device : torch.device
        The device (CPU or CUDA) to load the model onto.
    force_float32 : bool, default False
        If True, forces the model into float32 precision regardless of device.

    Returns
    -------
    tuple[Any, Any]
        A pair containing (model, tokenizer).
    """
    # Load tokenizer + model; float32 optional for stable grads (LayerNorm FP16 issues)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token  # Ensure padding token exists
    load_dtype = torch.float32 if (force_float32 or device.type != "cuda") else torch.float16
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    # Some models crash returning big intermediates with cache on; harmless to disable
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False  # Disable KV cache (not needed for scoring / IG)
    return model, tok


def format_prompt(text: str, task: str) -> str:
    """
    Wrap the input text in a task-specific instruction prompt.

    Parameters
    ----------
    text : str
        The user-provided content to classify.
    task : str
        The classification task identifier (e.g., 'toxicity', 'hate', 'offense').

    Returns
    -------
    str
        The formatted prompt string ending with a 'Label:' indicator.
    """
    # Simple instruction + user text -> ensures final token is "Label:" to predict after
    instruction = {
        "toxicity": "Decide if the following text is toxic or non-toxic. Answer with a single word.\nText: ",
        "hate": "Decide if the text is hateful or not hateful. Answer with a single word.\nText: ",
        "offense": "Decide if the text is offensive or not offensive. Answer with a single word.\nText: ",
    }[task]
    return f"{instruction}{text}\nLabel:"


@torch.no_grad()  # type: ignore[misc]
def label_logprob(model: Any, tok: Any, prompt_ids: torch.Tensor, label_str: str) -> float:
    """
    Compute the total log probability of a target label string given a prompt.

    Uses teacher forcing to calculate the combined log probability across all
    tokens that make up the target label.

    Parameters
    ----------
    model : Any
        The loaded LLM.
    tok : Any
        The tokenizer corresponding to the model.
    prompt_ids : torch.Tensor
        Tokenized prompt sequence.
    label_str : str
        The string label whose probability is to be calculated.

    Returns
    -------
    float
        The total log probability of the label.
    """
    label_ids = tok.encode(label_str, add_special_tokens=False)

    cur_ids = prompt_ids.clone()
    logp = 0.0

    for lid in label_ids:
        out = model(input_ids=cur_ids)
        next_logits = out.logits[:, -1, :]
        logp += functional.log_softmax(next_logits, dim=-1)[0, lid]

        cur_ids = torch.cat(
            [cur_ids, torch.tensor([[lid]], device=cur_ids.device)],
            dim=1,
        )

    return float(logp)


def score_and_predict(model: Any, tok: Any, text: str, task: str) -> dict[str, Any]:
    """
    Predict a label and calculate a confidence score for a given text.

    The score is calculated as the log probability difference between the
    positive and negative label options.

    Parameters
    ----------
    model : Any
        The loaded LLM.
    tok : Any
        The tokenizer.
    text : str
        The text to be scored.
    task : str
        The classification task identifier.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the formatted prompt, confidence score,
        binary prediction, and individual log probabilities.
    """
    # Score = log p(pos_label) - log p(neg_label); sign => prediction
    prompt = format_prompt(text, task)
    batch = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    prompt_ids = batch["input_ids"]
    y_pos, y_neg = LABELS[task]
    lp_pos = label_logprob(model, tok, prompt_ids, y_pos)
    lp_neg = label_logprob(model, tok, prompt_ids, y_neg)
    score = lp_pos - lp_neg
    pred = 1 if score > 0 else 0  # Positive if pos_label more likely
    return {
        "prompt": prompt,
        "score": score,
        "pred": pred,
        "labels": (y_pos, y_neg),
        "lp_pos": lp_pos,
        "lp_neg": lp_neg,
    }


def integrated_gradients(
    model: Any, tok: Any, text: str, task: str, steps: int = 32
) -> tuple[list[str], npt.NDArray[np.floating[Any]], str, float]:
    """
    Perform Integrated Gradients to identify which tokens influenced the model output.

    Calculates the importance of each token by integrating gradients along
    a path from a baseline (zero) input to the actual token embeddings.

    Parameters
    ----------
    model : Any
        The loaded LLM.
    tok : Any
        The tokenizer.
    text : str
        The text to explain.
    task : str
        The classification task.
    steps : int, default 32
        The number of integration steps to perform (higher is more accurate).

    Returns
    -------
    tuple[list[str], npt.NDArray, str, float]
        A tuple of (tokens, attribution_scores, full_prompt, model_score).
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    prompt = format_prompt(text, task)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    emb_layer = model.get_input_embeddings()
    x = emb_layer(input_ids).detach().to(model_dtype)
    x0 = torch.zeros_like(x)

    pos_label, neg_label = LABELS[task]
    pos_ids = tok.encode(pos_label, add_special_tokens=False)
    neg_ids = tok.encode(neg_label, add_special_tokens=False)

    def full_label_logprob(emb: torch.Tensor, label_ids: list[int]) -> torch.Tensor:
        """Compute log p(full_label | prompt) using teacher forcing.

        Only prompt embeddings get gradients.
        """
        cur_emb = emb
        cur_attn = attn.clone()
        total_logprob = torch.tensor(0.0, device=device, dtype=model_dtype)

        for lid in label_ids:
            out = model(inputs_embeds=cur_emb, attention_mask=cur_attn, use_cache=False)
            logits = out.logits[:, -1, :]
            logprobs = functional.log_softmax(logits, dim=-1)
            total_logprob = total_logprob + logprobs[0, lid]

            # Append next label token as *constant* embedding
            next_token = torch.tensor([[lid]], device=device)
            next_emb = emb_layer(next_token).to(model_dtype)

            cur_emb = torch.cat([cur_emb, next_emb], dim=1)
            cur_attn = torch.cat(
                [cur_attn, torch.ones((1, 1), device=device, dtype=cur_attn.dtype)],
                dim=1,
            )

        return total_logprob

    def score_fn(emb: torch.Tensor) -> torch.Tensor:
        pos_score = full_label_logprob(emb, pos_ids)
        neg_score = full_label_logprob(emb, neg_ids)
        return pos_score - neg_score

    # ----- Integrated Gradients -----
    alphas = torch.linspace(0, 1, steps=steps, device=device, dtype=model_dtype).view(-1, 1, 1, 1)

    grads = torch.zeros_like(x)

    for a in alphas:
        emb = (x0 + a * (x - x0)).requires_grad_(True)
        s = score_fn(emb)
        s.backward()
        grads += emb.grad.detach()

    avg_grads = grads / steps
    atts = (avg_grads * (x - x0)).sum(dim=-1).squeeze(0)
    atts = atts / (atts.abs().sum() + 1e-8)

    tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())

    with torch.no_grad():
        explained_score = float(score_fn(x))

    return tokens, atts.cpu().numpy(), prompt, explained_score


def save_heatmap(tokens: list[str], atts: npt.NDArray[np.floating[Any]], out_path: str) -> None:
    """
    Create and save a visualization of the token attribution scores.

    Parameters
    ----------
    tokens : list[str]
        List of tokens extracted from the text.
    atts : npt.NDArray
        Normalization attribution scores for each token.
    out_path : str
        File path where the plot image will be saved.

    Returns
    -------
    None
    """
    # Simple bar plot; token strings lightly cleaned for readability
    plt.figure(figsize=(max(6, len(tokens) * 0.2), 2.8))
    plt.bar(range(len(tokens)), atts)
    plt.xticks(
        range(len(tokens)),
        [t.replace("Ġ", "▯") for t in tokens],
        rotation=70,
        ha="right",
    )
    plt.ylabel("IG attribution")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_df_safely(path: str) -> pd.DataFrame:
    """
    Load a dataset while gracefully handling potential format issues.

    Parameters
    ----------
    path : str
        Path to the Parquet or CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    # Attempt Parquet; fallback to CSV if engine missing or read fails
    try:
        if path.endswith(".parquet"):
            if not (importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")):
                raise ImportError("No parquet engine (pyarrow/fastparquet) found.")
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] Failed to read '{path}' as Parquet ({e}); trying CSV…")
        return pd.read_csv(path)


def main() -> None:
    """
    Execute the full zero-shot scoring and interpretability pipeline.

    Orchestrates loading inputs, batch processing predictions through the LLM,
    optionally computing Integrated Gradients for a subset of samples,
    and saving results to disk.
    """
    # Parse CLI, load data/model, loop over rows, optional IG subset
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Parquet/CSV file with a text column")
    ap.add_argument("--text_col", required=True)
    ap.add_argument("--task", required=True, choices=list(LABELS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--max_rows", type=int, default=2000)
    ap.add_argument("--ig_rows", type=int, default=25, help="How many rows to run IG on")
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--save_heatmaps", action="store_true")
    ap.add_argument("--label_col", default=None, help="Copy label column from input into preds")
    ap.add_argument(
        "--id_cols",
        nargs="*",
        default=None,
        help="Copy identity columns from input into preds",
    )
    ap.add_argument(
        "--force_float32",
        action="store_true",
        help="Load model in float32 (safer for IG / LayerNorm).",
    )
    args = ap.parse_args()

    df = load_df_safely(args.inp)
    if len(df) > args.max_rows:
        df = df.iloc[: args.max_rows].copy()  # Hard cap for quick experimentation

    device = get_device()
    model, tok = load_llm(args.model, device, force_float32=args.force_float32)

    preds: list[dict[str, Any]] = []
    ig_records: list[dict[str, Any]] = []
    extra_cols = []
    if args.label_col and args.label_col in df.columns:
        extra_cols.append(args.label_col)
    if args.id_cols:
        for c in args.id_cols:
            if c in df.columns:
                extra_cols.append(c)

    for i in track(range(len(df)), description="Scoring"):
        # Truncate overly long texts (context length safety)
        text = str(df.iloc[i][args.text_col])[:4096]
        res = score_and_predict(model, tok, text, args.task)
        row = {
            "idx": i,
            "pred": res["pred"],
            "score": res["score"],
            "lp_pos": res["lp_pos"],
            "lp_neg": res["lp_neg"],
        }
        for c in extra_cols:
            row[c] = df.iloc[i][c]
        preds.append(row)

        if args.save_heatmaps and len(ig_records) < args.ig_rows:
            # Only compute IG for first N rows (expensive)
            toks, atts, prompt, _ = integrated_gradients(model, tok, text, args.task, steps=args.ig_steps)
            img_path = Path("outputs/ig_heatmaps") / f"row{i}.png"
            save_heatmap(toks, atts, str(img_path))
            ig_records.append({"idx": i, "heatmap": str(img_path), "prompt": prompt})

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_parquet(args.out, index=False)
    if ig_records:
        pd.DataFrame(ig_records).to_parquet(Path(args.out).with_suffix(".ig.parquet"), index=False)
    print(f"Saved predictions -> {args.out}")


if __name__ == "__main__":
    main()

"""Utilities for model inference, checkpointing, and record preparation."""

import glob
import json
import os
import re
from pathlib import Path
from typing import Any

import jsonlines
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def clean_json_output(text: str) -> str:
    """
    Strip code formatting artifacts to isolate a raw JSON string.

    Parameters
    ----------
    text : str
        The raw output from a model, potentially containing markdown fences.

    Returns
    -------
    str
        The cleaned JSON string.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _suffix_num(path) -> int:
    try:
        return int(Path(path).stem.split("_")[-1])
    except Exception:
        return -1


def save_checkpoint(scenes: list[dict], task_name: str, checkpoint_dir: str, step: int) -> None:
    """
    Persist the current state of a long-running inference task.

    Parameters
    ----------
    scenes : list of dict
        The current list of generated responses and metadata.
    task_name : str
        A unique name for the inference task.
    checkpoint_dir : str
        Directory where checkpoint files will be written.
    step : int
        The current iteration or record index.

    Returns
    -------
    None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_{task_name}_{step}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)
    print(f"[ckpt] saved → {path}")


def load_checkpoint(task_name: str, checkpoint_dir: str) -> tuple[list[dict], int]:
    """
    Recover the most recent state for a specific inference task.

    Parameters
    ----------
    task_name : str
        The name of the task to resume.
    checkpoint_dir : str
        The folder to search for checkpoint files.

    Returns
    -------
    scenes : list of dict
        The records updated so far.
    last_idx : int
        The index of the last processed record. Returns -1 if no checkpoint exists.
    """
    pattern = os.path.join(checkpoint_dir, f"ckpt_{task_name}_*.json")
    files = glob.glob(pattern)
    if not files:
        return [], -1

    latest = max(files, key=_suffix_num)
    with open(latest, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    last_idx = max((s.get("prompt_idx", -1) for s in scenes), default=-1)
    print(f"[ckpt] loaded ← {latest} (last_idx={last_idx})")
    return scenes, last_idx


def apply_chat_template(prompt: str, tokenizer: AutoTokenizer) -> str:
    """
    Convert a raw text prompt into a model-specific conversational format.

    Parameters
    ----------
    prompt : str
        The textual instruction.
    tokenizer : AutoTokenizer
        The tokenizer providing the chat template logic.

    Returns
    -------
    str
        The formatted prompt string with 'user' and 'system' tags applied.
    """
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )


def prepare_record(rec: dict, tokenizer: AutoTokenizer) -> tuple[dict[Any, Any], str]:
    """
    Isolate metadata from the prompt and format the prompt for models.

    Parameters
    ----------
    rec : dict
        A raw dataset record.
    tokenizer : AutoTokenizer
        The tokenizer for applying templates.

    Returns
    -------
    meta : dict
        The non-prompt fields of the record.
    prompt : str
        The template-formatted prompt string.
    """
    meta = {k: rec[k] for k in rec if k != "prompt"}
    prompt = apply_chat_template(rec["prompt"], tokenizer)
    return meta, prompt


def load_disk_records(path: str, limit: int = 200) -> list[dict[Any, Any]]:
    """
    Load a sampling of records from a local Hugging Face dataset.

    Parameters
    ----------
    path : str
        Location of the dataset on disk.
    limit : int, default 200
        Maximum number of samples to load.

    Returns
    -------
    list of dict
        The extracted records.
    """
    ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        ds = ds["train"]

    return [ds[i] for i in range(min(limit, len(ds)))]


def load_arrow_records(path: str, limit: int = 200) -> list[dict[Any, Any]]:
    """
    Load a sampling of records from a raw Arrow file.

    Parameters
    ----------
    path : str
        Path to the .arrow file.
    limit : int, default 200
        Maximum number of samples to load.

    Returns
    -------
    list of dict
        The extracted records.
    """
    ds = load_dataset("arrow", data_files=path, split="train")
    ds = ds.select(range(min(limit, len(ds))))
    return [dict(r) for r in ds]


QA_PATTERN = re.compile(
    r"Question:\s*(.*?)\s*Answer 1:\s*(.*?)\s*Answer 2:\s*(.*)",
    re.DOTALL,
)


def build_prompt_records(
    dataset: list[dict],
    templates: dict[str, list],
    template_key: str,
    reverse: bool = False,
) -> list[Any]:
    """
    Synthesize model-ready prompts from a raw dataset using templates.

    Parameters
    ----------
    dataset : list of dict
        The raw dataset items.
    templates : dict
        A library of prompt template fragments.
    template_key : str
        The specific template ID to use.
    reverse : bool, default False
        If True, the labeling hint in the prompt is flipped.

    Returns
    -------
    list of dict
        Formatted records including the final prompt and metadata.
    """
    tpl = templates[template_key]
    records = []

    for i, raw_item in enumerate(dataset):
        item = dict(raw_item)
        chosen = item.get("chosen_id", item.get("chosen"))

        if not all(k in item for k in ("q", "r1", "r2")):
            m = QA_PATTERN.search(item["prompt"])
            item["q"], item["r1"], item["r2"] = m.group(1), m.group(2), m.group(3)

        prompt = tpl[0] + item["q"] + tpl[1] + item["r1"] + item["r2"]
        hint = chosen if not reverse else (3 - chosen)
        prompt += tpl[3] + str(hint) + tpl[4]

        records.append(
            {
                "prompt_idx": i,
                "prompt": prompt,
                "chosen": chosen,
                "meta": item,
            }
        )

    return records


def run_best_of_n(
    records,
    model,
    tokenizer,
    output_path,
    checkpoint_dir,
    task_name,
    n=8,
    checkpoint_every=5,
    max_new_tokens=512,
    prompt_max_len=6400,
) -> None:
    """
    Perform Best-of-N generation with sampling and checkpointing.

    Generates multiple candidates for each prompt, allowing for downstream
    preference selection.

    Parameters
    ----------
    records : list
        The input prompt records.
    model : object
        The causal LLM for generation.
    tokenizer : object
        The corresponding tokenizer.
    output_path : str
        Path to save the final JSONL results.
    checkpoint_dir : str
        Folder for intermediate progress saves.
    task_name : str
         Label for the checkpointing system.
    n : int, default 8
        The number of samples to generate per prompt.
    checkpoint_every : int, default 5
        Frequency of checkpointing (in records).
    max_new_tokens : int, default 512
        Maximum length of the generated output.
    prompt_max_len : int, default 6400
        Hard limit for context truncation.

    Returns
    -------
    None
    """
    scenes, last_idx = load_checkpoint(task_name, checkpoint_dir)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = jsonlines.open(output_path, "w")

    for i, rec in enumerate(tqdm(records, desc="Best-of-N")):
        if i <= last_idx:
            continue

        meta, prompt = prepare_record(rec, tokenizer)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=prompt_max_len,
        ).to(model.device)

        expanded = {k: v.repeat(n, 1) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **expanded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.2,
                top_p=0.95,
            )

        in_len = inputs["attention_mask"].sum(dim=1)[0].item()
        gens = [
            clean_json_output(tokenizer.decode(out[j, in_len:], skip_special_tokens=True)) for j in range(out.size(0))
        ]

        scenes.append(
            {
                "prompt_idx": i,
                "prompt": prompt,
                "outputs": gens,
                "meta": meta,
            }
        )

        if len(scenes) % checkpoint_every == 0:
            save_checkpoint(scenes, task_name, checkpoint_dir, len(scenes))

    for s in scenes:
        writer.write(s)
    writer.close()


def run_batched_inference(
    records,
    model,
    tokenizer,
    batch_size=4,
    max_new_tokens=512,
) -> list[dict[Any, Any]]:
    """
    Excute deterministic inference using batched inputs for efficiency.

    Parameters
    ----------
    records : list of dict
        The input prompt records.
    model : object
        The model to use.
    tokenizer : object
        The tokenizer to use.
    batch_size : int, default 4
        Number of prompts to process in a single GPU pass.
    max_new_tokens : int, default 512
        Length limit for generation.

    Returns
    -------
    list of dict
        A collection of generation results (ID, prompt, output, and meta).
    """
    results = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prompts = [r["prompt"] for r in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=6400,
        ).to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gens = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for rec, gen in zip(batch, gens):
            results.append(
                {
                    "prompt_idx": rec["prompt_idx"],
                    "prompt": rec["prompt"],
                    "output": gen,
                    "meta": rec["meta"],
                }
            )

    return results

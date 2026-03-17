"""Helper functions for dataset loading, extraction, and processing."""

import os
import random
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset


def set_seed(seed: int = 2021) -> None:
    """
    Set random seeds for all relevant libraries to ensure reproducibility.

    Parameters
    ----------
    seed : int, default 2021
        The seed value to use for random number generators.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_judge_template() -> list[str]:
    """
    Provide the prompt fragments for building an LLM judge evaluation.

    Returns
    -------
    list of str
        The template components (prefix and separators) for the judge prompt.
    """
    return [
        """As an evaluation expert, given a question and its two possible answers, please choose which answer better satisfies coherence, accuracy, coverage, and the overall quality defined above.
Please output your judgment in JSON format, where "reason" is your explanation and "better_answer" is an integer value of 1 or 2, for example:
{"reason": "your explanation", "better_answer": 1}.
Below are the question and the candidate answers:
\nQuestion:""",
        "\nAnswer 1:",
        "\nAnswer 2:",
    ]


def load_parquet_dataset(parquet_path: str) -> Dataset:
    """
    Load a dataset from a parquet file on disk.

    Parameters
    ----------
    parquet_path : str
        The absolute path to the .parquet file.

    Returns
    -------
    Dataset
        The 'train' split of the loaded dataset.
    """
    ds_dict = load_dataset("parquet", data_files={"train": parquet_path})
    return ds_dict["train"]


def extract_qa(item, chosen_key, rejected_key, dataset_format: str) -> tuple[Any, Any | Literal[""], Any | Literal[""]]:
    """
    Parse a dataset item to obtain the question and both answer candidates.

    Supports structured chat formats and raw conversation strings.

    Parameters
    ----------
    item : dict
        A single record from the dataset.
    chosen_key : str
        The key containing the preferred response.
    rejected_key : str
        The key containing the non-preferred response.
    dataset_format : {'sky', 'hh'}
        The schema identifying how content is stored.

    Returns
    -------
    question : str
        The user's query or instruction.
    ans_chosen : str
        The preferred response content.
    ans_rejected : str
        The rejected response content.

    Raises
    ------
    ValueError
        If an unrecognized `dataset_format` is provided.
    """
    if dataset_format == "sky":
        question = item[chosen_key][0]["content"]
        ans_chosen = item[chosen_key][-1]["content"]
        ans_rejected = item[rejected_key][-1]["content"]

    elif dataset_format == "hh":

        def split_qa(text) -> tuple[Any, Any | Literal[""]]:
            parts = text.split("Assistant:", 1)
            human = parts[0].replace("Human:", "").strip()
            assistant = parts[1].strip() if len(parts) > 1 else ""
            return human, assistant

        question, ans_chosen = split_qa(item[chosen_key])
        _, ans_rejected = split_qa(item[rejected_key])

    else:
        raise ValueError("Unsupported dataset_format. Use 'sky' or 'hh'.")

    return question, ans_chosen, ans_rejected


def build_judge_dataset(dataset, dataset_format: str, tag: str = "reward-bench") -> Dataset:
    """
    Create a new dataset specifically formatted for judge inference.

    Each item in the original dataset is transformed into a prompt that
    asks an LLM to compare two potentially shuffled answers.

    Parameters
    ----------
    dataset : Dataset or list
        The source dataset containing QA pairs.
    dataset_format : str
        The format type ('sky' or 'hh').
    tag : str, default 'reward-bench'
        A descriptive label for this evaluation run.

    Returns
    -------
    Dataset
        A Hugging Face Dataset object with judge-ready prompts and metadata.
    """
    template = get_judge_template()

    new_data = {
        "prompt": [],
        "tag": [],
        "test_id": [],
        "chosen": [],
        "q": [],
        "r1": [],
        "r2": [],
    }

    for idx, item in enumerate(dataset):
        if random.random() < 0.5:
            chosen_key, rejected_key = "chosen", "rejected"
            label = 1
        else:
            chosen_key, rejected_key = "rejected", "chosen"
            label = 2

        question, ans_1, ans_2 = extract_qa(item, chosen_key, rejected_key, dataset_format)

        prompt = template[0] + question + template[1] + ans_1 + template[2] + ans_2

        new_data["prompt"].append(prompt)
        new_data["q"].append(question)
        new_data["r1"].append(ans_1)
        new_data["r2"].append(ans_2)
        new_data["chosen"].append(label)
        new_data["tag"].append(tag)
        new_data["test_id"].append(idx)

    return Dataset.from_dict(new_data)


def save_dataset(dataset: Dataset, save_dir: str) -> DatasetDict:
    """
    Export a dataset to a directory on the local filesystem.

    Parameters
    ----------
    dataset : Dataset
        The dataset object to save.
    save_dir : str
        The target directory for storage.

    Returns
    -------
    DatasetDict
        The dataset wrapped in a DatasetDict container.
    """
    os.makedirs(save_dir, exist_ok=True)
    ds_dict = DatasetDict({"train": dataset})
    ds_dict.save_to_disk(save_dir)
    return ds_dict


def preview_samples(dataset: Dataset, n: int = 3) -> None:
    """
    Print the contents of the first few prompts in a dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to inspect.
    n : int, default 3
        The number of samples to display.

    Returns
    -------
    None
    """
    print("\n Sample examples:")
    for i in range(min(n, len(dataset))):
        print(f"\n--- Sample {i} ---")
        print(dataset[i]["prompt"])

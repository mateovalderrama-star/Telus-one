# dataset_helpers.py

import os
import random
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset

def set_seed(seed: int = 2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_judge_template():
    return [
        """As an evaluation expert, given a question and its two possible answers, please choose which answer better satisfies coherence, accuracy, coverage, and the overall quality defined above.
Please output your judgment in JSON format, where "reason" is your explanation and "better_answer" is an integer value of 1 or 2, for example:
{"reason": "your explanation", "better_answer": 1}.
Below are the question and the candidate answers:
\nQuestion:""",
        "\nAnswer 1:",
        "\nAnswer 2:",
    ]

def load_parquet_dataset(parquet_path: str):
    ds_dict = load_dataset(
        "parquet",
        data_files={"train": parquet_path}
    )
    return ds_dict["train"]


def extract_qa(item, chosen_key, rejected_key, dataset_format: str):
    """
    Extract question and answers depending on dataset format.

    dataset_format:
        - "sky"  : structured chat format (list of dicts)
        - "hh"   : raw string conversation format
    """

    if dataset_format == "sky":
        question = item[chosen_key][0]["content"]
        ans_chosen = item[chosen_key][-1]["content"]
        ans_rejected = item[rejected_key][-1]["content"]

    elif dataset_format == "hh":
        def split_qa(text):
            parts = text.split("Assistant:", 1)
            human = parts[0].replace("Human:", "").strip()
            assistant = parts[1].strip() if len(parts) > 1 else ""
            return human, assistant

        question, ans_chosen = split_qa(item[chosen_key])
        _, ans_rejected = split_qa(item[rejected_key])

    else:
        raise ValueError("Unsupported dataset_format. Use 'sky' or 'hh'.")

    return question, ans_chosen, ans_rejected


def build_judge_dataset(dataset, dataset_format: str, tag: str = "reward-bench"):
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

    idx = 0
    for item in dataset:
        if random.random() < 0.5:
            chosen_key, rejected_key = "chosen", "rejected"
            label = 1
        else:
            chosen_key, rejected_key = "rejected", "chosen"
            label = 2

        question, ans_1, ans_2 = extract_qa(
            item,
            chosen_key,
            rejected_key,
            dataset_format
        )

        prompt = (
            template[0]
            + question
            + template[1]
            + ans_1
            + template[2]
            + ans_2
        )

        new_data["prompt"].append(prompt)
        new_data["q"].append(question)
        new_data["r1"].append(ans_1)
        new_data["r2"].append(ans_2)
        new_data["chosen"].append(label)
        new_data["tag"].append(tag)
        new_data["test_id"].append(idx)

        idx += 1

    return Dataset.from_dict(new_data)



def save_dataset(dataset: Dataset, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    ds_dict = DatasetDict({"train": dataset})
    ds_dict.save_to_disk(save_dir)
    return ds_dict


def preview_samples(dataset: Dataset, n: int = 3):
    print("\n Sample examples:")
    for i in range(min(n, len(dataset))):
        print(f"\n--- Sample {i} ---")
        print(dataset[i]["prompt"])

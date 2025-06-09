# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json

from datasets import load_dataset
from transformers import AutoTokenizer


rnd_seed = 42

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")


# Function to tokenize and count tokens for batches
def count_tokens(batch):
    tokens = tokenizer(batch['text'], truncation=False, padding=False)
    return {"num_tokens": [len(token) for token in tokens["input_ids"]]}


def get_filtered_github_dataset():
    ds = load_dataset(
        "codeparrot/codeparrot-clean",
        split="train",
        # languages=["Python"],
        # downlaod_mode="force_redownload",
        num_proc=32,
    )
    ds = ds.rename_column("content", "text")
    ds = ds.map(count_tokens, batched=True, batch_size=128)
    return ds.filter(lambda example: 200 <= example['num_tokens'] <= 1000)


def save_hf_dataset_in_jsonl(data, output_dir):
    for name, samples in data.items():
        out_file = os.path.join(output_dir, f"{name}.jsonl")
        saved_cnt = 0
        with open(out_file, "w") as wf:
            for row in samples:
                wf.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                saved_cnt += 1
        print(f"[Saving] saved {saved_cnt} to {out_file}")


def process_github(base_dir):
    print(f"########## github ##########")

    ds = get_filtered_github_dataset()
    splited_ds = ds.train_test_split(test_size=5000, shuffle=True, seed=rnd_seed)
    valid_splited_ds = splited_ds["test"].train_test_split(test_size=2000, shuffle=False)

    for ds_name, ds_size in [
        ("10k", 10_000), ("100k", 100_000), ("1m", 1_000_000),
    ]:
        save_dir = os.path.join(base_dir, "github-"+ds_name)
        data = {
            "train": splited_ds["train"].select(range(ds_size)),
            "valid": valid_splited_ds["train"],
            "test": valid_splited_ds["test"],
        }
        save_hf_dataset_in_jsonl(data, save_dir)

    print(f"[GitHub] Load {len(ds)} python files")
    print("#############################")


def process_flytech(base_dir):
    print("########## flytech/python-code-25k ##########")

    # Function to tokenize and count tokens for batches
    def clean_flytech(batch):
        cleaned_output = [
            out.removeprefix("```python").removesuffix("```").strip()
            for out in batch["text"]
        ]
        return {"text": cleaned_output}

    save_dir = os.path.join(base_dir, "flytech")

    # ds = load_dataset('flytech/python-codes-25k', split='train')
    ds = load_dataset("json", data_files="https://huggingface.co/datasets/flytech/python-codes-25k/resolve/main/python-codes-25k.jsonl")["train"]
    ds = ds.remove_columns("text")
    ds = ds.rename_column("output", "text")

    ds = ds.map(count_tokens, batched=True, batch_size=64)
    ds = ds.map(clean_flytech, batched=True, batch_size=64)

    splited_ds = ds.train_test_split(test_size=5000, shuffle=True, seed=rnd_seed)
    valid_splited_ds = splited_ds["test"].train_test_split(test_size=2000, shuffle=False)

    train_ds = splited_ds["train"]
    valid_ds = valid_splited_ds["train"]
    test_ds = valid_splited_ds["test"]
    data = {
        "train": train_ds,
        "valid": valid_ds,
        "test": test_ds,
    }
    save_hf_dataset_in_jsonl(data, save_dir)

    print(f"[Flytech] Load {len(ds)} python files, saving {len(train_ds)}/{len(valid_ds)}/{len(test_ds)}...")
    print("#############################")


def run():
    base_dir = "./data/code"

    process_flytech(base_dir)


if __name__ == "__main__":
    run()

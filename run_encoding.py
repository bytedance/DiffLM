# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import argparse

import torch
import numpy as np

from pathlib import Path
from tqdm import trange
from models.auto_encoder import VAELanguageModel, get_vae_data_args


def load_data_samples(dataset_dir):
    def load_json_lines(file_name):
        res = list()
        with open(os.path.join(dataset_dir, file_name), "r") as rf:
            for idx, line in enumerate(rf):
                lj = json.loads(line)
                res.append({
                    "origin_id": idx,
                    "text": lj["text"],
                })
        return res

    return dict(
        train = load_json_lines("train.jsonl"),
        valid = load_json_lines("valid.jsonl"),
        test = load_json_lines("test.jsonl"),
    )


@torch.inference_mode()
def do_latent_encoding(samples, vae_model, output_file, device=torch.device("cuda")):
    batch_size = 64
    encoded_mus = list()
    for i in trange(0, len(samples), batch_size):
        batch_samples = samples[i: i+batch_size]
        batch_texts = [s["text"] for s in batch_samples]
        batch_z, batch_mu, batch_logvar = vae_model.encode(batch_texts, device=device)
        encoded_mus.append(batch_mu.detach().cpu())
    encoded_mus = torch.cat(encoded_mus, dim=0)
    encoded_mus = encoded_mus.numpy()
    print(encoded_mus.shape)
    np.save(output_file, encoded_mus)


def run(args, device=torch.device("cuda")):
    # Load Model
    vae_model = VAELanguageModel.load_from_saved_model(args.model_path, device=device)

    # Load dataset
    encoding_dir = os.path.join(args.model_path, "encodings")
    os.makedirs(encoding_dir, exist_ok=False)

    data_args = get_vae_data_args(args.model_path)
    data_dir = Path(data_args["data_path"]).parent
    data_samples = load_data_samples(data_dir)
    for split, samples in data_samples.items():
        _z_file = os.path.join(encoding_dir, f"{split}.npy")
        do_latent_encoding(samples, vae_model, _z_file, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for argparse")
    parser.add_argument("--model_path", type=str, required=True, help="Input file path")
    # parser.add_argument("--dataset", type=str, required=True, help="Dataset directory path")
    args = parser.parse_args()

    run(args)

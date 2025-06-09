# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import time
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import transformers
from tqdm import tqdm
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from utils.common import setup_logger, set_seed
from models.diffusion import LatentDiffuser, DIFFUSION_MODEL_DIR


logger = logging.getLogger()
setup_logger(logger)


@dataclass_json
@dataclass
class TrainingArguments:
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"}) # no need for now
    gpu: int = field(default=0, metadata={"help": "The GPU index to use for training, set -1 to use CPU"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

    vae_model_path: str = field(
        default=None, metadata={"help": "Path to the vae model."}
    )
    do_eval: bool = field(
        default=False, metadata={"help": "Enable to do evaluating every epoch."}
    )

    num_epochs: int = field(
        default=10000, metadata={"help": "Number of training epochs"}
    )
    batch_size: int = field(
        default=512, metadata={"help": "Batch size"}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Inital learning rate, will descrese with ReduceLROnPlateau strategy"}
    )
    dim_noise: int = field(
        default=4096, metadata={"help": "The hidden dimision used by diffusion de-nosiser"},
    )
    denoising_impl: str = field(
        default="attn", metadata={"help": "The denoising network used by latent diffusion model, could be attn or mlp"},
    )
    denoising_layers: int = field(
        default=10, metadata={"help": "The denoising network used by latent diffusion model, could be attn or mlp"},
    )


def make_latent_train_dataset(args):
    train_file = os.path.join(args.vae_model_path, "encodings", "train.npy")
    train_z = torch.tensor(np.load(train_file), dtype=torch.float)
    latent_size = train_z.size(1)

    mean = train_z.mean(dim=0)
    # std = train_z.std(dim=0)
    train_data = (train_z - mean) / 2

    train_loader = DataLoader(
        train_data,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
    )

    if args.do_eval:
        eval_file = os.path.join(args.vae_model_path, "encodings", "valid.npy")
        eval_z = torch.tensor(np.load(eval_file), dtype=torch.float)
        assert eval_z.size(1) == latent_size, "Please ensuring the VAE latent dimision is equal for training and evaluation data."
        eval_data = (eval_z - mean) / 2
        eval_loader = DataLoader(
            eval_data,
            batch_size = args.batch_size * 4,
            shuffle=False,
        )
    else:
        eval_loader = None

    return dict(
        train_loader = train_loader,
        eval_loader = eval_loader,
        latent_size = latent_size,
    )


def prepare_diffusion_model(args):
    diff_model = LatentDiffuser(
        args.latent_size,
        args.dim_noise,
        denoising_impl=args.denoising_impl,
        denoising_layers=args.denoising_layers,
    )
    diff_model = diff_model.to(args.device)

    num_params = sum(p.numel() for p in diff_model.denoise_fn.parameters())
    logger.info(f"The number of parameters {num_params}")

    optimizer = torch.optim.AdamW(diff_model.parameters(), lr=args.learning_rate, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=20, verbose=True)

    return diff_model, optimizer, scheduler


def run(args):
    datasets = make_latent_train_dataset(args)
    args.latent_size = datasets["latent_size"]
    diff_model, optimizer, scheduler = prepare_diffusion_model(args)

    config_file = os.path.join(args.output_dir, "config.json")
    state_file = os.path.join(args.output_dir, "training_state.json")

    with open(config_file, "w") as wf:
        save_config = args.to_dict()
        save_config["latent_size"] = datasets["latent_size"]
        json.dump(save_config, wf, ensure_ascii=False, indent=4)

    patience = 0
    best_loss = float("inf")
    start_time = time.time()
    state_json = {
        "best_epoch": 0,
        "best_loss": None,
        "losses": list(),
        "eval_losses": list(),
    }
    for epoch in range(1, args.num_epochs+1):
        diff_model.train()

        pbar = tqdm(datasets["train_loader"], total=len(datasets["train_loader"]))
        pbar.set_description(f"Epoch {epoch} / {args.num_epochs}")

        len_input = 0
        train_loss = 0.0
        for batch in pbar:
            inputs = torch.tensor(batch, dtype=torch.float, device=args.device)
            loss = diff_model(inputs)
            loss = loss.mean()

            train_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        train_loss = train_loss / len_input

        if args.do_eval:
            eval_loss = 0.0
            eval_len = 0
            with torch.no_grad():
                eval_pbar = tqdm(datasets["eval_loader"], total=len(datasets["eval_loader"]), desc="Evaluating")
                for eval_batch in eval_pbar:
                    inputs = torch.tensor(eval_batch, dtype=torch.float, device=args.device)
                    loss = diff_model(inputs)
                    loss = loss.mean()
                    eval_loss += loss.item() * len(inputs)
                    eval_len += len(inputs)
                eval_loss = eval_loss / eval_len
            logger.info("Evaluation: {}".format({"loss": eval_loss}))
            state_json["eval_losses"].append(eval_loss)
            curr_loss = eval_loss
        else:
            curr_loss = train_loss

        scheduler.step(train_loss)
        state_json["losses"].append(train_loss)
        logger.info({"loss": train_loss, "learning_rate": scheduler.get_last_lr(), "epoch": epoch})

        if curr_loss < best_loss: # evaluation loss if do_eval, else train loss
            patience = 0
            best_loss = curr_loss
            state_json["best_epoch"] = epoch
            state_json["best_loss"] = best_loss
            torch.save(diff_model.state_dict(), f"{args.output_dir}/model_best.pt")
        else:
            patience += 1
            if patience == 500:
                logger.info("Early stopping")
                break

        if epoch % 100 == 0:
            torch.save(diff_model.state_dict(), f"{args.output_dir}/model_epoch_{epoch}.pt")

        with open(state_file, "w") as wf:
            json.dump(state_json, wf, ensure_ascii=False, indent=4)

    end_time = time.time()
    logger.info(f"Training time: {end_time - start_time} s")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed, 1)
    args.output_dir = os.path.join(
        args.vae_model_path,
        f"{DIFFUSION_MODEL_DIR}_{args.denoising_impl}{args.denoising_layers}",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")

    run(args)

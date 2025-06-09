#!/usr/bin/env python
# Copyright (c) 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018 NVIDIA CORPORATION.
# Copyright (c) 2020 Microsoft Research.
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/ChunyuanLI/Optimus/blob/master/code/examples/big_ae/run_lm_ae_pretraining.py.
#
# This modified file is released under the same license.

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import transformers

from models.trainer import CyclicalBetaTrainer
from models.auto_encoder import (
    DATA_CONFIG_NAME,
    MODEL_CONFIG_NAME,
    TRAINER_CONFIG_NAME,
    prepare_vae_model,
)
from utils.common import setup_logger, is_main_process
from utils.dataset import SupervisedLMDataset, LazySupervisedLMDataset


logger = logging.getLogger(__name__)
setup_logger(logger)

local_rank = None


@dataclass_json
@dataclass
class ModelArguments:
    encoder_name_or_path: Optional[str] = field(
        default="bert-base-cased", metadata={"help": "The encoder model checkpoint for weights initialization."}
    )
    decoder_name_or_path: Optional[str] = field(
        default="gpt2", metadata={"help": "The decoder model checkpoint for weights initialization."}
    )
    encoder_max_length: int = field(
        default=512, metadata={"help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    decoder_max_length: int = field(
        default=512, metadata={"help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    freeze_decoder: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze decoder model paramenters."},
    )
    vae_latent_size: Optional[int] = field(
        default=32, metadata={"help": "Latent space dimension."}
    )
    vae_adapter_size: Optional[int] = field(
        default=8, metadata={"help": "Token numbers of adapter for both kv_memory and soft prompt."}
    )
    vae_latent_method: str = field(
        default="soft_prompt",
        metadata={
            "help": "soft_prompt: VAE latent vector as a soft promt token embeddings (adding before the bos token); kv_memory: VAE latent vector as a past_key_values as previous memory; input_embed: VAE latent vector as a extra token embedding.",
            "choices": ["soft_prompt", "kv_memory", "input_embed", "prefix_soft_prompt"],
        },
    )
    vae_prefix_text: Optional[str] = field(
        default=None, metadata={"help": "Init text for M+N soft prompts -> vae_latent_method==prefix_soft_prompt, set None to random init the M embeddings"}
    )
    threshold_kl: float = field(
        default=None, metadata={"help": "The thresholding objective causes learning to give up driving down KL for dimensions of z that are already beneath the target compression rate."},
    )
    deterministic_connect: bool = field(
        default=False, metadata={"help": "Use deterministic inference to generate latent codes, i.e., standard auto-encoders."},
    )
    length_weighted_loss: bool = field(
        default=False, metadata={"help": "Whether to use length weighted loss."},
    )
    left_padding: bool = field(
        default=False, metadata={"help": "Whether to use left padding for decoder. NOTE: it might have some issues with soft prompt and kv memory, see https://github.com/huggingface/peft/issues/1093 for detailed comparison."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "eager: manual implementation of the attention; sdpa(torch>=2.1.1): attention using torch.nn.functional.scaled_dot_product_attention; or flash_attention_2: attention using Dao-AILab/flash-attention",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )


@dataclass_json
@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = field(
        default=False, metadata={"help": "Whether to lazy load dataset."}
    )
    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss."},
    )
    shuffle_column: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."},
    )


@dataclass_json
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    save_best_model_at_end: bool = field(
        default=False, metadata={"help": "Use cyclical target beta."},
    )
    do_cyclical: bool = field(
        default=False, metadata={"help": "Use cyclical target beta."},
    )
    num_cycle: int = field(
        default=10, metadata={"help": "The number of cyclical beta iterations."},
    )
    beta: float = field(
        default=1.0, metadata={"help": "The weighting hyper-parameter of the KL term in VAE."},
    )
    ratio_zero: float = field(
        default=0.5, metadata={"help": "Learning schedule, the percentage for the pure auto-encoding stage."},
    )
    ratio_increase: float = field(
        default=0.25, metadata={"help": "Learning schedule, the percentage for the annealing stage."},
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Force run some model when transformers version is lower than its requirments"},
    )


def extract_losses(eval_preds):
    loss_kl, loss_rec = eval_preds.predictions

    return {
        "loss_kl": loss_kl.mean(),
        "loss_rec": loss_rec.mean(),
    }


def make_supervised_mlm_data_module(
    encoder_tokenizer: transformers.PreTrainedTokenizer,
    decoder_tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    if is_main_process(local_rank):
        logger.info(f"[Data] Loading data from {data_args}.")

    dataset_cls = (
        LazySupervisedLMDataset if data_args.lazy_preprocess else SupervisedLMDataset
    )

    def load_json_dataset(file_path):
        with open(file_path, "r") as rf:
            if file_path.endswith("jsonl"):
                res_json = [json.loads(line) for line in rf if line.strip() != ""]
            elif file_path.endswith("json"):
                res_json = json.load(rf)
            else:
                raise NotImplementedError(f"not supported file type -> {file_path}")
        assert isinstance(res_json, list)
        return res_json

    train_json = load_json_dataset(data_args.data_path)
    train_dataset = dataset_cls(
        train_json,
        encoder_tokenizer,
        decoder_tokenizer,
        use_mlm=data_args.mlm,
        mlm_probability=data_args.mlm_probability,
    )

    if data_args.eval_data_path:
        eval_json = load_json_dataset(data_args.eval_data_path)
        eval_dataset = dataset_cls(
            eval_json,
            encoder_tokenizer,
            decoder_tokenizer,
            shuffle_column=data_args.shuffle_column,
            use_mlm=data_args.mlm,
            mlm_probability=data_args.mlm_probability,
        )
    else:
        eval_dataset = None

    print(train_dataset.source_texts[:5])
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train(
    trainer_cls: transformers.Trainer = CyclicalBetaTrainer,
    train_arg_cls: transformers.TrainingArguments = TrainingArguments
):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, train_arg_cls)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # reserve soft prompt tokens
    if model_args.vae_latent_method == "soft_prompt":
        model_args.decoder_max_length -= model_args.vae_adapter_size
    if model_args.vae_latent_method == "prefix_soft_prompt":
        model_args.decoder_max_length -= model_args.vae_adapter_size * 2

    vae_model, encoder_tokenizer, decoder_tokenizer = prepare_vae_model(model_args, training_args, local_rank)
    data_module = make_supervised_mlm_data_module(encoder_tokenizer, decoder_tokenizer, data_args)

    # Start training
    trainer = trainer_cls(
        args=training_args,
        model=vae_model,
        tokenizer=encoder_tokenizer,
        compute_metrics=extract_losses,
        **data_module,
    )

    # save model / data / training arguments
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        for fname, _args in [
            (MODEL_CONFIG_NAME, model_args),
            (DATA_CONFIG_NAME, data_args),
            (TRAINER_CONFIG_NAME, training_args),
        ]:
            with open(os.path.join(training_args.output_dir, fname), "w") as wf:
                json.dump(_args.to_dict(), wf, indent=4, ensure_ascii=False)

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Move the best model to output_dir
    if training_args.save_best_model_at_end:
        if trainer.is_world_process_zero():
            best_ckpt_path = Path(trainer.state.best_model_checkpoint)
            out_ckpt_path = Path(training_args.output_dir)
            for each_file in best_ckpt_path.glob("*"): # grabs all files
                each_file.rename(out_ckpt_path.joinpath(each_file.name)) # moves to output folder.
    else:
        trainer.save_state()
        trainer.save_model()


if __name__ == "__main__":
    train()

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

import logging
from typing import Optional
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

import transformers

from models.trainer import BacktrackingBetaTrainer
from utils.common import setup_logger
from run_vae_pretraining import train


logger = logging.getLogger(__name__)
setup_logger(logger)

local_rank = None


@dataclass_json
@dataclass
class BacktrackingTrainingArguments(transformers.TrainingArguments):
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    save_best_model_at_end: bool = field(
        default=False, metadata={"help": "Use cyclical target beta."},
    )
    max_beta: float = field(
        default=1e-2, metadata={"help": "The maximal weighting hyper-parameter of the KL term in VAE."},
    )
    min_beta: float = field(
        default=1e-5, metadata={"help": "The minimal weighting hyper-parameter of the KL term in VAE."},
    )
    lambda_beta_factor: float = field(
        default=0.7, metadata={"help": "The backtracking factor for adaptively dynamic beta changing, used when reconstruction loss fails to descrease"}
    )
    backtracking_interval: int = field(
        default=5, metadata={"help": "The interval for backtracking (number of evaluation)"}
    )
    early_stopping_interval: int = field(
        default=50, metadata={"help": "The interval for early stopping (number of evaluation)"}
    )


if __name__ == "__main__":
    train(
        trainer_cls=BacktrackingBetaTrainer,
        train_arg_cls=BacktrackingTrainingArguments,
    )

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import logging
import torch
import torch.nn as nn

from typing import Any, Optional, Union, List, Dict
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils.common import frange_cycle_zero_linear
from models.auto_encoder import MODEL_CONFIG_NAME


logger = logging.getLogger()


class SaveModelConfigCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return

        save_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        if not os.path.exists(save_dir):
            save_dir = args.output_dir
        save_file = os.path.join(save_dir, MODEL_CONFIG_NAME)

        # save model config
        if hasattr(model, "config"):
            with open(save_file, "w") as wf:
                json.dump(
                    model.config.to_dict(), wf,
                    indent=4, ensure_ascii=False,
                )


class CyclicalBetaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_list = None

    def get_cyclical_beta(self, ):
        if self.beta_list is None and self.args.do_cyclical:
            max_steps = self.state.max_steps
            # assert self.state.global_step == 0
            self.beta_list = frange_cycle_zero_linear(
                max_steps,
                n_cycle=self.args.num_cycle,
                start=0.0, stop=self.args.beta,
                ratio_zero=self.args.ratio_zero,
                ratio_increase=self.args.ratio_increase,
            )

        global_step = self.state.global_step
        cur_beta = self.beta_list[global_step]
        return cur_beta

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.args.do_cyclical:
            cur_beta = self.get_cyclical_beta()
        else:
            cur_beta = self.args.beta

        inputs.update({
            "beta": cur_beta,
        })

        return super().training_step(model, inputs)


class BacktrackingBetaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = self.args.max_beta
        self.min_beta = self.args.min_beta
        self.lambda_beta_factor = self.args.lambda_beta_factor
        self.backtracking_interval = self.args.backtracking_interval
        self.early_stopping_interval = self.args.early_stopping_interval
        self.patience = 0
        self.best_rec_loss = float("inf")

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        inputs.update({
            "beta": self.beta,
        })

        return super().training_step(model, inputs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # NOTE, beta = 0.0 by default, so eval_loss equals to eval_loss_rec
        cur_loss = metrics[f"{metric_key_prefix}_loss"]
        if cur_loss < self.best_rec_loss:
            self.patience = 0
            self.best_rec_loss = cur_loss
        else: # reconstruct loss fails to descrease
            self.patience += 1
            if self.patience >= self.early_stopping_interval: # early stopping
                self.control.should_training_stop = True
            if self.patience % self.backtracking_interval == 0 and self.beta > self.min_beta:
                prev_beta = self.beta
                self.beta *= self.lambda_beta_factor # lambda < 1.0
                logger.info(f"Backtracking current beta from {prev_beta} to {self.beta}.")

        return metrics

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import torch
import numpy as np
import pandas as pd
from evals.metric import TabularMetric


class TabularCopy(TabularMetric):
    name = "tabular_copy"

    def get_score(self):
        train_df = self.train_df
        syn_df = self.syn_df

        merged_df = pd.merge(train_df, syn_df)

        return {
            "overlapped_ratio": len(merged_df) / len(train_df),
        }


if __name__ == "__main__":
    tabular_dir = "./data/tabular/shoppers"
    syn_file = "./output/generations/bert-mistral-iv3-shoppers-cycle-iter10-ckpt7600-temp0.5.jsonl"

    evl = TabularCopy(tabular_dir, syn_file)
    print(evl.get_score())


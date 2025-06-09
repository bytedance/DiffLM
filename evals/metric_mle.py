# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import sys
import logging
import pandas as pd

from evals.metric import TabularMetric
from evals.mle.mle import get_evaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger = logging.getLogger()


class TabularMLE(TabularMetric):
    name = "tabular_mle"

    def __init__(self, tabular_dir, syn_file) -> None:
        super().__init__(tabular_dir, syn_file)

    def get_score(self, rerun_times=100):
        task_type = self.info["task_type"]
        test_data = self.test_df.to_numpy()
        syn_data = self.syn_df.to_numpy()

        if task_type == "regression":
            method_name = "XGBRegressor"
            metric_names = {
                "r2": True,
                "explained_variance": True,
                "MAE": False,
                "RMSE": False,
            }
        else:
            method_name = "XGBClassifier"
            metric_names = {
                "binary_f1": True,
                "roc_auc": True,
                "weighted_f1": True,
                "accuracy": True,
            }

        all_results = list()
        while len(all_results) < rerun_times:
            evaluator = get_evaluator(task_type)
            res_results = evaluator(syn_data, test_data, self.info)

            one_best_results = {
                n: -10000 if gb else 10000
                for n, gb in metric_names.items()
            }
            for scores in res_results:
                assert len(scores) == 1 and scores[0]["name"] == method_name
                method = scores[0]
                for metric, greater_is_better in metric_names.items():
                    if greater_is_better is True and method[metric] > one_best_results[metric]:
                        one_best_results[metric] = method[metric]
                    elif greater_is_better is False and method[metric] < one_best_results[metric]:
                        one_best_results[metric] = method[metric]
            # convert classification results to percentage
            if task_type != "regression":
                for metric, val in one_best_results.items():
                    one_best_results[metric] = val * 100
            all_results.append(one_best_results)

        res_df = pd.DataFrame.from_records(all_results)
        if task_type == "regression":
            best_run = res_df.loc[res_df["RMSE"].idxmin()].to_dict()
            worst_run = res_df.loc[res_df["RMSE"].idxmax()].to_dict()
        else:
            best_run = res_df.loc[res_df["roc_auc"].idxmax()].to_dict()
            worst_run = res_df.loc[res_df["roc_auc"].idxmin()].to_dict()
        return {
            "MLE_Best": best_run,
            "MLE_Worst": worst_run,
            "MLE_Run_Times": rerun_times,
        }

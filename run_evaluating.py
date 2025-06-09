# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import argparse

from pathlib import Path
from evals import *


def get_data_dir(data_path):
    if "adult" in data_path:
        return "./data/tabular/adult"
    elif "beijing" in data_path:
        return "./data/tabular/beijing"
    elif "default" in data_path:
        return "./data/tabular/default"
    elif "magic" in data_path:
        return "./data/tabular/magic"
    elif "news" in data_path:
        return "./data/tabular/news"
    elif "shoppers" in data_path:
        return "./data/tabular/shoppers"


def run_evaluation(args):
    if not args.do_eval:
        return

    data_dir = get_data_dir(args.output_file)

    eval_metrics = dict()
    eval_metrics.update(TabularDensity(data_dir, args.output_file).get_score())
    eval_metrics.update(TabularMLE(data_dir, args.output_file).get_score(rerun_times=100))

    with open(args.result_file, "w") as wf:
        json.dump(eval_metrics, wf, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for argparse")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluations.")
    parser.add_argument("--output_file", type=str, required=True, help="Evaluation result file path")
    parser.add_argument("--result_dir", type=str, default="./output/evaluations", help="Evaluation result file path")
    args = parser.parse_args()

    args.result_file = os.path.join(args.result_dir, Path(args.output_file).stem+".json")

    run_evaluation(args)

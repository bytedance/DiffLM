# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import argparse
import numpy as np
import pandas as pd

# Metrics
from sdmetrics.reports.single_table import QualityReport
from evals.metric import TabularMetric


class TabularDensity(TabularMetric):
    name = "tabular_density"

    def __init__(self, tabular_dir, syn_file) -> None:
        super().__init__(tabular_dir, syn_file)

    def get_score(self):
        info = self.info
        train_data = self.train_df
        syn_data = self.syn_df

        train_data.columns = range(len(train_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        metadata = info["metadata"]
        metadata["columns"] = {int(key): value for key, value in metadata["columns"].items()}

        new_real_data, new_syn_data, metadata = self.reorder(train_data, syn_data, info)

        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata)
        quality = qual_report.get_properties()

        # diag_report = DiagnosticReport()
        # diag_report.generate(new_real_data, new_syn_data, metadata)
        # diag = diag_report.get_properties()

        # Quality = (Shape_error + Shape_error) / 2

        # shapes = qual_report.get_details(property_name="Column Shapes")
        # trends = qual_report.get_details(property_name="Column Pair Trends")
        # coverages = diag_report.get_details("Coverage")

        # shapes.to_csv(f"{save_dir}/shape.csv")
        # trends.to_csv(f"{save_dir}/trend.csv")
        # coverages.to_csv(f"{save_dir}/coverage.csv")

        # Calculate the error rate and convert them to percentage
        shape_error = (1 - quality["Score"][0]) * 100
        trend_error = (1 - quality["Score"][1]) * 100

        return {
            "Column_Shapes_Error": shape_error,
            "Column_Pair_Trends_Error": trend_error,
        }


if __name__ == "__main__":
    tabular_dir = "./data/tabular/shoppers"
    syn_file = "./output/generations/random_prior/t5-3b-mistral-v3-shoppers-L1024-A32-temp0.5.jsonl"

    evl = TabularDensity(tabular_dir, syn_file)
    print(evl.get_score())

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval


class Metric(object):
    name = "metric"
    direction = "maximize"

    def get_score(self):
        pass


class JsonFormat(Metric):
    name = "JsonFormat"

    def __init__(self, syn_samples, required_keys=[]):
        super().__init__()
        self.syn_samples = syn_samples
        self.required_keys = required_keys
        self.json_samples = None

    def __load_json(self, json_str):
        try:
            res = json.loads(json_str)
        except Exception as e:
            try:
                res = literal_eval(json_str)
            except:
                res = None

        if not isinstance(res, dict):
            return None

        for k in self.required_keys:
            if k not in res.keys():
                return None

        return res

    def get_json_list(self, ):
        if self.json_samples is not None:
            return self.json_samples

        res = list()
        for sample in self.syn_samples:
            res.append(self.__load_json(sample))
        self.json_samples = res

        return res

    def get_score(self):
        all_jsons = self.get_json_list()
        success = sum(j is not None for j in all_jsons)
        return dict(
            json_format_success_rate = success / len(all_jsons)
        )


class TabularMetric(Metric):
    name = "tabular_metric"

    def __init__(self, tabular_dir, syn_file) -> None:
        super().__init__()
        self.tabular_dir = tabular_dir
        self.syn_file = syn_file
        self.load_tabular_info()

    @property
    def dataset_name(self):
        if "/adult" in self.tabular_dir:
            return "adult"
        elif "/beijing"in self.tabular_dir:
            return "beijing"
        elif "/default"in self.tabular_dir:
            return "default"
        elif "/magic"in self.tabular_dir:
            return "magic"
        elif "/news"in self.tabular_dir:
            return "news"
        elif "/shoppers"in self.tabular_dir:
            return "shoppers"
        else:
            raise NotImplementedError(f"not supported tabular dataset -> {self.tabular_dir}")

    def load_tabular_info(self):
        train_path = os.path.join(self.tabular_dir, "train.csv")
        test_path = os.path.join(self.tabular_dir, "test.csv")
        info_path = os.path.join(self.tabular_dir, "info.json")

        with open(info_path, "r") as f:
            self.info = json.load(f)
        self.build_metadata()

        # read synthetic text jsonl
        syn_suffix = Path(self.syn_file).suffix
        if syn_suffix == ".jsonl":
            syn_samples = list()
            with open(self.syn_file, "r") as rf:
                for line in rf:
                    json_str = json.loads(line)["text"]
                    syn_samples.append(json.loads(json_str))
            self.syn_df = pd.DataFrame.from_records(syn_samples)
        elif syn_suffix == ".csv":
            self.syn_df = pd.read_csv(self.syn_file)
        else:
            raise NotImplementedError(f"not supported synthetic file type -> {syn_suffix}")

        # reorder dataframe
        self.train_df = pd.read_csv(train_path)[self.info["column_names"]]
        self.test_df = pd.read_csv(test_path)[self.info["column_names"]]
        self.syn_df = self.syn_df[self.info["column_names"]]

        self.uniform_data()

    def uniform_data(self, check_number=False):
        origin_cnt = len(self.syn_df)

        # convert all category to string, and all numbers to float
        column_names = self.info["column_names"]
        num_col = [column_names[i] for i in self.info["num_col_idx"]]
        cat_col = [column_names[i] for i in self.info["cat_col_idx"]]
        tgt_col = [column_names[i] for i in self.info["target_col_idx"]]
        if self.info["task_type"] == "regression":
            num_col += tgt_col
        else:
            cat_col += tgt_col

        for df in [self.train_df, self.test_df, self.syn_df]:
            for col in num_col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            for col in cat_col:
                df[col] = df[col].astype(str)

        # filtering invalid rows
        for col in cat_col:
            col_vals = self.info["column_info"][col]["candidates"]
            self.syn_df = self.syn_df[self.syn_df[col].isin(col_vals)]

        if check_number:
            for col in num_col:
                col_min = self.info["column_info"][col]["min"]
                col_max = self.info["column_info"][col]["max"]
                self.syn_df = self.syn_df[self.syn_df[col].between(col_min, col_max)]

        # filtering `beijing` labels is None
        if self.dataset_name == "beijing":
            # self.syn_df = self.syn_df[~self.syn_df["pm2.5"].isna() & ~np.isinf(self.syn_df["pm2.5"])]
            self.syn_df = self.syn_df[~self.syn_df["pm2.5"].isna() & self.syn_df["pm2.5"].notnull()]
            self.train_df = self.train_df[~self.train_df["pm2.5"].isna() & self.train_df["pm2.5"].notnull()]
            self.test_df = self.test_df[~self.test_df["pm2.5"].isna() & self.test_df["pm2.5"].notnull()]

        print("[{} - {}] filtering from {} -> {}".format(self.tabular_dir, self.syn_file, origin_cnt, len(self.syn_df)))

        # pattern
        # if self.dataset_name == "adult":
        #     self.syn_df = self.syn_df[self.syn_df["income"].isin([">50K", "<=50K"])]
        # elif self.dataset_name == "beijing":
        #     # self.syn_df = self.syn_df[~self.syn_df["pm2.5"].isna() & ~np.isinf(self.syn_df["pm2.5"])]
        #     self.syn_df = self.syn_df[~self.syn_df["pm2.5"].isna() & self.syn_df["pm2.5"].notnull()]
        #     self.train_df = self.train_df[~self.train_df["pm2.5"].isna() & self.train_df["pm2.5"].notnull()]
        #     self.test_df = self.test_df[~self.test_df["pm2.5"].isna() & self.test_df["pm2.5"].notnull()]
        # elif self.dataset_name == "default":
        #     self.syn_df.loc[self.syn_df["default payment next month"] >= 2, "default payment next month"] = 1
        #     self.syn_df = self.syn_df[self.syn_df["default payment next month"].isin([1, 0])]
        # elif self.dataset_name == "magic":
        #     self.syn_df = self.syn_df[self.syn_df["class"].isin(["g", "h"])]

    def build_metadata(self):
        info = self.info

        num_col_idx = info["num_col_idx"].copy()
        cat_col_idx = info["cat_col_idx"].copy()
        target_col_idx = info["target_col_idx"].copy()

        task_type = info["task_type"]
        if task_type == "regression":
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        metadata_col = dict()
        for col_idx in num_col_idx:
            metadata_col[col_idx] = {
                "sdtype": "numerical",
                "computer_representation": "Float"
            }
        for col_idx in cat_col_idx:
            metadata_col[col_idx] = {
                "sdtype": "categorical"
            }

        self.info["metadata"] = {
            "columns": metadata_col,
        }

    def reorder(self, real_data, syn_data, info):
        num_col_idx = info["num_col_idx"].copy()
        cat_col_idx = info["cat_col_idx"].copy()
        target_col_idx = info["target_col_idx"].copy()

        task_type = info["task_type"]
        if task_type == "regression":
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        real_num_data = real_data[num_col_idx]
        real_cat_data = real_data[cat_col_idx]

        new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
        new_real_data.columns = range(len(new_real_data.columns))

        syn_num_data = syn_data[num_col_idx]
        syn_cat_data = syn_data[cat_col_idx]

        new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
        new_syn_data.columns = range(len(new_syn_data.columns))

        metadata = info["metadata"]
        columns = metadata["columns"]
        metadata["columns"] = {}

        for i in range(len(new_real_data.columns)):
            if i < len(num_col_idx):
                metadata["columns"][i] = columns[num_col_idx[i]]
            else:
                metadata["columns"][i] = columns[cat_col_idx[i-len(num_col_idx)]]

        return new_real_data, new_syn_data, metadata

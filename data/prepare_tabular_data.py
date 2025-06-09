# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


rnd_seed = 42


def save_tabular_jsonl(data, output_dir, target_col=None):
    for split, df in data.items():
        save_file = os.path.join(output_dir, f"{split}.jsonl")
        save_csv_file = os.path.join(output_dir, f"{split}.csv")

        df_no_id = df.drop(columns=["id"], inplace=False)
        df_no_id.to_csv(save_csv_file, index=False)

        with open(save_file, "w") as wf:
            for _, row in df.iterrows():
                row_dict = {k: v for k, v in row.to_dict().items() if k != "id"}
                res = {
                    "id": int(row["id"]),
                    "text": json.dumps(row_dict, ensure_ascii=False),
                }
                if target_col is not None:
                    res["target_col"] = target_col
                    res["target"] = row_dict[target_col]
                wf.write(json.dumps(res, ensure_ascii=False) + "\n")
        print(f"{len(df)} samples saved to {save_file}")


def save_dataset_info(data_df, task, task_type, names, num_col, cat_col, tgt_col, output_dir):
    assert len(set(num_col + cat_col + tgt_col)) == len(names)

    num_col_idx = sorted([names.index(x) for x in num_col])
    cat_col_idx = sorted([names.index(x) for x in cat_col])
    tgt_col_idx = sorted([names.index(x) for x in tgt_col])

    col_info = dict()
    for col in names:
        if col in num_col or (col == tgt_col and task_type == "regression"):
            col_info[col] = {
                "type": "float",
                "min": float(data_df[col].astype(float).min()),
                "max": float(data_df[col].astype(float).max()),
            }
        else:
            col_info[col] = {
                "type": "category",
                "candidates": [str(x) for x in data_df[col].unique()],
            }

    data_info = {
        "name": task,
        "task_type": task_type,
        "column_names": names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": tgt_col_idx,
        "column_info": col_info,
    }
    with open(os.path.join(output_dir, "info.json"), "w") as wf:
        json.dump(data_info, wf, indent=4, ensure_ascii=False)


def process_adult(base_dir):
    print("########## adult ##########")
    data_dir = os.path.join(base_dir, "adult")
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

    data_df = pd.read_csv(os.path.join(data_dir, "adult.data"), names=names, skipinitialspace=True)
    test_df = pd.read_csv(os.path.join(data_dir, "adult.test"), names=names, skipinitialspace=True)
    data_df["id"] = list(range(len(data_df))) # keep origin index
    test_df["id"] = list(range(len(data_df), len(data_df)+len(test_df)))
    test_df["income"] = test_df["income"].str.rstrip(".")
    train_df, valid_df = train_test_split(data_df, test_size=0.1, random_state=rnd_seed)

    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="income")

    # info
    full_df = pd.concat([data_df, test_df])
    num_col = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    cat_col = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    tgt_col = ["income", ]
    save_dataset_info(full_df, "adult", "binclass", names, num_col, cat_col, tgt_col, data_dir)

    print("###########################")


def process_beijing(base_dir):
    print("########## beijing ##########")

    data_dir = os.path.join(base_dir, "beijing")
    data_df = pd.read_csv(os.path.join(data_dir, "PRSA_data_2010.1.1-2014.12.31.csv"), header=0)
    data_df.rename(columns={"No": "id"}, inplace=True)

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="pm2.5")

    # info
    names = ["year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    num_col = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    cat_col = ["year", "month", "day", "hour", "cbwd"]
    tgt_col = ["pm2.5", ]
    save_dataset_info(data_df, "beijing", "regression", names, num_col, cat_col, tgt_col, data_dir)

    print("#############################")


def process_default(base_dir):
    print("########## default ##########")

    data_dir = os.path.join(base_dir, "default")
    data_df = pd.read_excel(os.path.join(data_dir, "default of credit card clients.xls"), skiprows=[0], dtype=int)
    data_df.rename(columns={"ID": "id"}, inplace=True)
    # data_df["id"] = data_df["id"].tolist()

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="default payment next month")

    # info
    names = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "default payment next month"]
    num_col = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    cat_col = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    tgt_col = ["default payment next month", ]
    save_dataset_info(data_df, "default", "binclass", names, num_col, cat_col, tgt_col, data_dir)

    print("#############################")


def process_magic(base_dir):
    print("########## magic ##########")

    data_dir = os.path.join(base_dir, "magic")
    names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    data_df = pd.read_csv(os.path.join(data_dir, "magic04.data"), names=names)
    data_df["id"] = list(range(len(data_df)))

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="class")

    # info
    names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    num_col = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
    cat_col = []
    tgt_col = ["class", ]
    save_dataset_info(data_df, "magic", "binclass", names, num_col, cat_col, tgt_col, data_dir)

    print("###########################")


def process_news(base_dir):
    print("########## news ##########")

    data_dir = os.path.join(base_dir, "news")
    data_df = pd.read_csv(os.path.join(data_dir, "OnlineNewsPopularity.csv"), header=0, skipinitialspace=True)
    data_df["id"] = list(range(len(data_df)))

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="shares")

    # info
    names = ["year", "month", "day", "hour", "pm2.5", "DEWP", "TEMP", "PRES", "cbwd", "Iws", "Is", "Ir"]
    num_col = []
    cat_col = []
    tgt_col = ""
    save_dataset_info(data_df, "adult", "binclass", names, num_col, cat_col, tgt_col, data_dir)

    print("#############################")


def process_shoppers(base_dir):
    print("########## shoppers ##########")

    data_dir = os.path.join(base_dir, "shoppers")
    data_df = pd.read_csv(os.path.join(data_dir, "online_shoppers_intention.csv"), header=0)
    data_df["id"] = list(range(len(data_df)))

    train_df, tmp_df = train_test_split(data_df, test_size=0.2, random_state=rnd_seed)
    valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=rnd_seed)
    data = {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }
    save_tabular_jsonl(data, data_dir, target_col="Revenue")

    # info
    names = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue"]
    num_col = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
    cat_col = ["Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"]
    tgt_col = ["Revenue", ]
    save_dataset_info(data_df, "shoppers", "binclass", names, num_col, cat_col, tgt_col, data_dir)

    print("#############################")


def run():
    base_dir = "./data/tabular"

    process_adult(base_dir)
    process_beijing(base_dir)
    process_default(base_dir)
    process_magic(base_dir)
    # process_news(base_dir)
    process_shoppers(base_dir)


if __name__ == "__main__":
    run()

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


SMALL_FONT = 18
NORMAL_FONT = 20
BIG_FONT = 28

# plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': NORMAL_FONT,
    'font.family': 'serif',
    'pdf.fonttype': 42,
})


def get_toolbench_category_mapping(cat_json):
    with open(cat_json, "r") as rf:
        cat_map = json.load(rf)

    res = dict()
    for one in cat_map:
        res[(one["tool_name"], one["api_name"])] = one["category"]
    return res


def prepare_toolbench(train_file, emb_file, cat_file):
    cat_map = get_toolbench_category_mapping(cat_file)

    samples = list()
    with open(train_file, "r") as rf:
        for line in rf:
            lj = json.loads(line)
            samples.append(json.loads(lj["text"]))

    latents = np.load(emb_file)

    # all_cats = set(cat_map.values())
    # sampled_cats = random.sample(list(all_cats), 10)
    sampled_cats = set(cat_map.values())
    # sampled_cats = [
    #     "Advertising",
    #     "Business",
    #     "Education",
    #     "Science",
    #     # "Tools",
    #     "Transportation",
    #     "Movies",
    #     "Music",
    #     "Gaming",
    #     "Food",
    #     # "Finance"
    # ]

    res_sample = list()
    res_embds = list()
    for one, vec in zip(samples, latents):
        tool = one["tool_name"]
        api = one["api_name"]
        cat = cat_map[(tool, api)]
        if cat in sampled_cats:
            one["label"] = cat
            res_sample.append(one)
            res_embds.append(vec)

    return pd.DataFrame(res_sample), np.array(res_embds)


def prepare_data(data_dir, emb_dir, split="train"):
    data_file = os.path.join(data_dir, f"{split}.jsonl")
    info_file = os.path.join(data_dir, "info.json")
    emb_file = os.path.join(emb_dir, f"{split}.npy")

    with open(info_file, "r") as rf:
        info = json.load(rf)
    target_col = info["column_names"][info["target_col_idx"][0]]

    samples = list()
    with open(data_file, "r") as rf:
        for line in rf:
            lj = json.loads(line)
            samples.append(json.loads(lj["text"]))
    sample_df = pd.DataFrame(samples)
    sample_df["label"] = sample_df[target_col]

    latents = np.load(emb_file)

    return sample_df, latents


def prepare_regression_data(reg_df, reg_emb):
    # 过滤掉标签为NaN或None的行
    valid_indices = reg_df['label'].notnull()
    reg_df = reg_df[valid_indices].reset_index(drop=True)
    reg_emb = reg_emb[valid_indices.values]

    # 将浮点数标签分成5个区间
    float_labels = reg_df['label']
    labels_binned, bin_edges = pd.qcut(float_labels, q=5, labels=False, retbins=True, duplicates='drop')
    # labels_binned, bin_edges = pd.cut(float_labels, bins=5, labels=False, retbins=True)
    reg_df['label_bin'] = labels_binned
    bin_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]

    return reg_df, reg_emb, bin_labels


def create_plot(datasets, embeddings, dataset_names, split="train"):
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    axs = axs.flatten()

    for i, (df, emb, ax) in enumerate(zip(datasets, embeddings, axs)):
        # 计算t-SNE
        tsne = TSNE(n_components=2, random_state=42)

        # 获取并处理标签
        if dataset_names[i] == "Beijing":
            df, emb, bin_labels = prepare_regression_data(df, emb)
            labels = df['label_bin']
            unique_labels = np.unique(labels)
            label_names = [bin_labels[int(l)] if not pd.isnull(l) else 'Unknown' for l in unique_labels]
        else:
            le = LabelEncoder()
            labels = le.fit_transform(df['label'])
            unique_labels = np.unique(labels)
            label_names = le.inverse_transform(unique_labels)

        # 绘制散点图
        emb_2d = tsne.fit_transform(emb)
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='viridis', s=36, rasterized=True)
        # 添加图例
        handles = [
            plt.Line2D(
                [], [],
                marker='o',
                color=scatter.cmap(scatter.norm(ul)),
                linestyle='',
                markersize=10
            ) for ul in unique_labels
        ]

        if dataset_names[i] != "Toolbench":
            ax.legend(handles, label_names, title="Label", loc='lower right', fontsize=SMALL_FONT, title_fontsize=NORMAL_FONT)

        # 设置标题和坐标轴
        ax.set_title(f"{dataset_names[i]}", fontsize=BIG_FONT)
        ax.set_xticks([])
        ax.set_yticks([])
        # 在子图下方标注数据集名称
        # ax.set_xlabel(dataset_names[i], fontsize=16)

    plt.tight_layout()

    save_file = f"./output/figures/latent_tsne_{split}.pdf"
    plt.savefig(save_file, format="pdf")


def run_plot(
    split = "valid",
):
    dataset = {
        "Adult": {
            "data_dir": "./data/tabular/adult",
            "emb_dir": "./output/models/bert-mistral-iv3-adult-prefix/encodings",
        },
        "Default": {
            "data_dir": "./data/tabular/default",
            "emb_dir": "./output/models/bert-mistral-iv3-default-prefix/encodings",
        },
        "Magic": {
            "data_dir": "./data/tabular/magic",
            "emb_dir": "./output/models/bert-mistral-iv3-magic-prefix/encodings",
        },
        "Shoppers": {
            "data_dir": "./data/tabular/shoppers",
            "emb_dir": "./output/models/bert-mistral-iv3-shoppers-freeze/encodings",
        },
        "Beijing": {
            "data_dir": "./data/tabular/beijing",
            "emb_dir": "./output/models/bert-mistral-iv3-beijing-prefix/encodings",
        },
    }

    tool_file = "./data/tool/toolbench/{}.jsonl".format(split)
    tool_emb_file = "./output/models/bert-mistral-iv3-toolbench-prefix/encodings/{}.npy".format(split)
    tool_cat_file = "./data/tool/toolbench/category.json"

    datasets = list()
    embeddings = list()
    dataset_names = list()
    for name, data_info in dataset.items():
        df, emb = prepare_data(data_info["data_dir"], data_info["emb_dir"], split=split)
        datasets.append(df)
        embeddings.append(emb)
        dataset_names.append(name)

    df, emb = prepare_toolbench(tool_file, tool_emb_file, tool_cat_file)
    datasets.append(df)
    embeddings.append(emb)
    dataset_names.append("Toolbench")

    create_plot(datasets, embeddings, dataset_names, split=split)


if __name__ == "__main__":
    fire.Fire(run_plot)

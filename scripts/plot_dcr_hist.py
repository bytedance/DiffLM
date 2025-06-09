# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import fire
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from evals.metric import TabularMetric


SMALL_FONT = 16
NORMAL_FONT = 18
BIG_FONT = 20

# plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': NORMAL_FONT,
    'font.family': 'serif',
    'pdf.fonttype': 42,
})

def remove_outliers(data):
    # 计算第一和第三四分位数
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    # 计算四分位距
    IQR = Q3 - Q1
    # 定义上下限
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    # 过滤数据
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data


def process_data(train_data, comp_data, info):
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    task_type = info["task_type"]
    if task_type == "regression":
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    train_data.columns = list(np.arange(len(train_data.columns)))
    comp_data.columns = list(np.arange(len(train_data.columns)))

    num_ranges = []
    for i in num_col_idx:
        num_ranges.append(train_data[i].max() - train_data[i].min())
    num_ranges = np.array(num_ranges)

    num_train_data = train_data[num_col_idx]
    cat_train_data = train_data[cat_col_idx]
    num_comp_data = comp_data[num_col_idx]
    cat_comp_data = comp_data[cat_col_idx]

    num_train_data_np = num_train_data.to_numpy()
    cat_train_data_np = cat_train_data.to_numpy().astype("str")
    num_comp_data_np = num_comp_data.to_numpy()
    cat_comp_data_np = cat_comp_data.to_numpy().astype("str")

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(cat_train_data_np)

    cat_train_data_oh = encoder.transform(cat_train_data_np).toarray()
    cat_comp_data_oh = encoder.transform(cat_comp_data_np).toarray()

    num_train_data_np = num_train_data_np / num_ranges
    num_comp_data_np = num_comp_data_np / num_ranges

    train_data_np = np.concatenate([num_train_data_np, cat_train_data_oh], axis=1)
    comp_data_np = np.concatenate([num_comp_data_np, cat_comp_data_oh], axis=1)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_data_th = torch.tensor(train_data_np).to(device)
    comp_data_th = torch.tensor(comp_data_np).to(device)

    res_dcrs = []
    batch_size = 100
    # batch_comp_data_np = comp_data_np[i*batch_size: (i+1) * batch_size]
    for i in range((comp_data_th.shape[0] // batch_size) + 1):
        if i != (comp_data_th.shape[0] // batch_size):
            batch_comp_data_th = comp_data_th[i*batch_size: (i+1) * batch_size]
        else:
            batch_comp_data_th = comp_data_th[i*batch_size:]

        dcr_comp = (batch_comp_data_th[:, None] - train_data_th).abs().sum(dim = 2).min(dim = 1).values
        res_dcrs.append(dcr_comp)

    return torch.cat(res_dcrs).cpu().numpy()


def prepare_dcrs(dataset, tabular_dir, syn_file):
    df_map = dict()
    difflm_tab = TabularMetric(tabular_dir, syn_file)
    df_map["Real"] = difflm_tab.test_df

    syn_dir = os.path.join("./output/tabular", dataset.lower())
    # for method in ["SMOTE", "Codi", "TabSyn", "GReaT"]:
    for method in ["Codi", "TabSyn", "GReaT"]:
        syn_file = os.path.join(syn_dir, f"{method}.csv")
        syn_tab = TabularMetric(tabular_dir, syn_file)
        df_map[method] = syn_tab.syn_df

    df_map["DiffLM"] = difflm_tab.syn_df
    train_df = difflm_tab.train_df
    info = difflm_tab.info

    dcrs_map = dict()
    for name, df in df_map.items():
        one_dcrs = process_data(train_df, df, info)
        dcrs_map[name] = remove_outliers(one_dcrs)

    return dcrs_map


def draw_dcrs(dataset, tabular_dir, syn_file, save_dir):
    dcrs_map = prepare_dcrs(dataset, tabular_dir, syn_file)
    # for name, dcrs in dcrs_map.items():
    #     print(name)
    #     print(len(dcrs))

    colors = sns.color_palette("Set2", 2)  # 使用Seaborn的Set2调色板

    # 获取所有数据的全局最小值和最大值
    all_data = np.concatenate(list(dcrs_map.values()))
    data_min = np.percentile(all_data, 1)  # 使用第1百分位数
    data_max = np.percentile(all_data, 99) # 使用第99百分位数

    # 定义bins，使得所有直方图的x轴范围一致
    num_bin = 31
    bins = np.linspace(data_min, data_max, num_bin)

    # 创建子图，1行6列
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)

    # 绘制剩余的柱状图
    for i, (name, data) in enumerate(dcrs_map.items()):
        weights = np.ones_like(data) / len(data)  # 计算比例
        if i == 0:
            # sns.kdeplot(data, ax=axes[i], color=colors[0], shade=True)
            # axes[0].hist(data, bins=bins, color=colors[0], edgecolor='black', weights=weights)
            axes[0].hist(data, bins=bins, color=colors[0], weights=weights)
            axes[0].set_title(f'{name}', fontsize=BIG_FONT)
        else:
            # sns.kdeplot(data, ax=axes[i], color=colors[1], shade=True)
            # axes[i].hist(data, bins=bins, color=colors[1], edgecolor='black', weights=weights)
            axes[i].hist(data, bins=bins, color=colors[1], weights=weights)
            axes[i].set_title(f'{name}', fontsize=BIG_FONT)

    # 设置公共的x轴和y轴标签
    fig.text(0.5, 0.04, f'DCR on {dataset}', ha='center', fontsize=BIG_FONT)
    fig.text(0.04, 0.5, 'Ratio', va='center', rotation='vertical', fontsize=BIG_FONT)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=NORMAL_FONT)

    # 调整布局
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])

    # 将图像保存为PDF文件
    save_file = os.path.join(save_dir, f"dcr_{dataset.lower()}.pdf")
    plt.savefig(save_file, format='pdf')


def run_plot(
    dataset = "Beijing"
):
    if dataset == "Beijing":
        tabular_dir = "./data/tabular/beijing"
        syn_file = "./output/generations/bert-mistral-iv3-beijing-prefix-diffused-temp0.5.jsonl"
    elif dataset == "Default":
        tabular_dir = "./data/tabular/default"
        syn_file = "./output/generations/bert-mistral-iv3-default-prefix-diffused-temp0.5.jsonl"
    else:
        raise NotImplementedError()

    save_dir = "./output/figures"
    os.makedirs(save_dir, exist_ok=True)

    draw_dcrs(dataset, tabular_dir, syn_file, save_dir)


if __name__ == "__main__":
    fire.Fire(run_plot)

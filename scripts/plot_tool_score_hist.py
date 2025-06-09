# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import fire
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


SMALL_FONT = 16
NORMAL_FONT = 18
BIG_FONT = 20

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': SMALL_FONT,
    'font.family': 'serif',
    'pdf.fonttype': 42,
})


def load_gpt_judgement_scores(score_file):
    res = list()
    with open(score_file, "r") as rf:
        for line in rf:
            lj = json.loads(line)
            res.append(lj["score"])
    return res


def plot_score_histgram(score_real, score_syn):
    save_file = "./output/figures/tool_score_hist.pdf"

    print(f"#real: {len(score_real)}, #syn: {len(score_syn)}")

    # Define the range of x-axis values (0 to 10)
    values = np.arange(0, 11)

    # Count frequencies for each value
    counts1 = [score_real.count(value) for value in values]
    counts2 = [score_syn.count(value) for value in values]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define custom colors
    # colors = plt.cm.tab20c([0.0, 0.2])
    colors = sns.color_palette("Set2", 2)
    color1, color2 = colors

    # color1 = '#fe9929'
    # color2 = '#74a9cf'

    # color1 = '#efb186'  # light orange
    # color2 = '#a0c3e4'  # light blue

    # Width of a single bar
    bar_width = 0.4

    # Positions of bars on x-axis
    r1 = values - bar_width / 2
    r2 = values + bar_width / 2

    # Plotting the bar charts
    ax.bar(r1, counts1, width=bar_width, color=color1, label='Real Tools', align='center')
    ax.bar(r2, counts2, width=bar_width, color=color2, label='DiffLM Tools', align='center')

    # Set x-axis ticks and labels
    ax.set_xticks(values)
    ax.set_xlabel('Score', fontsize=NORMAL_FONT)
    ax.set_ylabel('Frequency', fontsize=NORMAL_FONT)
    # ax.set_title('Frequency Histogram of Two Datasets', fontsize=BIG_FONT)
    ax.legend(fontsize=SMALL_FONT)

    # Tight layout for better spacing
    fig.tight_layout()

    # Save the figure as a PDF
    plt.savefig(save_file, format='pdf')


def run_plot(
    real_score_file = "./output/evaluations/judgements/api_score_real.jsonl",
    syn_score_file = "./output/evaluations/judgements/api_score_prefix.jsonl",
):
    real_score = load_gpt_judgement_scores(real_score_file)
    syn_score = load_gpt_judgement_scores(syn_score_file)
    plot_score_histgram(real_score, syn_score)


if __name__ == "__main__":
    fire.Fire(run_plot)

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import random

import fire
from tqdm import tqdm
from utils.openai.azure_openai_multi_client import AzureOpenAIMultiClient


prompt = """
Given two sets of tools under the same category, you need to determine better_set by following these rules:
1. Comprehensiveness of Covered Functions: Evaluate which set covers more relevant and essential functions within the category.
2. Accuracy of Tool Descriptions: Check if the tool descriptions are clear, precise, and accurately reflect each tool’s functionality.
3. Difficulty of Calling the Tools: Assess the complexity involved in using the tools, considering the inputs and outputs required.
4. Overall Quality Assessment: Consider any additional factors that may impact the overall quality of the tool sets.

Set A:
{set_a_data}

Set B:
{set_b_data}

If one set is better based on the above criteria, indicate better_set as "A" or "B. If both sets are of similar quality, indicate better_set as “equal”.
Now, provide your reasoning in "reason" and indicate "better_set" ("A" or "B" or "equal") in JSON format.
"""


class ToolPreferenceJudger():
    def __init__(self, real_file, syn_file) -> None:
        self.rng = random.Random(42)
        self.real_samples = self.load_tool_jsons(real_file)
        self.syn_samples = self.load_tool_jsons(syn_file)
        self.gpt_client = AzureOpenAIMultiClient(
            endpoint="chats",
            concurrency=3,
            wait_interval=0.05,
            data_template={
                "model": "gpt-3.5-turbo-1106",
                "response_format": {"type": "json_object"},
            }
        )

    def load_tool_jsons(self, file):
        res = list()
        with open(file, "r") as rf:
            for line in rf:
                lj = json.loads(line)
                res.append(json.loads(lj["text"]))
        return res

    def judge_better_set(self, save_file, error_file):
        tool_reals = set()
        tool_syns = set()
        for one in self.real_samples:
            tool_reals.add(one["tool_name"])
        for one in self.syn_samples:
            tool_syns.add(one["tool_name"])
        tool_inters = tool_reals | tool_syns

        not_covered_real = 0
        not_covered_syn = 0
        for one in self.real_samples:
            if one["tool_name"] not in tool_inters:
                not_covered_real += 1
        for one in self.syn_samples:
            if one["tool_name"] not in tool_inters:
                not_covered_syn += 1
        print(f"real not covered: {not_covered_real}, syn not covered: {not_covered_syn}")

        def make_requests():
            for cat in tool_inters:
                real_apis = set()
                for one in self.real_samples:
                    if one["tool_name"] == cat:
                        real_apis.add(one["api_name"])
                syn_apis = set()
                for one in self.syn_samples:
                    if one["tool_name"] == cat:
                        syn_apis.add(one["api_name"])
                real_str = json.dumps(list(real_apis), indent=4, ensure_ascii=False)
                syn_str = json.dumps(list(syn_apis), indent=4, ensure_ascii=False)
                if self.rng.random() < 0.5:
                    set_a = real_str
                    set_b = syn_str
                    is_real_a = True
                else:
                    set_a = syn_str
                    set_b = real_str
                    is_real_a = False
                self.gpt_client.request(
                    data={
                        "messages": [
                            {"role": "system", "content": "You are a helpful annotator, that help user to annotate data."},
                            {"role": "user", "content": prompt.format(set_a_data=set_a, set_b_data=set_b)},
                        ]
                    },
                    metadata={
                        "tool_name": cat,
                        "mapper": {
                            "A": "real" if is_real_a else "syn",
                            "B": "syn" if is_real_a else "real",
                        }
                    },
                )
        self.gpt_client.run_request_function(make_requests)

        with open(save_file, "w") as wf, open(error_file, "w") as ef:
            for result in tqdm(self.gpt_client, total=len(tool_inters)):
                tool_name = result.metadata["tool_name"]
                try:
                    content = result.response.choices[0].message.content
                    ans = json.loads(content)
                    reason = ans["reason"]
                    choice = ans["better_set"]
                    if choice not in ["A", "B", "equal"]:
                        print(f"{tool_name} failed: {content}")
                        raise Exception(f"{tool_name} failed: {content}")
                    if choice in ["A", "B"]:
                        choice = result.metadata["mapper"][choice]
                    one = {
                        "tool_name": tool_name,
                        "answer": choice,
                        "reason": reason,
                    }
                    wf.write(json.dumps(one, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"{tool_name} failed: {e}")
                    ef.write(tool_name + "\n")


def run_judgement(
    real_file = "./data/tool/toolbench/train.jsonl",
    syn_file = "./output/generations/bert-mistral-iv3-toolbench-prefix-temp0.5.jsonl",
    save_dir = "./output/evaluations/judgements/"
):
    save_file = os.path.join(save_dir, "toolbench_preference_prefix.jsonl")
    error_file = os.path.join(save_dir, "toolbench_preference_prefix.error")

    judger = ToolPreferenceJudger(real_file, syn_file)
    judger.judge_better_set(save_file, error_file)


if __name__ == "__main__":
    fire.Fire(run_judgement)

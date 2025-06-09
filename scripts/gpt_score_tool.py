# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import random

import fire
from tqdm import tqdm
from utils.openai.azure_openai_multi_client import AzureOpenAIMultiClient


prompt = """
Given a API, evaluate it and assign a score from 0 to 10 (with 10 being the highest quality and 0 being the lowest). Consider the aspects listed below when evaluating the API. Provide your reasoning in "reason" and include the "score" in JSON format.

Evaluation Aspects:
1. Clarity and Completeness of the Tool Description: Does the tool_description clearly and thoroughly explain the purpose and functionalities of the tool?
2. Specificity and Accuracy of the API Name and Description: Is the api_name descriptive and appropriate? Does the api_description accurately and specifically describe what the API does?
3. Parameter Definition and Completeness: Are the parameters well-defined, including types, properties, and required fields? Do they cover all necessary inputs for the API to function effectively?
4. Consistency Between Tool and API Descriptions: Is there a logical connection between the tool_description and the api_description? Do they complement each other to provide a full understanding of the API’s capabilities?
5. Ease of Integration and Use: Based on the provided information, how easy would it be for a developer to integrate and use the API? Are there any missing details that could hinder implementation?
6. Overall Usefulness and Applicability: Considering potential use cases, how valuable is the API? Does it meet the needs of its intended audience?

Instructions:
- For the API, analyze it based on the evaluation aspects.
- Summarize your findings and reasoning in a clear and concise manner in "reason".
- Assign a final score between 0 and 10, reflecting the overall quality of the API in "score" field.
- Present the output in JSON format.

API:
{api_data}

Now, provide your answer.
"""


class ToolScorer():
    def __init__(self, tool_file) -> None:
        self.rng = random.Random(42)
        self.tool_samples = self.load_tool_jsons(tool_file)
        self.gpt_client = AzureOpenAIMultiClient(
            endpoint="chats",
            concurrency=3,
            wait_interval=0.1,
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

    def score_tool_quality(self, save_file, error_file):
        api_samples = self.tool_samples

        def make_requests():
            for idx, api in enumerate(api_samples):
                api_str = json.dumps(api, ensure_ascii=False, indent=4)
                self.gpt_client.request(
                    data={
                        "messages": [
                            {"role": "system", "content": "You are a helpful annotator, that help user to annotate data."},
                            {"role": "user", "content": prompt.format(api_data=api_str)},
                        ]
                    },
                    metadata={
                        "index": idx,
                    },
                )
        self.gpt_client.run_request_function(make_requests)

        with open(save_file, "w") as wf, open(error_file, "w") as ef:
            for result in tqdm(self.gpt_client, total=len(api_samples)):
                idx = result.metadata["index"]
                try:
                    api = api_samples[idx]
                    content = result.response.choices[0].message.content
                    ans = json.loads(content)
                    reason = ans["reason"]
                    score = ans["score"]
                    if int(score) not in range(0, 11):
                        raise Exception("score not in range(0, 11)")
                    one = {
                        "index": idx,
                        "tool_name": api["tool_name"],
                        "api_name": api["api_name"],
                        "reason": reason,
                        "score": score,
                    }
                    wf.write(json.dumps(one, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"{idx} failed: {e}")
                    ef.write(f"{idx}\n")


def run_judgement(
    real_file = "./data/tool/toolbench/train.jsonl",
    syn_file = "./output/generations/bert-mistral-iv3-toolbench-prefix-temp0.5.jsonl",
    save_dir = "./output/evaluations/judgements",
):
    real_save_file = os.path.join(save_dir, "api_score_real.jsonl")
    real_error_file = os.path.join(save_dir, "api_score_real.error")
    judger_real = ToolScorer(real_file)
    judger_real.score_tool_quality(real_save_file, real_error_file)

    syn_save_file = os.path.join(save_dir, "api_score_syn.jsonl")
    syn_error_file = os.path.join(save_dir, "api_score_syn.error")
    judger_syn = ToolScorer(syn_file)
    judger_syn.score_tool_quality(syn_save_file, syn_error_file)


if __name__ == "__main__":
    fire.Fire(run_judgement)

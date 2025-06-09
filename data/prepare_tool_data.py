# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import random
import numpy as np
from glob import glob


"""
"parameters": {
    "type": "object",
    "properties": {
        "key_summarize": {
            "type": "string",
            "description": "words for searching filter, should not exceed 100 words. If the Question is less than 100 words, just use the Question, else use the summarization of the Question."
        },
        "filter_count": {
            "type": "integer",
            "description": "how many filters to add. default 1. Note that you can only add one filter each time, but you will get many filters to choose from. Never say add more than one filter in your response."
        }
    },
    "required": ["filter_count"]
}
"""
def toolbench_to_openai_structure(tool_json):
    tool_name = tool_json["name"].strip()
    tool_desc = tool_json["tool_description"] or ""
    tool_desc = tool_desc.strip()
    for api in tool_json["api_list"]:
        api_name = api["name"].strip()
        api_desc = api["description"] or ""
        api_desc = api_desc.strip()
        params = dict()
        required = list()
        for one in api["required_parameters"]:
            params[one["name"]] = {
                "type": one["type"],
                "description": one["description"],
                "default": one["default"],
            }
            required.append(one["name"])
        for one in api["optional_parameters"]:
            params[one["name"]] = {
                "type": one["type"],
                "description": one["description"],
                "default": one["default"],
            }

        yield {
            "tool_name": tool_name,
            "tool_description": tool_desc,
            "api_name": api_name,
            "api_description": api_desc,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required,
            }
        }


def generate_all_tools(toolenv_dir, save_dir):
    records = list()
    words = list()
    tools = 0
    for file in glob(os.path.join(toolenv_dir, "*", "*.json")):
        tools += 1
        with open(file, "r") as rf:
            tool_json = json.load(rf)
        for rec in toolbench_to_openai_structure(tool_json):
            rec_text = json.dumps(rec, ensure_ascii=False)
            words.append(len(rec_text.split()))
            records.append({"text": rec_text})

    print(f"got {tools} tools, {len(words)} apis, average {np.mean(words)} tokens per api.")
    print(random.choice(records))

    random.shuffle(records)
    val_set = records[:1000]
    test_set = records[1000:2000]
    train_set = records[2000:]
    for name, dataset in [
        ("all", records), ("train", train_set), ("valid", val_set), ("test", test_set),
    ]:
        save_file = os.path.join(save_dir, name+".jsonl")
        with open(save_file, "w") as wf:
            for one in dataset:
                wf.write(json.dumps(one, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    base_dir = "/dataset/ToolBench/data/toolenv/tools"
    save_dir = "./data/tool"

    generate_all_tools(base_dir, save_dir)

# -*- coding: utf-8 -*-
# @Time    : 8/2/23
# @Author  : Yaojie Shen
# @Project : FastChat
# @File    : azure_openai_multi_client.py
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import time
from utils.openai.openai_multi_client import OpenAIMultiClient
from typing import Optional
from openai import AzureOpenAI, AsyncAzureOpenAI


azure_cilent = AzureOpenAI(
    azure_endpoint="", # your azure service endpoint
    api_key="", # your api key
    api_version="2024-03-01-preview",
)

def get_answer(question: str, max_tokens: int, model: str = "gpt-3.5-turbo"):
    ans = None
    for _ in range(3):
        try:
            response = azure_cilent.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question},
                ],
                max_tokens=max_tokens,
            )
            ans = response.choices[0].message.content
            return ans
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(1)
    return ans


class AzureOpenAIMultiClient(OpenAIMultiClient):
    def __init__(
        self,
        api_key: str | None = "",
        api_base: str = "",
        api_version: str | None = "2024-03-01-preview",
        concurrency: int = 10,
        max_retries: int = 10,
        wait_interval: float = 0,
        retry_multiplier: float = 1,
        retry_max: float = 60,
        endpoint: Optional[str] = None,
        data_template: Optional[dict] = None,
        metadata_template: Optional[dict] = None,
        custom_api=None,
    ):
        self._api_version = api_version
        super(AzureOpenAIMultiClient, self).__init__(
            api_key=api_key,
            api_base=api_base,
            concurrency=concurrency,
            max_retries=max_retries,
            wait_interval=wait_interval,
            retry_multiplier=retry_multiplier,
            retry_max=retry_max,
            endpoint=endpoint,
            data_template=data_template,
            metadata_template=metadata_template,
            custom_api=custom_api
        )

    def prepare_client(self):
        return AsyncAzureOpenAI(
            api_key=self._api_key,
            api_version=self._api_version,
            azure_endpoint=self._api_base,
        )

# Copyright (c) 2023 cozodb
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025
#
# Original file was released under MIT, with the full license text
# available at https://github.com/cozodb/openai-multi-client/blob/main/openai_multi_client/__init__.py.
#
# This modified file is released under the same license.

import logging
import asyncio
from dataclasses import dataclass
from threading import Thread
from typing import Any, Optional, Callable

from aioprocessing import AioJoinableQueue, AioQueue, AioEvent, AioLock
from tenacity import wait_random_exponential, stop_after_attempt, AsyncRetrying, RetryError
from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


@dataclass
class Payload:
    endpoint: str
    data: dict
    metadata: Optional[dict]
    max_retries: int
    retry_multiplier: float
    retry_max: float
    attempt: int = 0
    failed: bool = False
    response: Any = None
    callback: Callable[["Payload"], None] = None
    preprocess: Callable[["Payload"], None] = None

    def call_callback(self):
        if self.callback:
            self.callback(self)

    def call_preprocess(self):
        if self.preprocess:
            self.preprocess(self)


class OpenAIMultiClient:
    def __init__(
        self,
        api_key: str = "",
        api_base: str | None = None,
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
        self._api_key = api_key
        self._api_base = api_base
        self._endpoint = endpoint
        self._wait_interval = wait_interval
        self._data_template = data_template or {}
        self._metadata_template = metadata_template or {}
        self._max_retries = max_retries
        self._retry_multiplier = retry_multiplier
        self._retry_max = retry_max
        self._concurrency = concurrency
        self._loop = asyncio.new_event_loop()
        self._in_queue = AioJoinableQueue(maxsize=9999999)
        self._out_queue = AioQueue(maxsize=concurrency)
        self._event_loop_thread = Thread(target=self._run_event_loop)
        self._event_loop_thread.start()
        self._mock_api = custom_api
        self._processing_lock = [AioLock() for _ in range(concurrency)]
        self._closed = AioEvent()
        self._close_lock = AioLock()
        for i in range(concurrency):
            asyncio.run_coroutine_threadsafe(self._worker(i), self._loop)
        self._async_client = self.prepare_client()

    def prepare_client(self):
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
        )

    def run_request_function(self, input_function, *args, stop_at_end=True, **kwargs):
        if stop_at_end:
            def f(*args, **kwargs):
                input_function(*args, **kwargs)
                self.close()
        else:
            f = input_function
        input_thread = Thread(target=f, args=args, kwargs=kwargs)
        input_thread.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _process_payload(self, payload: Payload) -> Payload:
        logger.debug(f"Processing payload: {payload}")
        if self._mock_api:
            payload.response = await self._mock_api(payload)
        elif payload.endpoint == "completions":
            payload.response = await self._async_client.completions.create(**payload.data)
        elif payload.endpoint == "chat.completions" or payload.endpoint == "chats":
            payload.response = await self._async_client.chat.completions.create(**payload.data)
        elif payload.endpoint == "embeddings":
            payload.response = await self._async_client.embeddings.create(**payload.data)
        elif payload.endpoint == "edits":
            payload.response = await self._async_client.edits.create(**payload.data)
        elif payload.endpoint == "images":
            payload.response = await self._async_client.images.create(**payload.data)
        elif payload.endpoint == "fine-tunes":
            payload.response = await self._async_client.fine_tunes.create(**payload.data)
        else:
            raise ValueError(f"Unknown endpoint {payload.endpoint}")
        payload.call_callback()
        return payload

    async def _worker(self, i):
        while True:
            payload = await self._in_queue.coro_get()

            if payload is None:
                logger.debug(f"Exiting worker {i}")
                self._in_queue.task_done()
                break

            payload.call_preprocess()
            try:
                async for attempt in AsyncRetrying(
                        wait=wait_random_exponential(multiplier=payload.retry_multiplier, max=payload.retry_max),
                        stop=stop_after_attempt(payload.max_retries)):
                    with attempt:
                        try:
                            payload.attempt = attempt.retry_state.attempt_number
                            payload = await self._process_payload(payload)
                            await self._out_queue.coro_put(payload)
                            self._in_queue.task_done()
                        except Exception:
                            logger.exception(f"Error processing {payload}")
                            raise
            except RetryError:
                payload.failed = True
                logger.error(f"Failed to process {payload}")
                await self._out_queue.coro_put(payload)
                self._in_queue.task_done()
            await asyncio.sleep(self._wait_interval)

    def close(self):
        try:
            with self._close_lock:
                if not self._closed.is_set():
                    self._closed.set()
                    for i in range(self._concurrency):
                        self._in_queue.put(None)
                    self._in_queue.join()
                    self._out_queue.put(None)
                    self._loop.call_soon_threadsafe(self._loop.stop)
                    self._event_loop_thread.join()
        except Exception as e:
            logger.error(f"Error closing: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        out = self._out_queue.get()
        if out is None:
            raise StopIteration
        return out

    def request(self,
                data: dict,
                endpoint: Optional[str] = None,
                metadata: Optional[dict] = None,
                preprocess: Any = None,
                callback: Any = None,
                max_retries: Optional[int] = None,
                retry_multiplier: Optional[float] = None,
                retry_max: Optional[float] = None):
        with self._close_lock:
            if not self._closed.is_set():
                payload = Payload(
                    endpoint=endpoint or self._endpoint,
                    data={**self._data_template, **data},
                    metadata={**self._metadata_template, **(metadata or {})},
                    preprocess=preprocess,
                    callback=callback,
                    max_retries=max_retries or self._max_retries,
                    retry_multiplier=retry_multiplier or self._retry_multiplier,
                    retry_max=retry_max or self._retry_max
                )
                self._in_queue.put(payload)
            else:
                raise RuntimeError("Cannot append request to a closed client!")

    def pull_all(self):
        for _ in self:
            pass


class OrderedPayload(Payload):
    put_counter: int

    def __init__(self, *args, put_counter, **kwargs):
        super().__init__(*args, **kwargs)
        self.put_counter = put_counter


class OpenAIMultiOrderedClient(OpenAIMultiClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._put_counter = 0
        self._get_counter = 0
        self._get_cache = {}
        self._stopped = False

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._stopped:
                out = None
            else:
                out = self._out_queue.get()
            if out is None:
                self._stopped = True
                if self._get_counter == self._put_counter:
                    raise StopIteration
                else:
                    out = self._get_cache[self._get_counter]
                    del self._get_cache[self._get_counter]
                    self._get_counter += 1
                    return out

            data_counter = out.put_counter
            if data_counter == self._get_counter:
                self._get_counter += 1
                return out
            self._get_cache[data_counter] = out
            if self._get_counter in self._get_cache:
                out = self._get_cache[self._get_counter]
                del self._get_cache[self._get_counter]
                self._get_counter += 1
                return out

    def request(self,
                data: dict,
                endpoint: Optional[str] = None,
                metadata: Optional[dict] = None,
                callback: Any = None,
                max_retries: Optional[int] = None,
                retry_multiplier: Optional[float] = None,
                retry_max: Optional[float] = None):
        payload = OrderedPayload(
            endpoint=endpoint or self._endpoint,
            data={**self._data_template, **data},
            metadata={**self._metadata_template, **(metadata or {})},
            callback=callback,
            max_retries=max_retries or self._max_retries,
            retry_multiplier=retry_multiplier or self._retry_multiplier,
            retry_max=retry_max or self._retry_max,
            put_counter=self._put_counter
        )
        self._put_counter += 1
        self._in_queue.put(payload)

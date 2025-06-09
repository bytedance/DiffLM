# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import math
import logging
import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache

from utils.module import split_heads
from utils.dataset import IGNORE_TOKEN_ID


logger = logging.getLogger()


VAE_PREFIX_SOFT_PROMPT_TEXT = "Generate a structured output based on the provided soft prompt, which is a random sample from the latent space. Ensure the output adheres to the specified format consistently."


class PromptEmbedding(torch.nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`torch.nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt embedding.

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """

    def __init__(self, config, num_virtual_tokens, init_text=None):
        super().__init__()
        self.config = config
        self.num_virtual_tokens = num_virtual_tokens
        self.init_text = init_text
        self.embedding = torch.nn.Embedding(num_virtual_tokens, config.hidden_size)

    def post_init_embeddings(self, word_embeddings=None):
        if self.init_text is not None and word_embeddings is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path, use_fast=False)
            init_token_ids = tokenizer(self.init_text, add_special_tokens=False)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            logger.info(f"Ininting M soft prompts from {self.init_text} -> {num_text_tokens}")
            if num_text_tokens > self.num_virtual_tokens:
                init_token_ids = init_token_ids[:self.num_virtual_tokens]
            elif num_text_tokens < self.num_virtual_tokens:
                num_reps = math.ceil(self.num_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            # Get and embed the text tokens
            init_token_ids = init_token_ids[:self.num_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids).to(word_embeddings.weight.device)
            word_embedding_weights = word_embeddings(init_token_ids).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)

    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings


class LatentProjector(nn.Module):
    def __init__(
        self,
        decoder_config,
        latent_method,
        latent_size,
        latent_adapter_size,
        latent_init_text=None,
    ) -> None:
        super().__init__()

        self.hidden_size = decoder_config.hidden_size
        self.num_layers = decoder_config.num_hidden_layers
        self.num_kv_heads = decoder_config.num_key_value_heads
        self.head_dim = decoder_config.hidden_size // decoder_config.num_attention_heads

        self.latent_method = latent_method
        self.latent_size = latent_size
        self.latent_adapter_size = latent_adapter_size

        if latent_method == "input_embed":
            self.vae_embeds = nn.Linear(
                self.latent_size, self.hidden_size, bias=False
            ) # share the same latent vector as the embeddings
        elif latent_method == "kv_memory":
            self.vae_memory_size = self.head_dim * self.num_kv_heads
            self.vae_memory_proj = nn.Linear(
                self.latent_size, self.vae_memory_size * self.num_layers, bias=False
            ) # different latent vector for each layer
        elif latent_method == "soft_prompt":
            self.vae_soft_proj = nn.Linear(
                self.latent_size, self.hidden_size * self.latent_adapter_size, bias=False,
            )
        elif latent_method == "prefix_soft_prompt":
            self.prefix_soft_embeds = PromptEmbedding(
                decoder_config, self.latent_adapter_size, latent_init_text,
            )
            self.vae_soft_proj = nn.Linear(
                self.latent_size, self.hidden_size * self.latent_adapter_size, bias=False,
            )
        else:
            raise NotImplementedError(f"not supported latent projection method -> {latent_method}")

    def _make_dummy_tensor(self, bsz, length, device, fill_value=None):
        res = torch.ones(
            (bsz, length), dtype=torch.long, device=device
        )
        if fill_value is not None:
            res = res.fill_(fill_value)
        return res

    def prepare_latent_projections(self, latent_vector, shared_memory=False):
        # VAE specific
        if self.latent_method == "input_embed": # used as extra embeddings to add on the others
            return self.vae_embeds(latent_vector)

        elif self.latent_method == "kv_memory":
            # (bs, memory_size * num_layers)
            tmp_memory = self.vae_memory_proj(latent_vector)
            if shared_memory:
                # the same latent vector shared by all layers
                tmp_memory = [tmp_memory.unsqueeze(-2), tmp_memory.unsqueeze(-2)] # query, key
                tmp_memory = [tmp_memory] * len(self.num_layers)
            else:
                # different latent vectors for each layer
                # (num_layers, bs, memory_size)
                past_split = torch.split(tmp_memory.unsqueeze(1), self.vae_memory_size, dim=2)
                tmp_memory = list(zip(past_split, past_split))
            # resize the past_key, past_values
            return list(
                tuple([
                    split_heads(past_key, self.num_kv_heads, self.head_dim),
                    split_heads(past_value, self.num_kv_heads, self.head_dim),
                ]) for past_key, past_value in tmp_memory
            )

        elif self.latent_method == "soft_prompt":
            # (bs, hidden_size * 8)
            vae_soft_prompt = self.vae_soft_proj(latent_vector)
            # (bs, 8->soft_tokens, hidden_size)
            return vae_soft_prompt.view(-1, self.latent_adapter_size, self.hidden_size)

        elif self.latent_method == "prefix_soft_prompt":
            bs = latent_vector.size(0)
            # 生成包含所有虚拟 token 索引的 tensor，形状为 (batch_size, num_virtual_tokens)
            indices = torch.arange(self.latent_adapter_size).unsqueeze(0).repeat(bs, 1)
            indices = indices.to(latent_vector.device)
            # (batch_size, num_virtual_tokens, hidden_size)
            prefix_soft_prompt = self.prefix_soft_embeds(indices)
            # (bs, hidden_size * 8)
            vae_soft_prompt = self.vae_soft_proj(latent_vector)
            # (bs, 8->soft_tokens, hidden_size)
            vae_soft_prompt = vae_soft_prompt.view(-1, self.latent_adapter_size, self.hidden_size)
            return torch.cat([prefix_soft_prompt, vae_soft_prompt], dim=1)

    def forward(self, latent_vector, inputs_embeds, attention_mask, labels, past_key_values=None):
        if latent_vector is None:
            return inputs_embeds, attention_mask, labels, past_key_values

        bsz, seq_length = inputs_embeds.shape[:2]

        # VAE specific
        vae_inputs = self.prepare_latent_projections(latent_vector, shared_memory=False)

        # VAE extra input embedding
        if self.latent_method == "input_embed":
            inputs_embeds = inputs_embeds + vae_inputs.unsqueeze(1)

        # VAE kv cache memory
        elif self.latent_method == "kv_memory":
            memory_size = vae_inputs[0][0].size(-2)
            if past_key_values is None:
                past_key_values = vae_inputs
            else:
                # raise NotImplementedError("Using for VAE latent memory tuning, not support past_key_values, need to check...")
                if isinstance(past_key_values, DynamicCache):
                    past_key_values = past_key_values.to_legacy_cache()
                past_key_values = [
                    tuple([
                        torch.cat([latent_key, past_key_values[layer_idx][0]], dim=-2),
                        torch.cat([latent_value, past_key_values[layer_idx][1]], dim=-2),
                    ]) for layer_idx, (latent_key, latent_value) in enumerate(vae_inputs)
                ]
            if attention_mask is not None:
                memory_attn = self._make_dummy_tensor(bsz, memory_size, device=inputs_embeds.device)
                attention_mask = torch.cat([memory_attn, attention_mask], dim=1)
            if isinstance(past_key_values, list):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # VAE soft prompt
        elif self.latent_method in {"soft_prompt", "prefix_soft_prompt"}:
            inputs_embeds = torch.cat([vae_inputs, inputs_embeds], dim=1)
            _bsz, _lg = vae_inputs.shape[:2]
            if attention_mask is not None:
                latent_attn = self._make_dummy_tensor(_bsz, _lg, device=inputs_embeds.device)
                attention_mask = torch.cat([latent_attn, attention_mask], dim=1)
            if labels is not None:
                latent_target = self._make_dummy_tensor(_bsz, _lg, device=inputs_embeds.device, fill_value=IGNORE_TOKEN_ID)
                labels = torch.cat([latent_target, labels], dim=1)

        return inputs_embeds, attention_mask, labels, past_key_values

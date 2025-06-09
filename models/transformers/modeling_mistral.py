# Copyright (c) 2023 Mistral AI and The HuggingFace Inc. team.
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py.
#
# This modified file is released under the same license.

import logging
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralRMSNorm,
    MistralModel,
    MistralDecoderLayer,
    MistralPreTrainedModel,
)

from utils.module import CausalLMPooling
from models.transformers.utils_vae import LatentProjector


logger = logging.getLogger(__name__)


class MistralForVAEEncoder(MistralPreTrainedModel):
    def __init__(self, config: MistralConfig, vae_latent_size=32, pooling_strategy="last"):
        super().__init__(config)
        self.model = MistralModel(config)

        # VAE specific, hidden vector -> mu and logvar
        self.latent_linear = nn.Linear(config.hidden_size, 2 * vae_latent_size, bias=False)
        self.pooler = CausalLMPooling(strategy=pooling_strategy)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        last_hidden_state = output[0]
        pooled_state = self.pooler(last_hidden_state, attention_mask)
        return {
            "encoded_state": pooled_state,
        }


class MistralForVAEDecoder(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: MistralConfig,
        vae_latent_size: Optional[int] = 32,
        vae_adapter_size: Optional[int] = 8,
        vae_latent_method: Optional[str] = "soft_prompt",
        vae_prefix_text: Optional[str] = None,
    ):
        super().__init__(config)
        self.model = MistralModel(config)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.latent_proj = LatentProjector(
            config,
            vae_latent_method,
            vae_latent_size,
            vae_adapter_size,
            latent_init_text=vae_prefix_text,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        latent_vector: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # passing embeddings to model
        inputs_embeds = self.model.embed_tokens(input_ids)

        # VAE Specific
        inputs_embeds, attention_mask, labels, past_key_values = self.latent_proj(
            latent_vector = latent_vector,
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            labels = labels,
            past_key_values = past_key_values,
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        # return loss
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

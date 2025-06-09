# Copyright (c) 2020 Microsoft Research.
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/ChunyuanLI/Optimus/blob/master/code/examples/big_ae/modules/vae.py.
#
# This modified file is released under the same license.

import os
import json
import torch
import logging
import torch.nn as nn

from safetensors.torch import load_model
from transformers import AutoConfig, AutoTokenizer
from models.transformers import (
    BertForVAEEncoder,
    T5ForVAEEncoder,
    LlamaForVAEEncoder,
    LlamaForVAEDecoder,
    MistralForVAEEncoder,
    MistralForVAEDecoder,
    Qwen2ForVAEEncoder,
    Qwen2ForVAEDecoder,
    GPT2LMHeadModelForVAEDecoder,
)
from utils.common import dotdict, is_main_process
from utils.generation import generate_from_latent_vector


logger = logging.getLogger(__name__)

SAVED_STATE_NAME = "model.safetensors"
MODEL_CONFIG_NAME = "config_model.json"
DATA_CONFIG_NAME = "config_data.json"
TRAINER_CONFIG_NAME = "config_trainer.json"


def prepare_vae_model(model_args, training_args=None, local_rank=0):
    # NOTE to compatible with old checkpoint
    if isinstance(model_args, dict) and "vae_prefix_text" not in model_args:
        model_args.vae_prefix_text = None

    torch_dtype = "auto"
    if training_args is not None and training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args is not None and training_args.fp16:
        torch_dtype = torch.float16

    is_llm_encoder = False
    if "bert" in model_args.encoder_name_or_path.lower():
        encoder_cls = BertForVAEEncoder
    elif "t5" in model_args.encoder_name_or_path.lower():
        encoder_cls = T5ForVAEEncoder
    elif "llama" in model_args.encoder_name_or_path.lower():
        is_llm_encoder = True
        encoder_cls = LlamaForVAEEncoder
    elif "mistral" in model_args.encoder_name_or_path.lower():
        is_llm_encoder = True
        encoder_cls = MistralForVAEEncoder
    elif "qwen" in model_args.encoder_name_or_path.lower():
        is_llm_encoder = True
        encoder_cls = Qwen2ForVAEEncoder
    else:
        raise NotImplementedError(f"not supported encoder {model_args.encoder_name_or_path}")

    if "gpt2" in model_args.decoder_name_or_path.lower() or "gpt-2" in model_args.decoder_name_or_path.lower():
        decoder_cls = GPT2LMHeadModelForVAEDecoder
    elif "llama" in model_args.decoder_name_or_path.lower():
        decoder_cls = LlamaForVAEDecoder
    elif "mistral" in model_args.decoder_name_or_path.lower():
        decoder_cls = MistralForVAEDecoder
    elif "qwen" in model_args.decoder_name_or_path.lower():
        decoder_cls = Qwen2ForVAEDecoder
    else:
        raise NotImplementedError(f"not supported encoder {model_args.decoder_name_or_path}")

    ## Encoder
    if is_llm_encoder:
        encoder_config = AutoConfig.from_pretrained(
            model_args.encoder_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
        )
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            model_args.encoder_name_or_path,
            model_max_length=model_args.encoder_max_length,
            padding_side="left" if model_args.left_padding else "right",
        )
        encoder_tokenizer.pad_token = encoder_tokenizer.unk_token
        encoder_tokenizer.pad_token_id = encoder_tokenizer.unk_token_id
    else:
        encoder_config = AutoConfig.from_pretrained(
            model_args.encoder_name_or_path,
        )
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            model_args.encoder_name_or_path,
            model_max_length=model_args.encoder_max_length,
        )
    encoder_model = encoder_cls.from_pretrained(
        model_args.encoder_name_or_path,
        config=encoder_config,
        vae_latent_size=model_args.vae_latent_size,
    )

    ## Decoder
    decoder_config = AutoConfig.from_pretrained(
        model_args.decoder_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        model_args.decoder_name_or_path,
        use_fast=False,
        add_bos_token=True,
        add_eos_token=True,
        padding_side="left" if model_args.left_padding else "right",
        model_max_length=model_args.decoder_max_length,
    )
    decoder_model = decoder_cls.from_pretrained(
        model_args.decoder_name_or_path,
        config = decoder_config,
        vae_latent_size = model_args.vae_latent_size,
        vae_adapter_size = model_args.vae_adapter_size,
        vae_latent_method = model_args.vae_latent_method,
        vae_prefix_text = model_args.vae_prefix_text,
    )

    # Temporary fix for llama-3 eos token
    if "Llama-3" in model_args.decoder_name_or_path:
        from tokenizers import processors
        bos = decoder_tokenizer.bos_token
        eos = decoder_tokenizer.eos_token
        decoder_tokenizer._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single=f"{bos}:0 $A:0 {eos}:0",
                    pair=f"{bos}:0 $A:0 {bos}:1 $B:1 {eos}:1",
                    special_tokens=[
                        (bos, decoder_tokenizer.bos_token_id),
                        (eos, decoder_tokenizer.eos_token_id),
                    ],
                ),
            ]
        )

    # NOTE: if the token_id exceed the vocab_size will cause failing in training process!
    # we need add special config and resize the embedding size!
    if decoder_tokenizer.unk_token is not None:
        decoder_tokenizer.pad_token = decoder_tokenizer.unk_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.unk_token_id
    else:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id
    decoder_model.resize_token_embeddings(len(decoder_tokenizer))

    if is_main_process(local_rank):
        logger.info(f"[Model] Creating VAE models from {model_args}")
        logger.info(f"[Model] tokens len: {len(decoder_tokenizer)}, set {decoder_tokenizer.pad_token} for padding...")

    model_vae = VAELanguageModel(model_args, encoder_model, decoder_model, encoder_tokenizer, decoder_tokenizer)
    return model_vae, encoder_tokenizer, decoder_tokenizer


def get_vae_data_args(model_path):
    with open(os.path.join(model_path, DATA_CONFIG_NAME), "r") as rf:
        data_args = json.load(rf)
    return data_args


class VAELanguageModel(nn.Module):
    """VAE with normal prior"""
    def __init__(self, args, encoder_model, decoder_model, encoder_tokenizer, decoder_tokenizer):
        super(VAELanguageModel, self).__init__()
        self.config = args

        self.latent_size = args.vae_latent_size
        self.threshold_kl = args.threshold_kl
        self.deterministic_connect = args.deterministic_connect
        self.length_weighted_loss = args.length_weighted_loss

        self.encoder = encoder_model
        self.decoder = decoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        if args.freeze_decoder:
            self.freeze_decoder()

        # Standard Normal prior
        # for reparameterize && latent generation
        self.prior = torch.distributions.normal.Normal(
            loc=torch.zeros(self.latent_size),
            scale=torch.ones(self.latent_size),
        )

        self._keys_to_ignore_on_save = None
        self.post_init()

    def post_init(self, ):
        latent_proj = self.decoder.latent_proj
        if latent_proj.latent_method == "prefix_soft_prompt":
            latent_proj.prefix_soft_embeds.post_init_embeddings(
                word_embeddings = self.decoder.get_input_embeddings(),
            )

    def freeze_decoder(self):
        for name, param in self.decoder.model.named_parameters():
            param.requires_grad = False
        for name, param in self.decoder.lm_head.named_parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def connect(self, encoder_feat, deterministic_connect=False, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        # (batch_size, nz)
        mean, logvar = self.encoder.latent_linear(encoder_feat).chunk(2, -1)
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        if self.deterministic_connect or deterministic_connect:
            logvar.fill_(0.0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        kl_reg = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1)

        if self.threshold_kl is not None:
            kl_mask = (kl_reg > self.threshold_kl).float()
            kl_reg = (kl_mask * kl_reg)

        kl_reg = torch.mean(kl_reg.mean(dim=1))
        return z, kl_reg

    def forward(
        self,
        input_ids,
        decoder_ids,
        decoder_labels,
        attention_mask=None,
        decoder_attn_mask=None,
        beta=0.0,
        **kwargs,
    ):
        enc_out = self.encoder(input_ids, attention_mask=attention_mask)
        encoded_feat = enc_out["encoded_state"]

        latent_z, loss_kl = self.connect(encoded_feat, nsamples=1)
        latent_z = latent_z.squeeze(1)

        # Decoding
        dec_out = self.decoder(
            input_ids=decoder_ids,
            latent_vector=latent_z,
            attention_mask=decoder_attn_mask,
            labels=decoder_labels,
            return_dict=True,
        )
        loss_rec = dec_out.loss

        loss = loss_rec + beta * loss_kl

        return dict(
            loss = loss,
            loss_kl = loss_kl,
            loss_rec = loss_rec,
        )

    def reconstruct(self, text_x, generation_settings={}, num_return_sequences=1):
        res_texts = list()
        for _ in range(num_return_sequences):
            z, vae_feat = self.encode(text_x, device=generation_settings.device)
            res_texts.append(
                self.decode(z, generation_settings)
            )
        return res_texts

    def encode(self, text_x, device="cpu"):
        inputs = self.encoder_tokenizer(
            text_x,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        enc_out = self.encoder(**inputs)
        encoder_feat = enc_out["encoded_state"]

        mu, logvar = self.encoder.latent_linear(encoder_feat).chunk(2, -1)
        latent_z = self.reparameterize(mu, logvar)

        # latent_z, loss_kl = self.connect(encoder_feat, nsamples=1)
        # latent_z = latent_z.squeeze(1)
        return latent_z, mu, logvar

    def decode(self, z, generation_args={}):
        context_tokens = self.decoder_tokenizer.encode(
            self.decoder_tokenizer.bos_token,
            add_special_tokens=False,
        )
        return generate_from_latent_vector(
            self.decoder,
            self.decoder_tokenizer,
            context_tokens = context_tokens,
            latent_vector = z,
            **generation_args,
        )

    # TODO: delete after utils.generation checked
    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        outputs = self.decoder(
            input_ids=x,
            latent_vector=z,
            labels=x,
        )
        loss_rec = outputs[0]
        return -loss_rec

    @staticmethod
    def load_from_saved_model(saved_model_path, device=None):
        config_file = os.path.join(saved_model_path, MODEL_CONFIG_NAME)
        pretrained_file = os.path.join(saved_model_path, SAVED_STATE_NAME)

        with open(config_file, "r") as rf:
            model_args = dotdict(json.load(rf))
        model, _, _ = prepare_vae_model(model_args)
        load_model(model, pretrained_file, strict=False)

        if device is not None:
            model = model.to(device)

        return model

    def save_model_config(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, MODEL_CONFIG_NAME), "w") as wf:
            json.dump(
                self.config.to_dict(), wf,
                indent=4, ensure_ascii=False,
            )

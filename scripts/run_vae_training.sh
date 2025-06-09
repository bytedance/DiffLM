#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

export WANDB_MODE=disabled

DATA_DIR=./data/tabular/shoppers
SAVE_DIR=./output/models

[ -d "${SAVE_DIR}" ] && rm -rf "${SAVE_DIR}"

TOTAL_EPOCH=200
SAVE_STEP=200
MICRO_BATCH_SIZE=2
LATENT_METHOD='soft_prompt'
LEARNING_RATE='1e-5'

ENCODER="bert-base-uncased" # bert-base-cased, google-t5/t5-3b
DECODER="mistralai/Mistral-7B-Instruct-v0.3" # meta-llama/Meta-Llama-3.1-8B-Instruct
FSDP_LAYER="MistralDecoderLayer" # MistralDecoderLayer, LlamaDecoderLayer

torchrun --nproc_per_node="${DIST_NPROC_PER_NODE}" --nnodes="${DIST_NNODES}" --node_rank="${DIST_NODE_RANK}" \
    --master_addr="${DIST_MASTER_ADDR}" --master_port="${DIST_MASTER_PORT}" \
    run_vae_pretraining_backtracking.py \
    --data_path ${DATA_DIR}/train.jsonl \
    --eval_data_path ${DATA_DIR}/valid.jsonl \
    --output_dir ${SAVE_DIR} \
    --lazy_preprocess \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap ${FSDP_LAYER} \
    --encoder_name_or_path ${ENCODER} \
    --decoder_name_or_path ${DECODER} \
    --encoder_max_length 512 \
    --decoder_max_length 512 \
    --left_padding \
    --attn_implementation "flash_attention_2" \
    --freeze_decoder True \
    --vae_adapter_size 32 \
    --vae_latent_size 1024 \
    --vae_latent_method ${LATENT_METHOD} \
    --tf32 True \
    --bf16 True \
    --do_train \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.005 \
    --lr_scheduler_type "cosine" \
    --max_beta 1e-1 \
    --min_beta 1e-3 \
    --backtracking_interval 2 \
    --early_stopping_interval 2 \
    --num_train_epochs ${TOTAL_EPOCH} \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEP} \
    --save_total_limit 2 \
    --save_only_model \
    --save_best_model_at_end \
    --metric_for_best_model loss \
    --greater_is_better False \
    --logging_steps 1

#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

if [ $# -ne 1 ]; then
    echo "usage $0 [PATH_OF_VAE_MODEL]"
    exit 1
fi


input_path="$1"

if [ ! -e "$input_path" ]; then
    echo "[ERROR] '$input_path' not exist!!!"
    exit 2
fi

# Encoding train/val/test data to embeddings with VAE encoder
python run_encoding.py --model_path ${input_path}

# training Latent Diffusion Model
CMD="\
    python run_diffusing.py \
    --vae_model_path ${input_path} \
    --denoising_impl mlp \
    --denoising_layers 5 \
    --dim_noise 4096 \
    --learning_rate 1e-4 \
    --do_eval
"
echo $CMD
eval $CMD
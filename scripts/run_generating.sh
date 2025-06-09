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

CMD="\
    python3 run_generation.py \
        --model_path ${input_path} \
        --output_dir ./output/generations \
        --temperature 0.5 \
        --check_column \
        --do_diffusion \
        --diffusion_name mlp5
"
echo $CMD
eval $CMD
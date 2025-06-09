# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import random
import logging
import argparse

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import torch
from tqdm import tqdm
from pathlib import Path
from functools import partial

from models.auto_encoder import VAELanguageModel, get_vae_data_args
from models.diffusion import LatentDiffuser, DIFFUSION_MODEL_DIR
from utils.common import setup_logger
from utils.generation import generate_from_latent_vector
from utils.diffusion import sample
from run_evaluating import run_evaluation
from run_encoding import load_data_samples


logger = logging.getLogger()
setup_logger(logger)

model_info = None


def init_model(vae_path, do_diffusion, num_gpu, diff_name=None):
    # 在这里初始化 Attacker 等需要全局使用的变量，通过current_process可以获取当前进程的唯一ID，从而设置环境变量
    global model_info

    p_idx = int(mp.current_process()._identity[0]) # base started with 1
    gpu_i = (p_idx - 1) % num_gpu
    gpu_device = torch.device(f"cuda:{gpu_i}")

    # load Diffusion Model
    diff_name = diff_name or "mlp1"
    diff_model_path = os.path.join(vae_path, f"{DIFFUSION_MODEL_DIR}_{diff_name}")

    if do_diffusion:
        assert os.path.exists(diff_model_path), f"{diff_model_path} not exists, please ensure the diffusion model is trained."
        diff_model, mean_z = LatentDiffuser.load_from_saved_model(diff_model_path, device=gpu_device)
    else:
        diff_model, mean_z = None, None

    # model_args["attn_implementation"] = "eager"
    vae_model = VAELanguageModel.load_from_saved_model(vae_path, device=gpu_device)

    model_info = {
        "vae_model": vae_model,
        "diff_model": diff_model,
        "diff_mean_z": mean_z,
        "device": gpu_device,
    }

    logger.info(
        "*******************************\n"
        f"initializing process-{p_idx}...]\n"
        f"\t GPU: {gpu_i}\n"
        "*******************************\n",
        # flush=True,
    )


@torch.inference_mode()
def get_prior_from_diffusion(
    vae_model,
    diff_model,
    train_z_mean = None,
    device = torch.device("cpu"),
):
    if train_z_mean is None:
        train_z_mean = 0.0

    rnd_prior = vae_model.prior.sample().unsqueeze(0)
    # rnd_prior = torch.randn([1, vae_model.latent_size], device=device)
    prior_next = sample(diff_model.precond_denoise_fn, rnd_prior, device=device)
    prior_next = prior_next * 2 + train_z_mean
    return prior_next


@torch.inference_mode()
def get_random_gaussian_prior(vae_model, device=torch.device("cpu")):
    rnd_prior = vae_model.prior.sample()
    return rnd_prior.unsqueeze(0).to(device)


def check_required_columns(gen_text, required_cols):
    try:
        res = json.loads(gen_text)
        return set(res.keys()) == set(required_cols)
    except:
        return False


def check_value_valid(gen_text, valid_val, check_number=False):
    try:
        res = json.loads(gen_text)
        flag = True
        for col, col_info in valid_val.items():
            if col not in res:
                flag = False
                break
            if col_info["type"] == "category":
                cands = col_info["candidates"]
                if res[col] not in cands:
                    flag = False
                    break
            elif check_number and col_info["type"] == "float":
                col_min = col_info["min"]
                col_max = col_info["max"]
                if res[col] < col_min or res[col] > col_max:
                    flag = False
                    break
        return flag
    except:
        return False


@torch.inference_mode()
def generate_one_random_sample(gen_idx, generation_settings, train_samples=None, required_cols=None, valid_vals=None):
    assert model_info is not None, "initialize failed, please check..."

    vae_model = model_info["vae_model"]
    diff_model = model_info["diff_model"]
    diff_mean_z = model_info["diff_mean_z"]
    device = model_info["device"]

    decoder_tokenizer = vae_model.decoder_tokenizer
    if "Qwen2" in vae_model.config.decoder_name_or_path:
        context_tokens = [151643, ] # hack Qwen2, which has no bos_token in tokenizer...
    else:
        context_tokens = decoder_tokenizer.encode(
            decoder_tokenizer.bos_token,
            add_special_tokens=False,
        )

    one_text = None
    sample_times = 0
    while one_text is None:
        sample_times += 1
        if train_samples is not None:
            rnd_sample = random.choice(train_samples)["text"]
            latent_z, mean, logvar = vae_model.encode(rnd_sample, device)
        elif diff_model is None:
            latent_z = get_random_gaussian_prior(vae_model, device)
        else:
            latent_z = get_prior_from_diffusion(vae_model, diff_model, diff_mean_z, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            one_text = generate_from_latent_vector(
                model = vae_model.decoder,
                tokenizer = decoder_tokenizer,
                context_tokens = context_tokens,
                latent_vector = latent_z,
                device = device,
                **generation_settings,
            )

        if valid_vals is not None and check_value_valid(one_text, valid_vals) is False:
            one_text = None
        elif required_cols is not None and check_required_columns(one_text, required_cols) is False:
            one_text = None

    return {
        "gen_idx": gen_idx,
        "text": one_text,
        "gen_times": sample_times,
    }


class MultiProcessingHelper:
    def multi_run(self, num_generations, output_file, func, num_workers=8, init_fn=None, init_args=None):
        pool = mp.Pool(num_workers, initializer=init_fn, initargs=init_args)
        pbar = tqdm(pool.imap(func, range(num_generations)), total=num_generations, dynamic_ncols=True)
        with open(output_file, "wt") as wf:
            for gen_res in pbar:
                wf.write(json.dumps(gen_res, ensure_ascii=False) + "\n")
        pbar.close()
        pool.close()
        pool.join()


def run_generation(args):
    generation_settings = {
        "max_new_tokens": 512,
        "temperature": args.temperature,
    }

    data_args = get_vae_data_args(args.model_path)
    data_dir = Path(data_args["data_path"]).parent

    if args.check_column:
        with open(os.path.join(data_dir, "info.json"), "r") as rf:
            info = json.load(rf)
        column_names = info["column_names"]
    else:
        column_names = None

    if args.check_value:
        with open(os.path.join(data_dir, "info.json"), "r") as rf:
            info = json.load(rf)
        valid_val = info["column_info"]
    else:
        valid_val = None

    if args.do_train_sampling:
        train_samples = load_data_samples(data_dir)["train"]
    else:
        train_samples = None

    with open(data_args["data_path"], "r") as rf:
        num_reals = sum(1 for _ in rf)
    gen_num = args.size_limit or num_reals

    num_gpu = torch.cuda.device_count()
    logger.info(f"Generating with {num_gpu} GPUs -> {args.output_file}.")
    logger.info(f"Filtering samples with target columns -> {column_names}")
    logger.info(f"Filtering samples with valid values -> {valid_val}")

    gen_func = partial(
        generate_one_random_sample,
        generation_settings=generation_settings,
        train_samples=train_samples,
        required_cols=column_names,
        valid_vals=valid_val,
    )

    worker = MultiProcessingHelper()
    worker.multi_run(
        gen_num,
        args.output_file,
        func=gen_func,
        num_workers=num_gpu,
        init_fn=init_model,
        init_args=(args.model_path, args.do_diffusion, num_gpu, args.diffusion_name),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script for argparse")

    parser.add_argument("--model_path", type=str, required=True, help="The VAE model path")
    parser.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature for generation")
    parser.add_argument("--size_limit", type=int, default=None, help="Maximal synthetic data samples to be generated. Use real_nums if leaves None.")
    parser.add_argument("--check_column", nargs="?", const=True, type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--check_value", nargs="?", const=True, type=lambda x: (str(x).lower() == "true"), default=False)

    parser.add_argument("--do_train_sampling", nargs="?", const=True, type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to use random train samples encoding as the random prior")
    parser.add_argument("--do_diffusion", nargs="?", const=True, type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to use diffusion model to generate the random prior")
    parser.add_argument("--do_eval", nargs="?", const=True, type=lambda x: (str(x).lower() == "true"), default=False, help="Whether to run evaluations.")
    parser.add_argument("--diffusion_name", type=str, default="attn10", help="The name of denoising network for diffusion model")

    parser.add_argument("--output_dir", type=str, default="./output/generations", help="Output file path")
    parser.add_argument("--result_dir", type=str, default="./output/evaluations", help="Evaluation result file path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    model_name = Path(args.model_path).parts[-1]
    if model_name.startswith("checkpoint-"):
        model_name = Path(args.model_path).parts[-2] + "-" + model_name
    if args.do_train_sampling:
        model_name += "-train-sampling"
    elif args.do_diffusion:
        model_name += "-diffused-" + args.diffusion_name
    out_fname = f"{model_name}-temp{args.temperature}"
    args.output_file = os.path.join(args.output_dir, out_fname+".jsonl")
    args.result_file = os.path.join(args.result_dir, out_fname+".json")

    run_generation(args)
    run_evaluation(args)

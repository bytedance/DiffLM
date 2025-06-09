# DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models

This repository contains the code and models for our paper 
[DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](https://arxiv.org/abs/2411.03250).

## News

- **[2025.05]** Initial release.

## Contents

- [DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](#difflm-controllable-synthetic-data-generation-via-diffusion-language-models)
  - [News](#news)
  - [Contents](#contents)
  - [Environment Preparation](#environment-preparation)
    - [Prepare Datasets](#prepare-datasets)
    - [(OR) Download Datasets](#or-download-datasets)
  - [Training](#training)
    - [Step 1. Train VAE Model](#step-1-train-vae-model)
    - [Step 2. Train Latent Diffusion Model](#step-2-train-latent-diffusion-model)
    - [Step 3. Generate Synthetic Data](#step-3-generate-synthetic-data)
  - [Citation](#citation)
  - [License](#license)


## Environment Preparation

- Python version: 3.10
- Torch version: 2.3.1

The dependencies are listed in [requirements.txt](requirements.txt). To install all the requirements, run:

```shell
pip install -r requirements.txt
```

### Prepare Datasets
```shell
python data/prepare_tabular_data.py
python data/prepare_code_data.py
python data/prepare_tool_data.py
```

You can create your own dataset using the pipeline above.

### (OR) Download Datasets
To use the datasets referenced in the paper, download and extract the zipped datasets from []().

Note: As the datasets are currently under review, please use the scripts above to generate them for now.

## Training

### Step 1. Train VAE Model

```shell
# Modify these arguments in training scripts
export DIST_NPROC_PER_NODE=8        # 8 GPUs per node
export DIST_NNODES=1                # 1 node for training on single machine
export DIST_NODE_RANK=0             # Rank of the current node (starting from 0)
export DIST_MASTER_ADDR=127.0.0.1   # IP address for the master node
export DIST_MASTER_PORT=29500       # Port for the master node

bash scripts/run_vae_training.sh
```

### Step 2. Train Latent Diffusion Model

After the VAE is trained, train the diffusion model with running the following scripts:

```shell
bash scripts/run_diffusion_training.py [PATH_OF_VAE_MODEL]
```

### Step 3. Generate Synthetic Data
```shell
bash scripts/run_generating.py [PATH_OF_VAE_MODEL]
```

## Citation

```text
@inproceedings{difflm,
    title={{DiffLM}: DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models},
    author={Ying Zhou, Xinyao Wang, Yulei Niu, Yaojie Shen, Lexin Tang, Fan Chen, Ben He, Le Sun and Longyin Wen},
    booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
    year={2025}
}
```

## License

[![Code License](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-red)](WEIGHT_LICENSE)

The weights of checkpoints are licensed under CC BY-NC 4.0 for non-commercial use. The codebase is licensed under MIT.
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses.
Users must comply with all terms and conditions of these original licenses.
The content produced by any version of DiffLM is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project.
This project does not accept any legal liability for the content of the model output, nor does it assume responsibility for any losses incurred due to the use of associated resources and output results.
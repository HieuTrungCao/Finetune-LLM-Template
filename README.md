<div align="center">

# Finetuning-LLM-Template

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)
[![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template) <br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
[![contributors](https://img.shields.io/github/contributors/ashleve/lightning-hydra-template.svg)](https://github.com/ashleve/lightning-hydra-template/graphs/contributors)

A clean template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/ashleve/lightning-hydra-template/generate) to initialize new repository.

_Suggestions are always welcome!_

</div>

<br>

## 📌  Introduction

**Why you might want to use it:**

✅ Save on boilerplate <br>
Easily add new models, datasets, tasks, experiments, and train on different accelerators, like multi-GPU, TPU or SLURM clusters.

✅ Education <br>
Thoroughly commented. You can use this repo as a learning resource.

✅ Reusability <br>
Collection of useful MLOps tools, configs, and code snippets. You can use this repo as a reference for various utilities.

**Why you might not want to use it:**

❌ Things break from time to time <br>
Lightning and Hydra are still evolving and integrate many libraries, which means sometimes things break. For the list of currently known problems visit [this page](https://github.com/ashleve/lightning-hydra-template/labels/bug).

❌ Not adjusted for data engineering <br>
Template is not really adjusted for building data pipelines that depend on each other. It's more efficient to use it for model prototyping on ready-to-use data.

❌ Overfitted to simple use case <br>
The configuration setup is built with simple lightning training in mind. You might need to put some effort to adjust it for different use cases, e.g. lightning fabric.

❌ Might not support your workflow <br>
For example, you can't resume hydra-based multirun or hyperparameter search.

> **Note**: _Keep in mind this is unofficial community project._

<br>

## Main Technologies

[Huggingface](https://huggingface.co/) - a central platform for machine learning, providing access to pre-trained models, datasets, and tools, particularly for natural language processing. It fosters a collaborative community, aiming to democratize AI by simplifying the development and deployment of machine learning applications.

[Transformers](https://huggingface.co/docs/transformers/index) - a lib provides APIs and tools to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you the time and resources required to train a model from scratch. These models support common tasks in different modalities.

[LoRA](https://huggingface.co/docs/diffusers/training/lora) - a popular and lightweight training technique that significantly reduces the number of trainable parameters. It works by inserting a smaller number of new weights into the model and only these are trained. This makes training with LoRA much faster, memory-efficient, and produces smaller model weights (a few hundred MBs), which are easier to store and share. LoRA can also be combined with other training techniques like DreamBooth to speedup training.

[PEFT](https://huggingface.co/docs/peft/index#peft) - a library for efficiently adapting large pretrained models to various downstream applications without fine-tuning all of a model’s parameters because it is prohibitively costly. PEFT methods only fine-tune a small number of (extra) model parameters - significantly decreasing computational and storage costs - while yielding performance comparable to a fully fine-tuned model. This makes it more accessible to train and store large language models (LLMs) on consumer hardware.
<br>

## Project Structure

The directory structure of new project looks like this:

```
├── config                  <- Hydra configs
│   ├── data                  <- Data configs
│   ├── training_arg          <- Training Args configs 
│   ├── log                   <- Log configs
│   ├── lora                  <- Lora configs
│   ├── model                 <- Model configs
│   ├── qlora                 <- Qlora configs
│   ├── trainer               <- Trainer configs
│   ├── infer.yaml
│   └── finetune.yaml
├── data                    <- Project data
├── outputs                 <- Logs generated by hydra and lightning loggers
├── src                     <- Source code
│   ├── callback              <- Callback scripts
│   ├── data                  <- Data scripts
│   ├── metric                <- Metric scripts
│   ├── model                 <- Model scripts
│   ├── utils                 <- Utility scripts
│   ├── finetune.py           <- Run finetuning
│   └── infer.py              <- Run evaluation
├── tests                   <- Tests of any kind
├── README.md
└── requirements.txt        <- File for installing python dependencies
```

<br>

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/HieuTrungCao/Finetune-LLM-Template.git
cd Finetune-LLM-Template

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Template contains example with Bitext-customer-support-llm-chatbot dataset.<br>
When running `python src/finetune.py` you should see something like this:

<div align="center">

![](https://github.com/ashleve/lightning-hydra-template/blob/resources/terminal.png)

</div>
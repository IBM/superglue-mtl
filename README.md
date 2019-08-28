# superglue-mtl

This is a multi-task learning framework for [SuperGlue](https://super.gluebenchmark.com/) benchmark. We mainly focus on designing multi-task training scheme and data augmentation techniques for  large pre-trained language models such as BERT, RoBERTa, XLNet. 
Currently, most of the models are adapted from [`Pytorch-Transformers`](https://github.com/huggingface/pytorch-transformers) maintained by Hugging Face.

**This repository is under consturction.**

## Quick Start

First please make sure to install the necessary packages:

```shell
pip install -r requirements.txt
```
Install `Pytorch-Transformers`: https://github.com/huggingface/pytorch-transformers#installation

Configure the environment variables to specify the data and experiment directories for checkpointing:

```shell
export SG_DATA=./data/superglue/
export EXP_DATA=./exp/
```

Run demo training on a small set of toy examples:

```shell
python run_main.py --demo \
    --tasks=BoolQ,CB,RTE \
    --do_train \
    --model_type=bert \
    --model_name=bert-base-uncased \
    --do_lower_case \
    --max_seq_length=128
    --output_dir=mtl-boolq-cb-rte_bert-base_max-seq-len-128_lr-1e-5 \
    --batch_size=16 \
    --logging_freq=2 \
    --warmup_steps=0 \
    --learning_rate=1e-5 \
    --max_grad_norm=5.0 \
    --num_train_epochs=15 \
```


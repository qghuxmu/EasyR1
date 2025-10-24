#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
export WANDB_API_KEY="4039be8f46d3f1bf00274658db376a4dee152cd4"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=limazhiluyao/MMK12@train \
    data.val_files=limazhiluyao/MMK12@test \
    data.prompt_key=question \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=6 \
    trainer.experiment_name=qwen2_5_vl_7b_mmk12_grpo_neg_samples3_coef0_randomimg \
    trainer.n_gpus_per_node=4 \
    trainer.save_limit=-1 \
    algorithm.adv_estimator=grpo_neg \
    algorithm.rollout_negative_samples=4 \
    algorithm.penalty_coef=0.0 \
    algorithm.apply_penalty_mask=True \
    algorithm.negatives_type=random_img

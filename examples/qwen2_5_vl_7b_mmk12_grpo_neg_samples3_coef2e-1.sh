#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=limazhiluyao/MMK12@train \
    data.val_files=limazhiluyao/MMK12@test \
    data.prompt_key=question \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.n=6 \
    trainer.experiment_name=qwen2_5_vl_7b_mmk12_grpo_neg_samples3_coef2e-1 \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=/mnt/blob-hptrainingwesteurope-pretraining-out/qingguo/easyr1/qwen2_5_vl_7b_mmk12_grpo_neg_samples3_coef2e-1 \
    trainer.load_checkpoint_path=/mnt/blob-hptrainingwesteurope-pretraining-out/qingguo/easyr1/qwen2_5_vl_7b_mmk12_grpo_neg_samples3_coef2e-1 \
    trainer.save_limit=-1 \
    algorithm.adv_estimator=grpo_neg \
    algorithm.rollout_negative_samples=3 \
    algorithm.penalty_coef=0.2

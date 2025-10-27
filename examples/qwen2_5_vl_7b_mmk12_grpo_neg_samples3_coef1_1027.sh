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
    trainer.experiment_name=qwen2_5_vl_7b_mmk12_grpo_neg_samples3_coef1_1027 \
    trainer.n_gpus_per_node=8 \
    trainer.save_limit=-1 \
    trainer.save_freq=25 \
    algorithm.adv_estimator=grpo_neg \
    algorithm.rollout_negative_samples=3 \
    algorithm.penalty_coef=1.0 \
    algorithm.apply_penalty_mask=True \
    algorithm.negatives_type=text_only

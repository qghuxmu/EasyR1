#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=limazhiluyao/MMK12@train \
    data.val_files=limazhiluyao/MMK12@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_mmk12_grpo \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=/mnt/blob-hptrainingwesteurope-pretraining-out/qingguo/easyr1/qwen2_5_vl_7b_mmk12_grpo \
    trainer.load_checkpoint_path=/mnt/blob-hptrainingwesteurope-pretraining-out/qingguo/easyr1/qwen2_5_vl_7b_mmk12_grpo

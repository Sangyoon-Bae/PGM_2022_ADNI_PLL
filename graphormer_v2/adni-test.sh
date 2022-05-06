#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

mkdir ./ckpts/adni-struct-countbest/
mv ./ckpts/aadni-struct-count/checkpoint_best.pt ./ckpts/adni-struct-count/best/checkpoint_best.pt

n_gpu=1
epoch=4
max_epoch=$((epoch + 1))
batch_size=128
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates*16/100))
CUDA_VISIBLE_DEVICES=1

python ./graphormer/evaluate/evaluate.py \
    --user-dir graphormer \
    --num-workers 16 \
    --ddp-backend=legacy_ddp \
    --dataset-name adni-struct-count \
    --dataset-source ogb \
    --task graph_prediction \
    --arch graphormer_slim \
    --num-classes 3 \
    --batch-size 64 \
    --data-buffer-size 20 \
    --save-dir ./ckpts/adni-struct-count/best \
    --split test \
    --metric auc \
    --seed 1 \
    --pre-layernorm
#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

n_gpu=1
epoch=4
max_epoch=$((epoch + 1))
batch_size=128
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates*16/100))

rm -rf ./ckpts/adni-struct-count
mkdir ./ckpts/adni-struct-count
# --best-checkpoint-metric loss \


CUDA_VISIBLE_DEVICES=2 fairseq-train \
--user-dir graphormer \
--num-workers 4 \
--ddp-backend=legacy_ddp \
--dataset-name adni-struct-count \
--dataset-source ogb \
--task graph_prediction \
--criterion auc \
--arch graphormer_slim \
--num-classes 3 \
--attention-dropout 0.1 --act-dropout 0.1 --dropout 0.1 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.0 \
--lr-scheduler polynomial_decay --power 1 --warmup-updates $warmup_updates --total-num-update $tot_updates \
--lr 2e-4 --end-learning-rate 1e-5 \
--batch-size $batch_size \
--fp16 \
--data-buffer-size 20 \
--max-epoch 1000 \
--seed 1 \
--pre-layernorm \
--save-dir ./ckpts/adni-struct-count
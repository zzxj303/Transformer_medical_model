#!/bin/bash -l

set -euo pipefail



SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/dev${PYTHONPATH:+:${PYTHONPATH}}"

data_prefix="./testdata/nonimg_smoke"
data_path="${data_prefix}/smoke_all.csv"
train_path="${data_prefix}/smoke_train.csv"
vld_path="${data_prefix}/smoke_vld.csv"
test_path="${data_prefix}/smoke_test.csv"
cnf_file="./dev/data/toml_files/default_conf_new.toml"

img_net="NonImg"
img_mode=-1
ckpt_path="./dev/ckpt/debug/model.pt"

python dev/train.py \
    --data_path "${data_path}" \
    --train_path "${train_path}" \
    --vld_path "${vld_path}" \
    --test_path "${test_path}" \
    --cnf_file "${cnf_file}" \
    --d_model 256 \
    --nhead 1 \
    --num_epochs 256 \
    --batch_size 32 \
    --lr 0.001 \
    --gamma 0 \
    --img_mode "${img_mode}" \
    --img_net "${img_net}" \
    --weight_decay 0.0005 \
    --ranking_loss \
    --save_intermediate_ckpts \
    --ckpt_path "${ckpt_path}"

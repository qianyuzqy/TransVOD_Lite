#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/our_models/exps_multi/swint_83.7
PY_ARGS=${@:1}
python -u main.py \
    --eval \
    --backbone swin_t_p4w7 \
    --epochs 7 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --dilation \
    --batch_size 1 \
    --num_frames 15 \
    --hidden_dim 256 \
    --lr_drop_epochs 5 6 \
    --num_workers 1 \
    --with_box_refine \
    --dataset_file 'vid_multi_eval' \
    --gap 1 \
    --is_shuffle \
    --resume ${EXP_DIR}/checkpoint0006.pth \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval_is_shuffle.$T

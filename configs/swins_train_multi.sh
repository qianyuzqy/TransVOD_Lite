#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps_small/train
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_s_p4w7 \
    --epochs 7 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --dilation \
    --batch_size 1 \
    --num_frames 15 \
    --hidden_dim 128 \
    --lr_drop_epochs 5 6 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file 'vid_multi' \
    --resume exps/our_models/exps_single/swins_84.7/checkpoint0005.pth \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T.txt
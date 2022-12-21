#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/our_models/exps_single/swinb
mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --eval \
    --backbone swin_b_p4w7 \
    --epochs 7 \
    --lr_drop_epochs 5 6 \
    --num_feature_levels 1\
    --num_queries 100 \
    --dilation \
    --batch_size 2 \
    --hidden_dim 256 \
    --num_workers 1 \
    --with_box_refine \
    --resume ${EXP_DIR}/checkpoint0006.pth \
    --dataset_file 'vid_single' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.eval_e7.$T.txt

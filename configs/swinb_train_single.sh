#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/singlebaseline_swin_384_level1024/swin_e7_ld56_nf1_dim256_wbox_dc5_bs2_numquery_100
mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 7 \
    --lr_drop_epochs 5 6 \
    --num_feature_levels 1\
    --num_queries 100 \
    --dilation \
    --batch_size 2 \
    --hidden_dim 256 \
    --num_workers 8 \
    --with_box_refine \
    --resume ./exps/our_models/COCO_pretrained_model/swinb_checkpoint0048.pth \
    --coco_pretrain \
    --dataset_file 'vid_single' \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T

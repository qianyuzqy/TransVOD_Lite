#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps_base/multibaseline_topk80_50_30_tdtdloss_sgl_88.3/swinb_multi_numframe12_dim256_agg256_tmp/e7_nf1_ld5,6_lr0.0002_nq100_wbox_MEGA_detrNorm_preSingle_dc5
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone swin_b_p4w7 \
    --epochs 7 \
    --num_feature_levels 1 \
    --num_queries 100 \
    --dilation \
    --batch_size 1 \
    --num_frames 12 \
    --hidden_dim 256 \
    --lr_drop_epochs 5 6 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file 'vid_multi' \
    --resume exps/our_models/exps_single/swinb_88.3/checkpoint0006.pth \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T

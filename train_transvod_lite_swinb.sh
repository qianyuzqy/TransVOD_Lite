#!/bin/bash
GPUS_PER_NODE=1 ./tools/run_dist_launch.sh $1 swinb_train $2 configs/swinb_train_multi.sh

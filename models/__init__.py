# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .deformable_detr_single import build as build_single
from .deformable_detr_multi import build as build_multi


def build_model(args):
    if args.dataset_file == "vid_multi":
        return build_multi(args)
    elif args.dataset_file == "vid_multi_eval":
        return build_multi(args)
    else: # args.dataset_file == "vid_single":
        return build_single(args)



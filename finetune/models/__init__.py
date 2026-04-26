# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from . import vision_transformer as vits
from . import trans_classification as trans25d
import torch

def build_backbone_model(args, only_teacher=True, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim

def build_backbone_model_from_cfg(cfg, only_teacher=True):
    return build_backbone_model(cfg.model, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)

def build_base25d_avg_model_from_cfg(cfg, only_teacher=True):
    """
    Backward-compatible alias for older trainer imports.
    The public release currently ships the Trans25D classification model builder.
    """
    return build_trans25d_classification_model_from_cfg(cfg, only_teacher=only_teacher)

def build_trans25d_classification_model_from_cfg(cfg, only_teacher=True):
    backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    model = trans25d.Trans25D_Classification(
        backbone_model,
        embed_dim,
        cfg.model.use_n_blocks,
        cfg.model.use_avgpool,
        cfg.model.num_classes,
        cfg.model.num_decoder_layers,
        cfg.model.trans_nhead,
        cfg.model.trans_dim_feedforward_ratio,
    )
    return model, embed_dim

def build_visualize_trans25d_classification_model_from_cfg(cfg, only_teacher=True):
    backbone_model, embed_dim = build_backbone_model_from_cfg(cfg, only_teacher=only_teacher)
    model = trans25d.VisualizeTrans25D_Classification(backbone_model, embed_dim, cfg.model.use_n_blocks, cfg.model.use_avgpool, cfg.model.num_classes, cfg.model.num_decoder_layers, cfg.model.trans_nhead, cfg.model.trans_dim_feedforward_ratio)
    return model, embed_dim

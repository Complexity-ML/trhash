"""Translate a detector checkpoint configuration into trainer CLI options."""

from __future__ import annotations


def architecture_arguments(config) -> list[str]:
    arguments = [
        "--architecture-version",
        str(config.architecture_version),
        "--neck-mode",
        str(config.neck_mode),
        "--image-size",
        str(config.image_size),
        "--patch-size",
        str(config.patch_size),
        "--vision-hidden-size",
        str(config.vision_hidden_size),
        "--vision-layers",
        str(config.vision_layers),
        "--vision-heads",
        str(config.vision_heads),
        "--vision-num-experts",
        str(config.vision_num_experts),
        "--vision-top-k",
        str(config.vision_top_k),
        "--vision-expert-width",
        str(config.vision_expert_width),
        "--assignment-top-k",
        str(config.assignment_top_k),
        "--reg-max",
        str(config.reg_max),
        "--head-hidden-size",
        str(config.head_hidden_size),
        "--dfl-loss-weight",
        str(config.dfl_loss_weight),
        "--quality-focal-beta",
        str(config.quality_focal_beta),
        "--box-loss-weight",
        str(config.box_loss_weight),
        "--quality-loss-weight",
        str(config.quality_loss_weight),
        "--box-l1-weight",
        str(config.box_l1_weight),
        "--box-iou-weight",
        str(config.box_iou_weight),
    ]
    flags = (
        (not config.multi_scale, "--single-scale"),
        (config.p2_head, "--p2-head"),
        (not config.dynamic_assignment, "--static-assignment"),
        (not config.stal_enabled, "--no-stal"),
        (not config.progressive_loss_enabled, "--no-progressive-loss"),
    )
    arguments.extend(flag for enabled, flag in flags if enabled)
    return arguments

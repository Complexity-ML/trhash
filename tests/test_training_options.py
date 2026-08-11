from types import SimpleNamespace

from trhash.training_options import architecture_arguments


def test_checkpoint_architecture_is_forwarded_to_sft():
    config = SimpleNamespace(
        image_size=224,
        patch_size=8,
        vision_hidden_size=128,
        vision_layers=4,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=48,
        assignment_top_k=5,
        one_to_one_loss_weight=1.0,
        box_loss_weight=5.0,
        objectness_loss_weight=1.0,
        class_loss_weight=1.0,
        box_l1_weight=0.25,
        box_iou_weight=1.0,
        focal_alpha=0.5,
        focal_gamma=2.0,
        objectness_loss_type="varifocal",
        varifocal_alpha=0.75,
        varifocal_gamma=2.0,
        multi_scale=True,
        p2_head=True,
        dynamic_assignment=True,
        stal_enabled=True,
        progressive_loss_enabled=True,
        end_to_end=True,
    )

    arguments = architecture_arguments(config)

    assert arguments[arguments.index("--image-size") + 1] == "224"
    assert arguments[arguments.index("--vision-num-experts") + 1] == "4"
    assert "--p2-head" in arguments
    assert "--no-end-to-end" not in arguments

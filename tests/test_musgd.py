import torch

from trhash.optimizers import MuSGD, build_musgd_parameter_groups, named_learning_rates


class TinyVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.conv = torch.nn.Conv2d(2, 2, 1)


def _optimizer(model: torch.nn.Module) -> MuSGD:
    groups = build_musgd_parameter_groups(
        model,
        learning_rate=lambda name: (0.02, "experts") if "conv" in name else (0.01, "base"),
        momentum=0.9,
        weight_decay=0.001,
    )
    return MuSGD(groups, muon_weight=0.2, sgd_weight=1.0)


def test_musgd_routes_matrix_and_conv_weights_without_adamw_state():
    model = TinyVisionModel()
    optimizer = _optimizer(model)
    group_by_parameter = {
        id(parameter): group for group in optimizer.param_groups for parameter in group["params"]
    }

    assert group_by_parameter[id(model.projection.weight)]["use_muon"] is True
    assert group_by_parameter[id(model.conv.weight)]["use_muon"] is True
    assert group_by_parameter[id(model.projection.bias)]["use_muon"] is False
    assert group_by_parameter[id(model.norm.weight)]["use_muon"] is False
    assert group_by_parameter[id(model.norm.weight)]["weight_decay"] == 0.0
    assert named_learning_rates(optimizer) == {"base": 0.01, "experts": 0.02}

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert set(optimizer.state[model.projection.weight]) == {
        "muon_momentum",
        "sgd_momentum",
    }
    assert set(optimizer.state[model.projection.bias]) == {"momentum_buffer"}
    assert all("exp_avg" not in state for state in optimizer.state.values())
    assert all("exp_avg_sq" not in state for state in optimizer.state.values())


def test_musgd_state_dict_roundtrip_restores_both_momentum_components():
    source = TinyVisionModel()
    optimizer = _optimizer(source)
    for parameter in source.parameters():
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    restored = TinyVisionModel()
    restored_optimizer = _optimizer(restored)
    restored_optimizer.load_state_dict(optimizer.state_dict())

    source_state = optimizer.state[source.projection.weight]
    restored_state = restored_optimizer.state[restored.projection.weight]
    assert torch.equal(source_state["muon_momentum"], restored_state["muon_momentum"])
    assert torch.equal(source_state["sgd_momentum"], restored_state["sgd_momentum"])

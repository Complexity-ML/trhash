import sys
from types import ModuleType

from trhash.optimizers import MuSGD
from trhash.training_entrypoint import main


def test_training_entrypoint_injects_sdk_owned_musgd(monkeypatch):
    called = []
    training = ModuleType("complexity.generative.detection.training")
    training.main = lambda: called.append(True)
    detection = ModuleType("complexity.generative.detection")
    detection.training = training
    generative = ModuleType("complexity.generative")
    generative.detection = detection
    complexity = ModuleType("complexity")
    complexity.generative = generative
    monkeypatch.setitem(sys.modules, "complexity", complexity)
    monkeypatch.setitem(sys.modules, "complexity.generative", generative)
    monkeypatch.setitem(sys.modules, "complexity.generative.detection", detection)
    monkeypatch.setitem(
        sys.modules,
        "complexity.generative.detection.training",
        training,
    )

    main()

    assert called == [True]
    assert training.MuSGD is MuSGD
    assert training.build_musgd_parameter_groups.__module__ == "trhash.optimizers.musgd"
    assert training.named_learning_rates.__module__ == "trhash.optimizers.musgd"

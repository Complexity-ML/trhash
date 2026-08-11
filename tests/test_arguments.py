import pytest

from trhash.arguments import assignments, optional_bool, reject_unknown, require


def test_key_value_arguments():
    values = assignments(["model=a/b", "expert-lr-multiplier=1.5"])

    assert require(values, "model") == "a/b"
    assert values == {"expert_lr_multiplier": "1.5"}


def test_argument_errors_are_explicit():
    with pytest.raises(ValueError, match="expected key=value"):
        assignments(["model"])
    with pytest.raises(ValueError, match="duplicate"):
        assignments(["model=a", "model=b"])
    with pytest.raises(ValueError, match="unknown"):
        reject_unknown({"wat": "1"})


def test_boolean_argument():
    assert optional_bool({"save": "yes"}, "save")
    assert not optional_bool({}, "save")

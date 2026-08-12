"""TR-Hash fine-tuning process entry point.

The SDK owns the optimizer contract. The optional research adapter supplies
the detector architecture and data loop until those components are available
as a native checkpoint runtime.
"""

from __future__ import annotations

from .optimizers import MuSGD, build_musgd_parameter_groups, named_learning_rates


def main() -> None:
    try:
        from complexity.generative.detection import training as detector_training
    except ImportError as error:
        raise RuntimeError(
            'fine-tuning currently requires the optional `pip install "trhash[local]"` adapter'
        ) from error

    detector_training.MuSGD = MuSGD
    detector_training.build_musgd_parameter_groups = build_musgd_parameter_groups
    detector_training.named_learning_rates = named_learning_rates
    detector_training.main()


if __name__ == "__main__":
    main()

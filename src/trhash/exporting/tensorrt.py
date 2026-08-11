"""TensorRT engine export from the parity-checked ONNX graph."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ..metadata import metadata_from_checkpoint
from .common import parity_inputs, prepare_export, tensor_outputs
from .onnx import export_onnx


def _tensorrt_major_version(trt) -> int:
    try:
        return int(str(trt.__version__).split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"unsupported TensorRT version: {trt.__version__}") from error


def _prepare_fp16_onnx(onnx_path: Path, temporary: Path) -> Path:
    try:
        import onnx
        from modelopt.onnx.autocast import convert_to_mixed_precision
    except ImportError as error:
        raise RuntimeError(
            'TensorRT 11 FP16 export requires `pip install "trhash[tensorrt]"` '
            "with NVIDIA ModelOpt"
        ) from error

    fp16_path = temporary / "model-fp16.onnx"
    converted = convert_to_mixed_precision(
        onnx_path=str(onnx_path),
        low_precision_type="fp16",
        keep_io_types=True,
    )
    onnx.save(converted, str(fp16_path))
    return fp16_path


def export_tensorrt(
    backend,
    *,
    output: Union[str, Path] = "runs/export",
    precision: str = "fp16",
    max_batch: int = 32,
    workspace_gb: float = 1.0,
    verify: bool = True,
) -> Path:
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError('TensorRT export requires `pip install "trhash[tensorrt]"`') from error
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires a CUDA GPU")
    if precision not in {"fp16", "fp32"}:
        raise ValueError("TensorRT precision must be fp16 or fp32")
    if max_batch < 1 or workspace_gb <= 0:
        raise ValueError("max_batch and workspace_gb must be positive")

    output_path, detector, example = prepare_export(backend, output)
    engine_path = output_path / "model.engine"
    with tempfile.TemporaryDirectory(prefix="trhash-tensorrt-") as temporary:
        onnx_bundle = export_onnx(
            backend,
            output=temporary,
            dynamic_batch=True,
            verify=False,
        )
        onnx_path = onnx_bundle / "model.onnx"
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        if precision == "fp16" and not builder.platform_has_fast_fp16:
            raise RuntimeError("the selected GPU does not support fast TensorRT FP16")
        trt_major = _tensorrt_major_version(trt)
        if precision == "fp16" and trt_major >= 11:
            onnx_path = _prepare_fp16_onnx(onnx_path, Path(temporary))

        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT failed to parse ONNX: {errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gb * (1 << 30)),
        )
        if precision == "fp16" and trt_major < 11:
            config.set_flag(trt.BuilderFlag.FP16)

        network_input = network.get_input(0)
        profile = builder.create_optimization_profile()
        image_shape = tuple(example.shape[1:])
        optimal_batch = min(8, max_batch)
        if not profile.set_shape(
            network_input.name,
            (1, *image_shape),
            (optimal_batch, *image_shape),
            (max_batch, *image_shape),
        ):
            raise RuntimeError("TensorRT rejected the dynamic batch optimization profile")
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("TensorRT could not add the optimization profile")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        engine_path.write_bytes(bytes(serialized))

    metadata_from_checkpoint(backend, model_file=engine_path.name).save(output_path)
    if verify:
        from ..backends.tensorrt import TensorRTBackend

        runtime = TensorRTBackend(output_path)
        tolerance = 3e-2 if precision == "fp16" else 1e-4
        with torch.inference_mode():
            for values in parity_inputs(example, dynamic_batch=max_batch >= 2):
                expected = tensor_outputs(detector(values))
                actual = tensor_outputs(runtime._predict_raw(values.numpy()))
                if len(actual) != len(expected):
                    raise AssertionError("TensorRT engine returned the wrong number of outputs")
                for actual_array, expected_tensor in zip(actual, expected):
                    np.testing.assert_allclose(
                        actual_array,
                        expected_tensor.detach().cpu().numpy(),
                        atol=tolerance,
                        rtol=tolerance,
                    )
    return output_path

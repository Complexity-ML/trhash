"""TensorRT runtime using PyTorch CUDA tensors as device buffers."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..bundle import resolve_bundle
from ..metadata import ModelMetadata
from .portable import PortableDetectionBackend


class TensorRTBackend(PortableDetectionBackend):
    def __init__(
        self,
        model: Union[str, Path],
        *,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as error:
            raise RuntimeError('TensorRT inference requires `pip install "trhash[tensorrt]"`') from error
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires a CUDA GPU")
        self.trt = trt
        self.torch = torch
        self.device = torch.device(device or "cuda")
        if self.device.type != "cuda":
            raise ValueError("TensorRT device must be cuda or cuda:N")
        self.model_id = str(model)
        self.bundle = resolve_bundle(model, revision=revision, token=token)
        self.metadata = ModelMetadata.load(self.bundle)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            (self.bundle / self.metadata.model_file).read_bytes()
        )
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize this engine on the current GPU")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self._lock = threading.Lock()
        names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        inputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        outputs = [name for name in names if name not in inputs]
        if len(inputs) != 1 or set(outputs) != set(self.metadata.output_names):
            raise ValueError("TensorRT graph inputs/outputs do not match bundle metadata")
        self.input_name = inputs[0]
        self.output_names = self.metadata.output_names
        self.names = self.metadata.class_names
        self.providers = [f"TensorRT:{trt.__version__}:{self.device}"]

    def _torch_dtype(self, tensor_name: str):
        numpy_dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(tensor_name)))
        choices = {
            np.dtype(np.float16): self.torch.float16,
            np.dtype(np.float32): self.torch.float32,
            np.dtype(np.int32): self.torch.int32,
            np.dtype(np.int64): self.torch.int64,
        }
        try:
            return choices[numpy_dtype]
        except KeyError as error:
            raise TypeError(f"unsupported TensorRT tensor dtype: {numpy_dtype}") from error

    def _predict_raw(self, pixels: np.ndarray):
        torch = self.torch
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            input_tensor = torch.from_numpy(np.ascontiguousarray(pixels)).to(
                device=self.device,
                dtype=self._torch_dtype(self.input_name),
            )
            if not self.context.set_input_shape(self.input_name, tuple(input_tensor.shape)):
                raise ValueError(f"TensorRT rejected input shape {tuple(input_tensor.shape)}")
            if not self.context.set_tensor_address(self.input_name, input_tensor.data_ptr()):
                raise RuntimeError("TensorRT rejected the input tensor address")
            output_tensors = []
            for output_name in self.output_names:
                output_shape = tuple(self.context.get_tensor_shape(output_name))
                if any(dimension < 0 for dimension in output_shape):
                    raise RuntimeError(
                        f"TensorRT returned unresolved output shape {output_shape}"
                    )
                output_tensor = torch.empty(
                    output_shape,
                    dtype=self._torch_dtype(output_name),
                    device=self.device,
                )
                if not self.context.set_tensor_address(
                    output_name, output_tensor.data_ptr()
                ):
                    raise RuntimeError("TensorRT rejected an output tensor address")
                output_tensors.append(output_tensor)
            stream = torch.cuda.current_stream(self.device)
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT inference execution failed")
            stream.synchronize()
            outputs = tuple(output.cpu().numpy() for output in output_tensors)
            return outputs[0] if len(outputs) == 1 else outputs

    def serve(self, **options) -> None:
        from ..server.runner import run_server

        run_server(self.model_id, device=str(self.device), **options)

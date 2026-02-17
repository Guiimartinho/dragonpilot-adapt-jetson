"""
TensorRT inference runner for Jetson AGX Xavier.

Builds TensorRT engines from ONNX models with FP16 precision for Volta tensor cores.
Supports engine caching to avoid rebuild on every boot.
Falls back gracefully to tinygrad if TensorRT is unavailable.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Engine cache directory on NVMe storage
ENGINE_CACHE_DIR = Path(os.getenv("TRT_ENGINE_CACHE", "/data/trt_engines"))

# Build configuration
MAX_WORKSPACE_SIZE = 1 << 30  # 1 GB workspace
FP16_ENABLED = True
DLA_CORE = -1  # -1 = GPU only, 0 or 1 = DLA core index


def _get_onnx_hash(onnx_path: str) -> str:
  """Compute hash of ONNX file for cache invalidation."""
  h = hashlib.sha256()
  with open(onnx_path, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
      h.update(chunk)
  return h.hexdigest()[:16]


def _engine_cache_path(onnx_path: str, dla_core: int = -1) -> Path:
  """Get cache path for a TensorRT engine built from an ONNX model."""
  name = Path(onnx_path).stem
  onnx_hash = _get_onnx_hash(onnx_path)
  suffix = f"_dla{dla_core}" if dla_core >= 0 else "_gpu"
  return ENGINE_CACHE_DIR / f"{name}_{onnx_hash}{suffix}.trt"


class TensorRTRunner:
  """
  TensorRT inference runner with FP16 support and engine caching.

  Usage:
    runner = TensorRTRunner(onnx_path, fp16=True)
    outputs = runner.infer({"input_img": np_array, ...})
  """

  def __init__(self, onnx_path: str, fp16: bool = FP16_ENABLED,
               dla_core: int = DLA_CORE, max_workspace: int = MAX_WORKSPACE_SIZE):
    try:
      import tensorrt as trt
    except ImportError:
      raise ImportError(
        "tensorrt package not found. Install with: "
        "sudo apt-get install python3-libnvinfer python3-libnvinfer-dev"
      )

    self.trt = trt
    self.onnx_path = onnx_path
    self.fp16 = fp16
    self.dla_core = dla_core
    self.max_workspace = max_workspace

    self._logger = trt.Logger(trt.Logger.WARNING)
    self._runtime = trt.Runtime(self._logger)
    self._engine: Optional[trt.ICudaEngine] = None
    self._context: Optional[trt.IExecutionContext] = None

    # CUDA stream and device memory
    self._stream = None
    self._d_inputs: dict[str, int] = {}
    self._d_outputs: dict[str, int] = {}
    self._h_outputs: dict[str, np.ndarray] = {}
    self._bindings: list[int] = []

    self._build_or_load_engine()

  def _build_or_load_engine(self):
    """Load cached engine or build from ONNX."""
    import tensorrt as trt
    try:
      import pycuda.driver as cuda
      import pycuda.autoinit  # noqa: F401
    except ImportError:
      raise ImportError(
        "pycuda package not found. Install with: pip install pycuda"
      )

    cache_path = _engine_cache_path(self.onnx_path, self.dla_core)

    if cache_path.exists():
      logger.info("Loading cached TensorRT engine: %s", cache_path)
      with open(cache_path, "rb") as f:
        engine_data = f.read()
      self._engine = self._runtime.deserialize_cuda_engine(engine_data)
      if self._engine is not None:
        self._setup_context(cuda)
        return
      logger.warning("Failed to deserialize cached engine, rebuilding")

    logger.info("Building TensorRT engine from: %s (fp16=%s, dla=%d)",
                self.onnx_path, self.fp16, self.dla_core)
    self._engine = self._build_engine(trt)

    # Cache the engine
    ENGINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    serialized = self._engine.serialize()
    with open(cache_path, "wb") as f:
      f.write(serialized)
    logger.info("Cached TensorRT engine to: %s", cache_path)

    self._setup_context(cuda)

  def _build_engine(self, trt):
    """Build TensorRT engine from ONNX with FP16/DLA support."""
    builder = trt.Builder(self._logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, self._logger)

    with open(self.onnx_path, "rb") as f:
      if not parser.parse(f.read()):
        for i in range(parser.num_errors):
          logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError(f"Failed to parse ONNX model: {self.onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace)

    # FP16 mode for Volta tensor cores (sm_72)
    if self.fp16 and builder.platform_has_fast_fp16:
      config.set_flag(trt.BuilderFlag.FP16)
      logger.info("FP16 mode enabled for Volta tensor cores")

    # DLA configuration
    if self.dla_core >= 0:
      if builder.num_DLA_cores > self.dla_core:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = self.dla_core
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        # DLA requires FP16 or INT8
        if not config.get_flag(trt.BuilderFlag.FP16):
          config.set_flag(trt.BuilderFlag.FP16)
        logger.info("DLA core %d enabled with GPU fallback", self.dla_core)
      else:
        logger.warning("DLA core %d not available (%d cores found), using GPU",
                       self.dla_core, builder.num_DLA_cores)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
      raise RuntimeError("TensorRT engine build failed")
    return self._runtime.deserialize_cuda_engine(engine)

  def _setup_context(self, cuda):
    """Allocate device memory and create execution context."""
    self._stream = cuda.Stream()
    self._context = self._engine.create_execution_context()

    num_io = self._engine.num_io_tensors
    self._bindings = [0] * num_io

    for i in range(num_io):
      name = self._engine.get_tensor_name(i)
      shape = self._engine.get_tensor_shape(name)
      dtype = self._engine.get_tensor_dtype(name)
      np_dtype = self._trt_dtype_to_numpy(dtype)
      size = int(np.prod(shape)) * np.dtype(np_dtype).itemsize

      if self._engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
        d_mem = cuda.mem_alloc(size)
        self._d_inputs[name] = d_mem
        self._bindings[i] = int(d_mem)
        self._context.set_tensor_address(name, int(d_mem))
      else:
        d_mem = cuda.mem_alloc(size)
        h_mem = np.empty(tuple(shape), dtype=np_dtype)
        self._d_outputs[name] = d_mem
        self._h_outputs[name] = h_mem
        self._bindings[i] = int(d_mem)
        self._context.set_tensor_address(name, int(d_mem))

    logger.info("TensorRT context ready: %d inputs, %d outputs",
                len(self._d_inputs), len(self._d_outputs))

  def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Run inference with the TensorRT engine.

    Args:
      inputs: Dictionary mapping input tensor names to numpy arrays.

    Returns:
      Dictionary mapping output tensor names to numpy arrays.
    """
    import pycuda.driver as cuda

    # Copy inputs to device
    for name, arr in inputs.items():
      if name not in self._d_inputs:
        continue
      cuda.memcpy_htod_async(self._d_inputs[name], np.ascontiguousarray(arr), self._stream)

    # Execute
    self._context.execute_async_v3(stream_handle=self._stream.handle)

    # Copy outputs back
    results = {}
    for name, d_mem in self._d_outputs.items():
      cuda.memcpy_dtoh_async(self._h_outputs[name], d_mem, self._stream)

    self._stream.synchronize()

    for name, h_mem in self._h_outputs.items():
      results[name] = h_mem.copy()

    return results

  def get_input_shapes(self) -> dict[str, tuple]:
    """Get the shapes of all input tensors."""
    shapes = {}
    for i in range(self._engine.num_io_tensors):
      name = self._engine.get_tensor_name(i)
      if self._engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
        shapes[name] = tuple(self._engine.get_tensor_shape(name))
    return shapes

  def get_output_shapes(self) -> dict[str, tuple]:
    """Get the shapes of all output tensors."""
    shapes = {}
    for i in range(self._engine.num_io_tensors):
      name = self._engine.get_tensor_name(i)
      if self._engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT:
        shapes[name] = tuple(self._engine.get_tensor_shape(name))
    return shapes

  def _trt_dtype_to_numpy(self, trt_dtype) -> np.dtype:
    """Convert TensorRT dtype to numpy dtype."""
    import tensorrt as trt
    mapping = {
      trt.float32: np.float32,
      trt.float16: np.float16,
      trt.int8: np.int8,
      trt.int32: np.int32,
      trt.bool: np.bool_,
    }
    if hasattr(trt, 'uint8'):
      mapping[trt.uint8] = np.uint8
    return mapping.get(trt_dtype, np.float32)

  def destroy(self):
    """Release all TensorRT and CUDA resources."""
    self._context = None
    self._engine = None
    self._d_inputs.clear()
    self._d_outputs.clear()
    self._h_outputs.clear()
    self._stream = None

  def __del__(self):
    self.destroy()


class TensorRTModelRunner:
  """
  High-level model runner that wraps TensorRTRunner for use in modeld/dmonitoringmodeld.

  Provides the same interface as tinygrad pickle-based runners:
    output = runner(**inputs)  ->  numpy array
  """

  def __init__(self, onnx_path: str, fp16: bool = True, dla_core: int = -1):
    self._runner = TensorRTRunner(onnx_path, fp16=fp16, dla_core=dla_core)
    self._output_name = None

  def __call__(self, **kwargs) -> np.ndarray:
    """Run inference, returning concatenated outputs as a flat numpy array."""
    # Convert any non-numpy inputs
    np_inputs = {}
    for name, val in kwargs.items():
      if hasattr(val, 'numpy'):
        np_inputs[name] = val.numpy()
      elif isinstance(val, np.ndarray):
        np_inputs[name] = val
      else:
        np_inputs[name] = np.asarray(val)

    outputs = self._runner.infer(np_inputs)

    # Return first output (models have single output tensor)
    if self._output_name is None:
      self._output_name = list(outputs.keys())[0]
    return outputs[self._output_name].flatten().astype(np.float32)

  def destroy(self):
    self._runner.destroy()

"""
TensorRT inference runner for Jetson AGX Xavier.

Uses ctypes to interface with trt_runtime.so (C++ bridge library)
that wraps the TensorRT C++ API directly, bypassing the Python 3.8-only
tensorrt package that is incompatible with our Python 3.11 venv.

Engine building uses trtexec CLI. Runtime inference uses the TRT C++ API.
"""
from __future__ import annotations

import os
import ctypes
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Engine cache directory on NVMe storage
ENGINE_CACHE_DIR = Path(os.getenv("TRT_ENGINE_CACHE", "/data/trt_engines"))

# Build configuration
FP16_ENABLED = True
DLA_CORE = -1  # -1 = GPU only, 0 or 1 = DLA core index
MAX_WORKSPACE_MB = 512  # 512 MB workspace (safe for 32GB Jetson with concurrent processes)

# TRT dtype code -> numpy dtype (must match trt_runtime.cpp IOTensor.dtype)
_TRT_DTYPE_MAP = {
  0: np.float32,   # kFLOAT
  1: np.float16,   # kHALF
  2: np.int8,      # kINT8
  3: np.int32,     # kINT32
  4: np.uint8,     # kUINT8
}

# Paths to search for the compiled C bridge library
_LIB_SEARCH_PATHS = [
  Path(__file__).parent / 'trt_runtime.so',
  Path('/data/openpilot/selfdrive/modeld/runners/trt_runtime.so'),
]

_lib: Optional[ctypes.CDLL] = None


def _load_lib() -> ctypes.CDLL:
  global _lib
  if _lib is not None:
    return _lib

  for p in _LIB_SEARCH_PATHS:
    if p.exists():
      _lib = ctypes.CDLL(str(p))
      _setup_prototypes(_lib)
      logger.info("Loaded trt_runtime.so from %s", p)
      return _lib

  raise RuntimeError(
    "trt_runtime.so not found. Build with:\n"
    "  g++ -shared -fPIC -O2 -o trt_runtime.so trt_runtime.cpp -lnvinfer -lcudart"
  )


def _setup_prototypes(lib: ctypes.CDLL):
  """Set ctypes function signatures for type safety."""
  lib.trt_load_engine.argtypes = [ctypes.c_char_p]
  lib.trt_load_engine.restype = ctypes.c_void_p

  lib.trt_num_io.argtypes = [ctypes.c_void_p]
  lib.trt_num_io.restype = ctypes.c_int

  lib.trt_tensor_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.trt_tensor_name.restype = ctypes.c_char_p

  lib.trt_tensor_is_input.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.trt_tensor_is_input.restype = ctypes.c_int

  lib.trt_tensor_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.trt_tensor_size.restype = ctypes.c_int64

  lib.trt_tensor_dtype.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.trt_tensor_dtype.restype = ctypes.c_int

  lib.trt_tensor_ndims.argtypes = [ctypes.c_void_p, ctypes.c_int]
  lib.trt_tensor_ndims.restype = ctypes.c_int

  lib.trt_tensor_dim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
  lib.trt_tensor_dim.restype = ctypes.c_int

  lib.trt_set_input.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64]
  lib.trt_set_input.restype = None

  lib.trt_execute.argtypes = [ctypes.c_void_p]
  lib.trt_execute.restype = None

  lib.trt_get_output.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64]
  lib.trt_get_output.restype = None

  lib.trt_sync.argtypes = [ctypes.c_void_p]
  lib.trt_sync.restype = None

  lib.trt_destroy.argtypes = [ctypes.c_void_p]
  lib.trt_destroy.restype = None

  lib.trt_build_engine.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int]
  lib.trt_build_engine.restype = ctypes.c_int


def _ensure_trt_compatible_onnx(onnx_path: str) -> str:
  """If ONNX model has LayerNormalization (unsupported in TRT 8.5), decompose it.
  Returns path to compatible ONNX (may be the original or a _trt.onnx variant)."""
  try:
    import onnx
    model = onnx.load(onnx_path)
    has_layernorm = any(n.op_type == 'LayerNormalization' for n in model.graph.node)
    if not has_layernorm:
      return onnx_path

    # Create decomposed version
    trt_onnx_path = onnx_path.replace('.onnx', '_trt.onnx')
    if os.path.exists(trt_onnx_path):
      # Check if it's up to date (same mtime as original or newer)
      if os.path.getmtime(trt_onnx_path) >= os.path.getmtime(onnx_path):
        logger.info("Using cached TRT-compatible ONNX: %s", trt_onnx_path)
        return trt_onnx_path

    logger.info("Decomposing LayerNormalization ops for TRT 8.5 compatibility")
    from openpilot.selfdrive.modeld.runners.onnx_layernorm_decompose import decompose_layernorm
    model = decompose_layernorm(model)
    onnx.save(model, trt_onnx_path)
    return trt_onnx_path
  except ImportError:
    logger.warning("onnx package not available, using original ONNX file")
    return onnx_path


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
  TensorRT inference runner using ctypes bridge to trt_runtime.so.

  Bypasses the Python 3.8-only tensorrt package by calling TRT C++ API
  through plain C functions exposed via ctypes.

  Usage:
    runner = TensorRTRunner(onnx_path, fp16=True)
    outputs = runner.infer({"input_img": np_array, ...})
  """

  def __init__(self, onnx_path: str, fp16: bool = FP16_ENABLED,
               dla_core: int = DLA_CORE, max_workspace_mb: int = MAX_WORKSPACE_MB):
    self._lib = _load_lib()
    self.onnx_path = _ensure_trt_compatible_onnx(onnx_path)
    self.fp16 = fp16
    self.dla_core = dla_core
    self.max_workspace_mb = max_workspace_mb
    self._ctx = None

    # IO tensor metadata
    self._inputs: dict[str, tuple] = {}   # name -> (index, shape, np_dtype, size_bytes)
    self._outputs: dict[str, tuple] = {}  # name -> (index, shape, np_dtype, size_bytes)
    self._h_outputs: dict[str, np.ndarray] = {}  # pre-allocated host output buffers

    self._build_or_load_engine()

  def _build_or_load_engine(self):
    """Load cached engine or build from ONNX via trtexec."""
    cache_path = _engine_cache_path(self.onnx_path, self.dla_core)

    if not cache_path.exists():
      logger.info("Building TRT engine from %s (fp16=%s, dla=%d)",
                  self.onnx_path, self.fp16, self.dla_core)
      ENGINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
      ret = self._lib.trt_build_engine(
        self.onnx_path.encode(),
        str(cache_path).encode(),
        1 if self.fp16 else 0,
        self.dla_core,
        self.max_workspace_mb
      )
      if ret != 0:
        raise RuntimeError(f"trtexec engine build failed (exit code {ret}) for {self.onnx_path}")
      logger.info("TRT engine cached to: %s", cache_path)
    else:
      logger.info("Loading cached TRT engine: %s", cache_path)

    self._ctx = self._lib.trt_load_engine(str(cache_path).encode())
    if not self._ctx:
      # Cache might be stale, try rebuilding
      logger.warning("Failed to load cached engine, rebuilding")
      cache_path.unlink(missing_ok=True)
      ret = self._lib.trt_build_engine(
        self.onnx_path.encode(),
        str(cache_path).encode(),
        1 if self.fp16 else 0,
        self.dla_core,
        self.max_workspace_mb
      )
      if ret != 0:
        raise RuntimeError(f"trtexec rebuild failed (exit code {ret})")
      self._ctx = self._lib.trt_load_engine(str(cache_path).encode())
      if not self._ctx:
        raise RuntimeError(f"Failed to load rebuilt TRT engine: {cache_path}")

    self._setup_io()

  def _setup_io(self):
    """Query engine IO tensors and pre-allocate output buffers."""
    num_io = self._lib.trt_num_io(self._ctx)
    for i in range(num_io):
      name = self._lib.trt_tensor_name(self._ctx, i).decode()
      is_input = self._lib.trt_tensor_is_input(self._ctx, i)
      ndims = self._lib.trt_tensor_ndims(self._ctx, i)
      shape = tuple(self._lib.trt_tensor_dim(self._ctx, i, d) for d in range(ndims))
      dtype_code = self._lib.trt_tensor_dtype(self._ctx, i)
      np_dtype = _TRT_DTYPE_MAP.get(dtype_code, np.float32)
      size_bytes = self._lib.trt_tensor_size(self._ctx, i)

      if is_input:
        self._inputs[name] = (i, shape, np_dtype, size_bytes)
        logger.debug("  input[%d] '%s': shape=%s dtype=%s (%d bytes)", i, name, shape, np_dtype, size_bytes)
      else:
        self._outputs[name] = (i, shape, np_dtype, size_bytes)
        self._h_outputs[name] = np.empty(shape, dtype=np_dtype)
        logger.debug("  output[%d] '%s': shape=%s dtype=%s (%d bytes)", i, name, shape, np_dtype, size_bytes)

    logger.info("TRT engine ready: %d inputs, %d outputs", len(self._inputs), len(self._outputs))

  def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run inference. Copies inputs to GPU, executes, copies outputs back."""
    # Upload inputs to device
    for name, arr in inputs.items():
      if name not in self._inputs:
        continue
      idx, shape, np_dtype, size_bytes = self._inputs[name]
      arr = np.ascontiguousarray(arr, dtype=np_dtype)
      self._lib.trt_set_input(self._ctx, idx, arr.ctypes.data, arr.nbytes)

    # Execute inference
    self._lib.trt_execute(self._ctx)

    # Download outputs from device
    for name, (idx, shape, np_dtype, size_bytes) in self._outputs.items():
      out = self._h_outputs[name]
      self._lib.trt_get_output(self._ctx, idx, out.ctypes.data, out.nbytes)

    # Synchronize stream
    self._lib.trt_sync(self._ctx)

    return {name: arr.copy() for name, arr in self._h_outputs.items()}

  def get_input_shapes(self) -> dict[str, tuple]:
    return {name: info[1] for name, info in self._inputs.items()}

  def get_output_shapes(self) -> dict[str, tuple]:
    return {name: info[1] for name, info in self._outputs.items()}

  def destroy(self):
    if self._ctx:
      self._lib.trt_destroy(self._ctx)
      self._ctx = None

  def __del__(self):
    self.destroy()


class TensorRTModelRunner:
  """
  High-level model runner wrapping TensorRTRunner for modeld/dmonitoringmodeld.

  Provides the same callable interface as tinygrad pickle-based runners:
    output = runner(**inputs)  ->  numpy array
  """

  def __init__(self, onnx_path: str, fp16: bool = True, dla_core: int = -1):
    self._runner = TensorRTRunner(onnx_path, fp16=fp16, dla_core=dla_core)
    self._output_name = None

  def __call__(self, **kwargs) -> np.ndarray:
    np_inputs = {}
    for name, val in kwargs.items():
      if hasattr(val, 'numpy'):
        np_inputs[name] = val.numpy()
      elif isinstance(val, np.ndarray):
        np_inputs[name] = val
      else:
        np_inputs[name] = np.asarray(val)

    outputs = self._runner.infer(np_inputs)

    if self._output_name is None:
      self._output_name = list(outputs.keys())[0]
    return outputs[self._output_name].flatten().astype(np.float32)

  def destroy(self):
    self._runner.destroy()

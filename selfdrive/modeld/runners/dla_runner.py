"""
NVDLA inference runner for Jetson AGX Xavier driver monitoring model.

Offloads the dmonitoring model to DLA (Deep Learning Accelerator) engine,
completely freeing the GPU for the driving vision model.

Xavier has 2 DLA cores. This uses DLA core 0 by default.
Requires TensorRT as the DLA backend.
"""

import logging
from pathlib import Path

import numpy as np

from openpilot.selfdrive.modeld.runners.tensorrt_runner import TensorRTRunner

logger = logging.getLogger(__name__)

# DLA configuration
DEFAULT_DLA_CORE = 0
FALLBACK_DLA_CORE = 1


class DLARunner:
  """
  NVDLA runner that offloads inference to Xavier's DLA engines.

  Uses TensorRT with DeviceType.DLA for execution.
  Falls back to DLA core 1 if core 0 fails, then to GPU.

  Usage:
    runner = DLARunner(onnx_path)
    output = runner.infer({"input_img": np_array, "calib": np_array})
  """

  def __init__(self, onnx_path: str, dla_core: int = DEFAULT_DLA_CORE, fp16: bool = True):
    self.onnx_path = onnx_path
    self._runner = None
    self._using_dla = False
    self._active_core = -1

    self._runner, self._using_dla, self._active_core = self._try_build(
      onnx_path, dla_core, fp16
    )

  def _try_build(self, onnx_path: str, dla_core: int, fp16: bool):
    """Try building DLA runner with fallback chain: DLA core 0 -> DLA core 1 -> GPU."""
    # Try requested DLA core
    try:
      runner = TensorRTRunner(onnx_path, fp16=fp16, dla_core=dla_core)
      logger.info("DLA runner initialized on DLA core %d", dla_core)
      return runner, True, dla_core
    except Exception as e:
      logger.warning("Failed to build on DLA core %d: %s", dla_core, e)

    # Try fallback DLA core
    fallback = FALLBACK_DLA_CORE if dla_core == DEFAULT_DLA_CORE else DEFAULT_DLA_CORE
    try:
      runner = TensorRTRunner(onnx_path, fp16=fp16, dla_core=fallback)
      logger.info("DLA runner initialized on fallback DLA core %d", fallback)
      return runner, True, fallback
    except Exception as e:
      logger.warning("Failed to build on DLA core %d: %s", fallback, e)

    # Fallback to GPU via TensorRT (still faster than tinygrad)
    logger.warning("DLA unavailable, falling back to TensorRT GPU")
    runner = TensorRTRunner(onnx_path, fp16=fp16, dla_core=-1)
    return runner, False, -1

  @property
  def is_using_dla(self) -> bool:
    """Whether inference is actually running on DLA hardware."""
    return self._using_dla

  @property
  def active_dla_core(self) -> int:
    """Active DLA core index, or -1 if using GPU."""
    return self._active_core

  def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run inference on DLA (or GPU fallback)."""
    return self._runner.infer(inputs)

  def destroy(self):
    """Release all resources."""
    if self._runner is not None:
      self._runner.destroy()
      self._runner = None


class DLAModelRunner:
  """
  High-level DLA model runner for use in dmonitoringmodeld.

  Provides the same callable interface:
    output = runner(**inputs)  ->  numpy array
  """

  def __init__(self, onnx_path: str, dla_core: int = DEFAULT_DLA_CORE):
    self._runner = DLARunner(onnx_path, dla_core=dla_core, fp16=True)
    self._output_name = None

  def __call__(self, **kwargs) -> np.ndarray:
    """Run inference, returning output as flat numpy array."""
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

  @property
  def is_using_dla(self) -> bool:
    return self._runner.is_using_dla

  def destroy(self):
    self._runner.destroy()

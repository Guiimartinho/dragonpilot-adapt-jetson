#!/usr/bin/env python3
import os
from openpilot.system.hardware import TICI, JETSON
if TICI:
  os.environ['DEV'] = 'QCOM'
elif JETSON:
  os.environ['DEV'] = 'CUDA'
  os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
  os.environ.setdefault('FLOAT16', '1')
  os.environ.setdefault('JIT_BATCH_SIZE', '32')  # Consolidate kernels into unified CUDA graph
  os.environ.setdefault('TC', '1')  # Enable tensor cores on Volta sm_72
else:
  os.environ['DEV'] = 'CPU'
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import time
import pickle
import numpy as np
from pathlib import Path

from cereal import messaging
from cereal.messaging import PubMaster, SubMaster
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import config_realtime_process
from openpilot.common.transformations.model import dmonitoringmodel_intrinsics
from openpilot.common.transformations.camera import _ar_ox_fisheye, _os_fisheye
from openpilot.selfdrive.modeld.models.commonmodel_pyx import CLContext, MonitoringModelFrame
from openpilot.selfdrive.modeld.parse_model_outputs import sigmoid, safe_exp
from openpilot.selfdrive.modeld.runners.tinygrad_helpers import qcom_tensor_from_opencl_address

PROCESS_NAME = "selfdrive.modeld.dmonitoringmodeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')
MODEL_PKL_PATH = Path(__file__).parent / 'models/dmonitoring_model_tinygrad.pkl'
MODEL_ONNX_PATH = Path(__file__).parent / 'models/dmonitoring_model.onnx'
METADATA_PATH = Path(__file__).parent / 'models/dmonitoring_model_metadata.pkl'

# DLA/TensorRT support for Jetson (offloads dmonitoring from GPU to DLA)
USE_DLA = JETSON and os.getenv("USE_DLA", "1") == "1"
USE_TENSORRT = JETSON and os.getenv("USE_TENSORRT", "1") == "1"
_dla_available = False
_trt_available = False
if USE_DLA:
  try:
    from openpilot.selfdrive.modeld.runners.dla_runner import DLAModelRunner
    _dla_available = True
  except ImportError:
    pass
if not _dla_available and USE_TENSORRT:
  try:
    from openpilot.selfdrive.modeld.runners.tensorrt_runner import TensorRTModelRunner
    _trt_available = True
  except ImportError:
    pass


class ModelState:
  inputs: dict[str, np.ndarray]
  output: np.ndarray

  def __init__(self, cl_ctx):
    with open(METADATA_PATH, 'rb') as f:
      model_metadata = pickle.load(f)
      self.input_shapes = model_metadata['input_shapes']
      self.output_slices = model_metadata['output_slices']

    self.frame = MonitoringModelFrame(cl_ctx)
    self.numpy_inputs = {
      'calib': np.zeros(self.input_shapes['calib'], dtype=np.float32),
    }

    self.tensor_inputs = {k: Tensor(v, device='NPY').realize() for k,v in self.numpy_inputs.items()}

    # Initialize inference backend: DLA (preferred) > TensorRT GPU > tinygrad
    self._use_dla = False
    self._use_trt = False
    if _dla_available and MODEL_ONNX_PATH.exists():
      try:
        self._dla_runner = DLAModelRunner(str(MODEL_ONNX_PATH), dla_core=0)
        self._use_dla = True
        cloudlog.warning("dmonitoringmodeld: DLA runner loaded (core=%d, using_dla=%s)",
                         self._dla_runner._runner.active_dla_core, self._dla_runner.is_using_dla)
      except Exception as e:
        cloudlog.warning("dmonitoringmodeld: DLA init failed (%s), trying TensorRT GPU", e)
    if not self._use_dla and _trt_available and MODEL_ONNX_PATH.exists():
      try:
        self._trt_runner = TensorRTModelRunner(str(MODEL_ONNX_PATH), fp16=True)
        self._use_trt = True
        cloudlog.warning("dmonitoringmodeld: TensorRT GPU runner loaded")
      except Exception as e:
        cloudlog.warning("dmonitoringmodeld: TensorRT init failed (%s), using tinygrad", e)
    if not self._use_dla and not self._use_trt:
      with open(MODEL_PKL_PATH, "rb") as f:
        self.model_run = pickle.load(f)

  def run(self, buf: VisionBuf, calib: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, float]:
    self.numpy_inputs['calib'][0,:] = calib

    t1 = time.perf_counter()

    input_img_cl = self.frame.prepare(buf, transform.flatten())
    if TICI:
      # The imgs tensors are backed by Qualcomm opencl memory, only need init once
      if 'input_img' not in self.tensor_inputs:
        self.tensor_inputs['input_img'] = qcom_tensor_from_opencl_address(input_img_cl.mem_address, self.input_shapes['input_img'], dtype=dtypes.uint8)
    else:
      # Generic path: OpenCL buffer → CPU → Tensor (used by Jetson CUDA, PC CPU)
      self.tensor_inputs['input_img'] = Tensor(self.frame.buffer_from_cl(input_img_cl).reshape(self.input_shapes['input_img']), dtype=dtypes.uint8).realize()

    if self._use_dla or self._use_trt:
      # DLA/TensorRT path: convert tensors to numpy for TRT inference
      np_inputs = {}
      for k, v in self.tensor_inputs.items():
        np_inputs[k] = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
      runner = self._dla_runner if self._use_dla else self._trt_runner
      output = runner(**np_inputs)
    else:
      output = self.model_run(**self.tensor_inputs).contiguous().realize().uop.base.buffer.numpy()

    t2 = time.perf_counter()
    return output, t2 - t1

def slice_outputs(model_outputs, output_slices):
  return  {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}

def parse_model_output(model_output):
  parsed = {}
  parsed['wheel_on_right'] = sigmoid(model_output['wheel_on_right'])
  for ds_suffix in ['lhd', 'rhd']:
    face_descs = model_output[f'face_descs_{ds_suffix}']
    parsed[f'face_descs_{ds_suffix}'] = face_descs[:, :-6]
    parsed[f'face_descs_{ds_suffix}_std'] = safe_exp(face_descs[:, -6:])
    for key in ['face_prob', 'left_eye_prob', 'right_eye_prob','left_blink_prob', 'right_blink_prob', 'sunglasses_prob', 'using_phone_prob']:
      parsed[f'{key}_{ds_suffix}'] = sigmoid(model_output[f'{key}_{ds_suffix}'])
  return parsed

def fill_driver_data(msg, model_output, ds_suffix):
  msg.faceOrientation = model_output[f'face_descs_{ds_suffix}'][0, :3].tolist()
  msg.faceOrientationStd = model_output[f'face_descs_{ds_suffix}_std'][0, :3].tolist()
  msg.facePosition = model_output[f'face_descs_{ds_suffix}'][0, 3:5].tolist()
  msg.facePositionStd = model_output[f'face_descs_{ds_suffix}_std'][0, 3:5].tolist()
  msg.faceProb = model_output[f'face_prob_{ds_suffix}'][0, 0].item()
  msg.leftEyeProb = model_output[f'left_eye_prob_{ds_suffix}'][0, 0].item()
  msg.rightEyeProb = model_output[f'right_eye_prob_{ds_suffix}'][0, 0].item()
  msg.leftBlinkProb = model_output[f'left_blink_prob_{ds_suffix}'][0, 0].item()
  msg.rightBlinkProb = model_output[f'right_blink_prob_{ds_suffix}'][0, 0].item()
  msg.sunglassesProb = model_output[f'sunglasses_prob_{ds_suffix}'][0, 0].item()
  msg.phoneProb = model_output[f'using_phone_prob_{ds_suffix}'][0, 0].item()

def get_driverstate_packet(model_output, frame_id: int, location_ts: int, exec_time: float, gpu_exec_time: float):
  msg = messaging.new_message('driverStateV2', valid=True)
  ds = msg.driverStateV2
  ds.frameId = frame_id
  ds.modelExecutionTime = exec_time
  ds.gpuExecutionTime = gpu_exec_time
  ds.rawPredictions = model_output['raw_pred']
  ds.wheelOnRightProb = model_output['wheel_on_right'][0, 0].item()
  fill_driver_data(ds.leftDriverData, model_output, 'lhd')
  fill_driver_data(ds.rightDriverData, model_output, 'rhd')
  return msg


def main():
  config_realtime_process(6 if JETSON else 7, 5)

  cl_context = CLContext()
  model = ModelState(cl_context)
  cloudlog.warning("models loaded, dmonitoringmodeld starting")

  cloudlog.warning("connecting to driver stream")
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True, cl_context)
  while not vipc_client.connect(False):
    time.sleep(0.1)
  assert vipc_client.is_connected()
  cloudlog.warning(f"connected with buffer size: {vipc_client.buffer_len}")

  sm = SubMaster(["liveCalibration"])
  pm = PubMaster(["driverStateV2"])

  calib = np.zeros(model.numpy_inputs['calib'].size, dtype=np.float32)
  model_transform = None

  while True:
    buf = vipc_client.recv()
    if buf is None:
      continue

    if model_transform is None:
      cam = _os_fisheye if buf.width == _os_fisheye.width else _ar_ox_fisheye
      model_transform = np.linalg.inv(np.dot(dmonitoringmodel_intrinsics, np.linalg.inv(cam.intrinsics))).astype(np.float32)

    sm.update(0)
    if sm.updated["liveCalibration"]:
      calib[:] = np.array(sm["liveCalibration"].rpyCalib)

    t1 = time.perf_counter()
    model_output, gpu_execution_time = model.run(buf, calib, model_transform)
    t2 = time.perf_counter()
    raw_pred = model_output.tobytes() if SEND_RAW_PRED else b''
    model_output = slice_outputs(model_output, model.output_slices)
    model_output = parse_model_output(model_output)
    model_output['raw_pred'] = raw_pred
    msg = get_driverstate_packet(model_output, vipc_client.frame_id, vipc_client.timestamp_sof, t2 - t1, gpu_execution_time)
    pm.send("driverStateV2", msg)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")

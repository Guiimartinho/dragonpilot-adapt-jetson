#!/usr/bin/env python3
"""
Jetson camera daemon with MIPI CSI-2 support.

Priority chain:
  1. MIPI CSI-2 via V4L2 (native ISP, zero CPU, NV12 direct)
  2. USB webcam via OpenCV (fallback)

Publishes frames to VisionIPC for modeld/dmonitoringmodeld consumption.
"""

import os
import time
import select
import threading
import logging

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

logger = logging.getLogger(__name__)

ROAD_CAM_CSI = os.getenv("ROAD_CAM_CSI")
DRIVER_CAM_CSI = os.getenv("DRIVER_CAM_CSI")
FPS = 20
VIPC_BUFFER_COUNT = 20


def _try_csi_camerad():
  """Try to start CSI camera daemon. Returns JetsonCamerad or None."""
  try:
    from openpilot.system.camerad.cameras.camera_jetson import JetsonCamerad
    camerad = JetsonCamerad()
    if camerad.detect_and_open(road_cam=ROAD_CAM_CSI, driver_cam=DRIVER_CAM_CSI):
      return camerad
  except Exception as e:
    cloudlog.warning("CSI camera init failed: %s", e)
  return None


class JetsonCameradMain:
  """Main camera daemon for Jetson with CSI/webcam auto-detection."""

  def __init__(self):
    self._csi = _try_csi_camerad()
    self._use_csi = self._csi is not None and self._csi.is_csi

    if self._use_csi:
      cloudlog.warning("Using MIPI CSI-2 cameras")
      self._init_csi()
    else:
      cloudlog.warning("CSI cameras not available, delegating to webcam camerad")

  def _init_csi(self):
    """Initialize VisionIPC server for CSI cameras."""
    self.pm = messaging.PubMaster(["roadCameraState", "driverCameraState"])
    self.vipc_server = VisionIpcServer("camerad")

    road_cam = self._csi._cameras.get("road")
    if road_cam:
      self.vipc_server.create_buffers(
        VisionStreamType.VISION_STREAM_ROAD, VIPC_BUFFER_COUNT,
        int(road_cam.width), int(road_cam.height)
      )

    driver_cam = self._csi._cameras.get("driver")
    if driver_cam:
      self.vipc_server.create_buffers(
        VisionStreamType.VISION_STREAM_DRIVER, VIPC_BUFFER_COUNT,
        int(driver_cam.width), int(driver_cam.height)
      )

    self.vipc_server.start_listener()

  def _send_frame(self, frame_data: bytes, frame_id: int, cam_type: str):
    """Send a frame via VisionIPC and publish camera state."""
    ts_ns = int(time.monotonic() * 1e9)

    if cam_type == "road":
      stream = VisionStreamType.VISION_STREAM_ROAD
      msg_type = "roadCameraState"
    else:
      stream = VisionStreamType.VISION_STREAM_DRIVER
      msg_type = "driverCameraState"

    self.vipc_server.send(stream, frame_data, frame_id, ts_ns, ts_ns)

    dat = messaging.new_message(msg_type, valid=True)
    msg = {
      "frameId": frame_id,
      "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    setattr(dat, msg_type, msg)
    self.pm.send(msg_type, dat)

  def _camera_loop(self, cam_name: str):
    """Per-camera capture loop."""
    rk = Ratekeeper(FPS, None)
    frame_id = 0

    while True:
      result = self._csi.get_frame(cam_name)
      if result is None:
        time.sleep(0.001)
        continue

      frame_data, _ = result
      self._send_frame(frame_data, frame_id, cam_name)
      frame_id += 1
      rk.keep_time()

  def run(self):
    """Run the camera daemon."""
    if not self._use_csi:
      # Fall back to webcam camerad
      from openpilot.tools.webcam.camerad import main as webcam_main
      webcam_main()
      return

    if not self._csi.start():
      cloudlog.error("Failed to start CSI cameras, falling back to webcam")
      from openpilot.tools.webcam.camerad import main as webcam_main
      webcam_main()
      return

    cloudlog.warning("Jetson CSI camerad started")

    threads = []
    for cam_name in self._csi._cameras:
      t = threading.Thread(target=self._camera_loop, args=(cam_name,), daemon=True)
      t.start()
      threads.append(t)

    for t in threads:
      t.join()


def main():
  camerad = JetsonCameradMain()
  camerad.run()


if __name__ == "__main__":
  main()

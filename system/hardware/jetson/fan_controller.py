import numpy as np

from openpilot.common.realtime import DT_HW
from openpilot.common.swaglog import cloudlog
from openpilot.common.pid import PIDController
from openpilot.system.hardware.fan_controller import BaseFanController


class JetsonFanController(BaseFanController):
  # Jetson AGX Xavier PWM fan control via /sys/devices/pwm-fan/target_pwm
  # Range: 0-255 (mapped to 0-100% for the interface)

  FAN_PWM_PATH = "/sys/devices/pwm-fan/target_pwm"

  def __init__(self) -> None:
    super().__init__()
    cloudlog.info("Setting up Jetson fan handler")

    self.last_ignition = False
    # PID with proportional gain for faster response to temp spikes
    self.controller = PIDController(k_p=0.5, k_i=2e-3, k_d=1e-2, rate=(1 / DT_HW))
    self._last_pwm = 0
    self._hysteresis_band = 5  # Prevent oscillation (5% band)

  def update(self, cur_temp: float, ignition: bool) -> int:
    self.controller.pos_limit = 100 if ignition else 40
    self.controller.neg_limit = 40 if ignition else 0

    if ignition != self.last_ignition:
      self.controller.reset()

    # Target 65C when driving (lower for sustained GPU load), 70C when parked
    target_temp = 65 if ignition else 70
    error = cur_temp - target_temp
    fan_pwr_out = int(self.controller.update(
                      error=error,
                      feedforward=np.interp(cur_temp, [50.0, 90.0], [0, 100])
                    ))

    # Hysteresis: prevent rapid fan speed oscillation
    if abs(fan_pwr_out - self._last_pwm) < self._hysteresis_band:
      fan_pwr_out = self._last_pwm

    # Write PWM value to sysfs (0-255 scale)
    pwm_value = int(np.interp(fan_pwr_out, [0, 100], [0, 255]))
    try:
      with open(self.FAN_PWM_PATH, 'w') as f:
        f.write(str(pwm_value))
    except (FileNotFoundError, PermissionError):
      pass

    self._last_pwm = fan_pwr_out
    self.last_ignition = ignition
    return fan_pwr_out

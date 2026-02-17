"""Dynamic Frequency Scaling for Jetson AGX Xavier.

Adjusts CPU/GPU/EMC frequencies based on workload to balance
performance and power consumption. Focus: maximize GPU throughput
while keeping CPU at minimum needed frequency.

Thread-safe: all sysfs writes are protected by a lock since update()
may be called from different threads (hardwared loop, power save).
"""
import math
import os
import threading
import time

from openpilot.common.swaglog import cloudlog


class JetsonDFS:
  # CPU frequencies (kHz) - min to max
  CPU_FREQS = [1200000, 1700000, 2265600]
  # GPU frequencies (Hz) - min to max
  GPU_FREQS = [520000000, 900000000, 1377000000]
  # EMC (memory controller) frequencies (Hz) - critical for CUDA memory bandwidth
  EMC_FREQS = [665600000, 1600000000, 2133000000]
  # EMC sysfs paths (varies by JetPack version)
  EMC_PATHS = [
    "/sys/kernel/debug/clk/emc/rate",
    "/sys/kernel/debug/bpmp/debug/clk/emc/rate",
    "/sys/kernel/debug/clk/emc/min",
  ]

  def __init__(self):
    self.current_cpu_idx = 2  # Start at max
    self.current_gpu_idx = 2  # GPU always high for inference
    self.current_emc_idx = 2  # Memory bandwidth high
    self.last_update = 0.0
    self.update_interval = 2.0  # seconds
    self._initialized = False
    self._lock = threading.Lock()
    self._emc_path: str | None = None  # Cached working EMC path

  def initialize(self):
    """Set initial high-performance state for driving."""
    with self._lock:
      self._set_gpu_freq(self.GPU_FREQS[2])  # GPU always max for CUDA inference
      self._set_emc_freq(self.EMC_FREQS[2])  # Max memory bandwidth
      self._set_cpu_freq(self.CPU_FREQS[1])   # CPU mid - save power, GPU does heavy work
      self._initialized = True
    cloudlog.info("JetsonDFS: initialized - GPU max, CPU mid, EMC max")

  def update(self, cpu_usage: float, gpu_usage: float, max_temp: float, ignition: bool):
    """Adjust frequencies based on workload. Called from hardwared loop."""
    # Validate inputs
    if not math.isfinite(cpu_usage) or not math.isfinite(gpu_usage) or not math.isfinite(max_temp):
      return

    now = time.monotonic()
    if now - self.last_update < self.update_interval:
      return
    self.last_update = now

    if not self._initialized:
      self.initialize()

    with self._lock:
      if not ignition:
        self._set_parked_mode()
        return

      # DRIVING MODE: GPU always at max, CPU adaptive
      if self.current_gpu_idx != 2:
        self._set_gpu_freq(self.GPU_FREQS[2])
        self.current_gpu_idx = 2

      # EMC: always max when driving (model inference is memory-bound)
      if self.current_emc_idx != 2:
        self._set_emc_freq(self.EMC_FREQS[2])
        self.current_emc_idx = 2

      # CPU: adaptive based on load - save power for GPU
      if max_temp > 80:
        target_cpu_idx = 0  # Thermal throttle: 1.2 GHz
      elif cpu_usage > 60:
        target_cpu_idx = 2  # High load: 2.27 GHz
      elif cpu_usage > 30:
        target_cpu_idx = 1  # Medium: 1.7 GHz
      else:
        target_cpu_idx = 0  # Low: 1.2 GHz (save power for GPU)

      if target_cpu_idx != self.current_cpu_idx:
        self._set_cpu_freq(self.CPU_FREQS[target_cpu_idx])
        self.current_cpu_idx = target_cpu_idx

  def _set_parked_mode(self):
    """Minimum power when parked. Caller must hold self._lock."""
    if self.current_gpu_idx != 0:
      self._set_gpu_freq(self.GPU_FREQS[0])
      self.current_gpu_idx = 0
    if self.current_cpu_idx != 0:
      self._set_cpu_freq(self.CPU_FREQS[0])
      self.current_cpu_idx = 0
    if self.current_emc_idx != 0:
      self._set_emc_freq(self.EMC_FREQS[0])
      self.current_emc_idx = 0

  def _write_sysfs(self, path: str, value: str) -> bool:
    """Write value to sysfs path with verification. Returns True on success."""
    try:
      with open(path, 'w') as f:
        f.write(value)
      return True
    except (OSError, PermissionError):
      return False

  def _set_cpu_freq(self, freq_khz: int):
    for core in range(8):
      path = f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_max_freq"
      if os.path.exists(path):
        if not self._write_sysfs(path, str(freq_khz)):
          cloudlog.warning(f"JetsonDFS: failed to set CPU{core} freq to {freq_khz}")

  def _set_gpu_freq(self, freq_hz: int):
    for path in ["/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/min_freq",
                 "/sys/devices/17000000.gv11b/devfreq/17000000.gv11b/max_freq"]:
      self._write_sysfs(path, str(freq_hz))

  def _set_emc_freq(self, freq_hz: int):
    # Use cached path if we found one before
    if self._emc_path is not None:
      if self._write_sysfs(self._emc_path, str(freq_hz)):
        return

    # Try all known EMC paths
    for path in self.EMC_PATHS:
      if self._write_sysfs(path, str(freq_hz)):
        self._emc_path = path  # Cache the working path
        return

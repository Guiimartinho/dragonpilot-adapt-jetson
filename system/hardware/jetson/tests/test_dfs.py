"""Tests for Jetson Dynamic Frequency Scaling."""
import math
import threading
import unittest
from unittest.mock import patch, mock_open, call

from openpilot.system.hardware.jetson.dfs import JetsonDFS


class TestJetsonDFS(unittest.TestCase):
  def setUp(self):
    self.dfs = JetsonDFS()

  def test_initial_state(self):
    assert self.dfs.current_cpu_idx == 2
    assert self.dfs.current_gpu_idx == 2
    assert self.dfs.current_emc_idx == 2
    assert not self.dfs._initialized

  @patch("builtins.open", mock_open())
  @patch("os.path.exists", return_value=True)
  def test_initialize(self, mock_exists):
    self.dfs.initialize()
    assert self.dfs._initialized

  @patch("builtins.open", mock_open())
  @patch("os.path.exists", return_value=True)
  def test_update_ignition_off_parks(self, mock_exists):
    self.dfs._initialized = True
    self.dfs.last_update = 0  # Force update
    self.dfs.update(10.0, 20.0, 50.0, ignition=False)
    assert self.dfs.current_cpu_idx == 0
    assert self.dfs.current_gpu_idx == 0
    assert self.dfs.current_emc_idx == 0

  @patch("builtins.open", mock_open())
  @patch("os.path.exists", return_value=True)
  def test_update_high_cpu_usage(self, mock_exists):
    self.dfs._initialized = True
    self.dfs.last_update = 0
    self.dfs.current_cpu_idx = 0  # Start low
    self.dfs.update(70.0, 50.0, 60.0, ignition=True)
    assert self.dfs.current_cpu_idx == 2  # Should go to max

  @patch("builtins.open", mock_open())
  @patch("os.path.exists", return_value=True)
  def test_update_thermal_throttle(self, mock_exists):
    self.dfs._initialized = True
    self.dfs.last_update = 0
    self.dfs.current_cpu_idx = 2
    self.dfs.update(30.0, 50.0, 85.0, ignition=True)
    assert self.dfs.current_cpu_idx == 0  # Thermal throttle

  def test_nan_inputs_ignored(self):
    """NaN inputs should not trigger any frequency changes."""
    self.dfs._initialized = True
    self.dfs.last_update = 0
    initial_cpu = self.dfs.current_cpu_idx
    self.dfs.update(float('nan'), 50.0, 60.0, ignition=True)
    assert self.dfs.current_cpu_idx == initial_cpu

  def test_inf_inputs_ignored(self):
    self.dfs._initialized = True
    self.dfs.last_update = 0
    initial_cpu = self.dfs.current_cpu_idx
    self.dfs.update(50.0, float('inf'), 60.0, ignition=True)
    assert self.dfs.current_cpu_idx == initial_cpu

  @patch("builtins.open", mock_open())
  @patch("os.path.exists", return_value=True)
  def test_update_interval_throttle(self, mock_exists):
    """Updates within the interval should be skipped."""
    import time
    self.dfs._initialized = True
    self.dfs.last_update = time.monotonic()  # Just updated
    initial_cpu = self.dfs.current_cpu_idx
    self.dfs.update(70.0, 50.0, 60.0, ignition=True)
    # Should not have changed since we're within the interval
    assert self.dfs.current_cpu_idx == initial_cpu

  def test_thread_safety(self):
    """Concurrent updates should not cause race conditions."""
    self.dfs._initialized = True
    errors = []

    def worker(ignition):
      try:
        for _ in range(10):
          self.dfs.last_update = 0
          self.dfs.update(50.0, 50.0, 60.0, ignition=ignition)
      except Exception as e:
        errors.append(e)

    with patch("builtins.open", mock_open()), patch("os.path.exists", return_value=True):
      threads = [threading.Thread(target=worker, args=(True,)),
                 threading.Thread(target=worker, args=(False,))]
      for t in threads:
        t.start()
      for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety errors: {errors}"

  def test_emc_path_caching(self):
    """Successful EMC path should be cached."""
    assert self.dfs._emc_path is None
    with patch("builtins.open", mock_open()):
      self.dfs._set_emc_freq(2133000000)
      assert self.dfs._emc_path is not None

  @patch("builtins.open", side_effect=OSError("Permission denied"))
  def test_write_sysfs_failure(self, mock_file):
    result = self.dfs._write_sysfs("/fake/path", "123")
    assert result is False


if __name__ == '__main__':
  unittest.main()

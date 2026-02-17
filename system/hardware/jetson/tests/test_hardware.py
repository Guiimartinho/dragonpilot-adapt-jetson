"""Tests for Jetson hardware abstraction layer."""
import threading
import unittest
from unittest.mock import patch, mock_open, MagicMock

from openpilot.system.hardware.jetson.hardware import Jetson


class TestJetsonHardware(unittest.TestCase):
  def setUp(self):
    self.hw = Jetson()

  def test_get_device_type(self):
    assert self.hw.get_device_type() == "pc"

  @patch("builtins.open", mock_open(read_data="# R35 (release), REVISION: 4.1\n"))
  def test_get_os_version(self):
    result = self.hw.get_os_version()
    assert "R35" in result

  def test_get_os_version_missing(self):
    with patch("builtins.open", side_effect=FileNotFoundError):
      result = self.hw.get_os_version()
      assert result == "JetPack unknown"

  @patch("builtins.open", mock_open(read_data="0x12345678"))
  def test_get_serial(self):
    result = self.hw.get_serial()
    assert result == "0x12345678"

  def test_get_serial_missing(self):
    with patch("builtins.open", side_effect=FileNotFoundError):
      result = self.hw.get_serial()
      assert result == "jetson-unknown"

  @patch("builtins.open", mock_open(read_data="750\n"))
  def test_get_gpu_usage_percent(self):
    result = self.hw.get_gpu_usage_percent()
    assert result == 75.0

  def test_get_gpu_usage_missing(self):
    with patch("builtins.open", side_effect=FileNotFoundError):
      result = self.hw.get_gpu_usage_percent()
      assert result == 0

  def test_get_modem_temperatures(self):
    assert self.hw.get_modem_temperatures() == []

  def test_get_thermal_config(self):
    config = self.hw.get_thermal_config()
    assert len(config.cpu) > 0
    assert len(config.gpu) > 0

  def test_get_network_type_none(self):
    with patch("os.listdir", return_value=[]):
      from cereal import log
      result = self.hw.get_network_type()
      assert result == log.DeviceState.NetworkType.none

  def test_get_sim_info(self):
    info = self.hw.get_sim_info()
    assert info['sim_id'] == ''
    assert info['data_connected'] is False

  def test_get_voltage_missing(self):
    with patch("os.path.exists", return_value=False):
      result = self.hw.get_voltage()
      assert result == 0.0

  def test_get_current_missing(self):
    with patch("os.path.exists", return_value=False):
      result = self.hw.get_current()
      assert result == 0.0

  def test_watchdog_stop(self):
    """Verify the watchdog thread can be stopped cleanly."""
    assert not self.hw._watchdog_stop.is_set()
    self.hw.stop_watchdog()
    assert self.hw._watchdog_stop.is_set()

  @patch("subprocess.check_output")
  def test_set_power_save_on(self, mock_subprocess):
    with patch("builtins.open", mock_open()):
      self.hw.set_power_save(True)
      mock_subprocess.assert_called()

  @patch("subprocess.check_output")
  def test_set_power_save_off(self, mock_subprocess):
    with patch("builtins.open", mock_open()):
      self.hw.set_power_save(False)
      mock_subprocess.assert_called()

  def test_get_current_power_draw_zero(self):
    with patch("builtins.open", side_effect=FileNotFoundError):
      with patch("os.path.exists", return_value=False):
        result = self.hw.get_current_power_draw()
        assert result == 0


if __name__ == '__main__':
  unittest.main()

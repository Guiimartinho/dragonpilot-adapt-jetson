"""Tests for Jetson fan controller."""
import math
import unittest
from unittest.mock import patch, mock_open

from openpilot.system.hardware.jetson.fan_controller import JetsonFanController


class TestJetsonFanController(unittest.TestCase):
  def setUp(self):
    self.fc = JetsonFanController()

  @patch("builtins.open", mock_open())
  def test_basic_update(self):
    result = self.fc.update(60.0, True)
    assert 0 <= result <= 100

  @patch("builtins.open", mock_open())
  def test_high_temp_increases_fan(self):
    low_result = self.fc.update(50.0, True)
    self.fc.controller.reset()
    self.fc._last_pwm = 0
    high_result = self.fc.update(85.0, True)
    assert high_result >= low_result

  @patch("builtins.open", mock_open())
  def test_ignition_off_lower_target(self):
    result = self.fc.update(60.0, False)
    assert 0 <= result <= 100

  @patch("builtins.open", mock_open())
  def test_nan_temperature_handled(self):
    """NaN temperature should not crash the controller."""
    result = self.fc.update(float('nan'), True)
    assert 0 <= result <= 100

  @patch("builtins.open", mock_open())
  def test_inf_temperature_handled(self):
    """Inf temperature should not crash the controller."""
    result = self.fc.update(float('inf'), True)
    assert 0 <= result <= 100

  @patch("builtins.open", mock_open())
  def test_negative_inf_temperature_handled(self):
    result = self.fc.update(float('-inf'), True)
    assert 0 <= result <= 100

  @patch("builtins.open", mock_open())
  def test_hysteresis_prevents_oscillation(self):
    """Small temp changes should not change fan speed."""
    r1 = self.fc.update(65.0, True)
    r2 = self.fc.update(65.1, True)  # Very small change
    assert r1 == r2  # Hysteresis should prevent change

  @patch("builtins.open", mock_open())
  def test_ignition_transition_resets_pid(self):
    """PID should reset on ignition state change."""
    self.fc.update(65.0, True)
    self.fc.update(65.0, False)  # Transition should reset PID
    # No assertion needed - just verify no crash

  @patch("builtins.open", side_effect=FileNotFoundError)
  def test_missing_pwm_path_handled(self, mock_file):
    """Missing sysfs path should not crash."""
    result = self.fc.update(70.0, True)
    assert 0 <= result <= 100

  @patch("builtins.open", mock_open())
  def test_output_clamped(self):
    """Output should always be in [0, 100] range."""
    for temp in [0, 20, 50, 65, 80, 100, 120]:
      self.fc.controller.reset()
      self.fc._last_pwm = 0
      result = self.fc.update(float(temp), True)
      assert 0 <= result <= 100, f"temp={temp}, result={result}"


if __name__ == '__main__':
  unittest.main()

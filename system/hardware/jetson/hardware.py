import os
import subprocess

from cereal import log
from openpilot.system.hardware.base import HardwareBase, LPABase, ThermalConfig, ThermalZone

NetworkType = log.DeviceState.NetworkType
NetworkStrength = log.DeviceState.NetworkStrength


class Jetson(HardwareBase):
  def get_os_version(self):
    try:
      with open("/etc/nv_tegra_release") as f:
        return f.read().strip().split(",")[0]
    except FileNotFoundError:
      return "JetPack unknown"

  def get_device_type(self):
    return "jetson"

  def reboot(self, reason=None):
    subprocess.check_output(["sudo", "reboot"])

  def uninstall(self):
    pass

  def get_imei(self, slot):
    return ""

  def get_serial(self):
    for path in ["/sys/module/fuse_burn/parameters/tegra_chip_uid",
                 "/sys/module/tegra_fuse/parameters/tegra_chip_uid"]:
      try:
        with open(path) as f:
          return f.read().strip()
      except FileNotFoundError:
        continue
    return "jetson-unknown"

  def get_network_info(self):
    return None

  def get_network_type(self):
    # Check for ethernet first, then wifi
    try:
      for iface in os.listdir("/sys/class/net/"):
        if iface == "lo":
          continue
        operstate_path = f"/sys/class/net/{iface}/operstate"
        if os.path.exists(operstate_path):
          with open(operstate_path) as f:
            if f.read().strip() == "up":
              if iface.startswith("eth") or iface.startswith("enp"):
                return NetworkType.ethernet
              elif iface.startswith("wlan") or iface.startswith("wlp"):
                return NetworkType.wifi
    except Exception:
      pass
    return NetworkType.none

  def get_sim_info(self):
    return {
      'sim_id': '',
      'mcc_mnc': None,
      'network_type': ["Unknown"],
      'sim_state': ["ABSENT"],
      'data_connected': False
    }

  def get_sim_lpa(self) -> LPABase:
    raise NotImplementedError("SIM LPA not available on Jetson")

  def get_network_strength(self, network_type):
    return NetworkStrength.unknown

  def get_current_power_draw(self):
    # INA3221 power monitor on Jetson AGX Xavier
    # Channel 0: GPU, Channel 1: CPU, Channel 2: SoC
    total_power = 0
    for channel in range(3):
      path = f"/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0/in{channel + 1}_input"
      curr_path = f"/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0/curr{channel + 1}_input"
      try:
        with open(path) as f:
          voltage_mv = int(f.read().strip())
        with open(curr_path) as f:
          current_ma = int(f.read().strip())
        total_power += (voltage_mv * current_ma) / 1_000_000  # Convert to watts
      except (FileNotFoundError, ValueError):
        pass
    # Fallback: try the simpler sysfs path
    if total_power == 0:
      try:
        for base in ["/sys/bus/i2c/drivers/ina3221/1-0040/hwmon", "/sys/bus/i2c/drivers/ina3221x/1-0040/hwmon"]:
          if os.path.exists(base):
            hwmon = os.listdir(base)[0]
            for ch in range(1, 4):
              try:
                with open(f"{base}/{hwmon}/in{ch}_input") as f:
                  v = int(f.read().strip())
                with open(f"{base}/{hwmon}/curr{ch}_input") as f:
                  c = int(f.read().strip())
                total_power += (v * c) / 1_000_000
              except (FileNotFoundError, ValueError):
                pass
            break
      except (OSError, IndexError):
        pass
    return total_power

  def get_som_power_draw(self):
    return self.get_current_power_draw()

  def shutdown(self):
    subprocess.check_output(["sudo", "shutdown", "-h", "now"])

  def get_thermal_config(self):
    return ThermalConfig(
      cpu=[ThermalZone("CPU-therm")],
      gpu=[ThermalZone("GPU-therm")],
      memory=ThermalZone("Tdiode_tegra"),
      pmic=[ThermalZone("PMIC-Die")],
    )

  def set_screen_brightness(self, percentage):
    pass

  def get_screen_brightness(self):
    return 0

  def set_power_save(self, powersave_enabled):
    # nvpmodel: 0 = MAXN (30W), 2 = MODE_15W, 1 = MODE_10W
    try:
      if powersave_enabled:
        subprocess.check_output(["sudo", "nvpmodel", "-m", "2"], stderr=subprocess.STDOUT)
      else:
        subprocess.check_output(["sudo", "nvpmodel", "-m", "0"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
      pass

  def get_gpu_usage_percent(self):
    try:
      with open("/sys/devices/gpu.0/load") as f:
        return int(f.read().strip()) / 10
    except (FileNotFoundError, ValueError):
      return 0

  def get_modem_temperatures(self):
    return []

  def initialize_hardware(self):
    # Lock clocks to maximum for consistent performance
    try:
      subprocess.check_output(["sudo", "jetson_clocks"], stderr=subprocess.STDOUT)
    except (subprocess.CalledProcessError, FileNotFoundError):
      pass

  def get_networks(self):
    return None

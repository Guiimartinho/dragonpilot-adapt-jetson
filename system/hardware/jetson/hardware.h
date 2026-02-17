#pragma once

#include <cstdlib>
#include <fstream>
#include <string>

#include "common/util.h"
#include "system/hardware/base.h"

class HardwareJetson : public HardwareNone {
public:
  static std::string get_os_version() {
    std::string ver = util::read_file("/etc/nv_tegra_release");
    return ver.empty() ? "JetPack unknown" : ver.substr(0, ver.find(','));
  }

  static std::string get_name() { return "jetson"; }
  static cereal::InitData::DeviceType get_device_type() { return cereal::InitData::DeviceType::PC; }

  static std::string get_serial() {
    std::string uid = util::read_file("/sys/module/tegra_fuse/parameters/tegra_chip_uid");
    return uid.empty() ? "jetson-unknown" : util::strip(uid);
  }

  static int get_voltage() {
    // INA3221 channel 1 voltage (mV)
    std::string v = util::read_file("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0/in1_input");
    return v.empty() ? 0 : std::atoi(v.c_str());
  }

  static int get_current() {
    // INA3221 channel 1 current (mA)
    std::string c = util::read_file("/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon0/curr1_input");
    return c.empty() ? 0 : std::atoi(c.c_str());
  }

  static std::map<std::string, std::string> get_init_logs() {
    return {
      {"os_version", get_os_version()},
      {"serial", get_serial()},
      {"lsblk", util::check_output("lsblk -o NAME,SIZE,STATE,VENDOR,MODEL,REV,SERIAL")},
    };
  }

  static bool PC() { return false; }
  static bool TICI() { return false; }
  static bool AGNOS() { return false; }
};

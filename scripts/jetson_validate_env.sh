#!/bin/bash
# =============================================================================
# Jetson AGX Xavier Environment Validation Script
# DragonPilot 0.10.3 - Phase 0 Validation
# =============================================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
FAIL=0
WARN=0

check_pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; ((PASS++)); }
check_fail() { echo -e "  ${RED}[FAIL]${NC} $1"; ((FAIL++)); }
check_warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; ((WARN++)); }

echo "=============================================="
echo " Jetson AGX Xavier - Environment Validation"
echo " DragonPilot 0.10.3 Port"
echo "=============================================="
echo ""

# --- 1. Platform Marker ---
echo "1. Platform Marker"
if [ -f /JETSON ]; then
  check_pass "/JETSON marker file exists"
else
  check_fail "/JETSON marker file missing (run: sudo touch /JETSON)"
fi
echo ""

# --- 2. Architecture ---
echo "2. Architecture"
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
  check_pass "Architecture: $ARCH"
else
  check_fail "Expected aarch64, got: $ARCH"
fi

if [ -f /etc/nv_tegra_release ]; then
  TEGRA_VER=$(head -1 /etc/nv_tegra_release)
  check_pass "Tegra release: $TEGRA_VER"
else
  check_fail "/etc/nv_tegra_release not found - is this a Jetson?"
fi
echo ""

# --- 3. Python ---
echo "3. Python"
if command -v python3.11 &>/dev/null; then
  PY_VER=$(python3.11 --version 2>&1)
  check_pass "Python 3.11: $PY_VER"
else
  check_fail "Python 3.11 not found (run: sudo apt install python3.11 python3.11-dev python3.11-venv)"
fi

if [ -n "$VIRTUAL_ENV" ]; then
  VENV_PY=$(python --version 2>&1)
  check_pass "Virtual env active: $VIRTUAL_ENV ($VENV_PY)"
else
  check_warn "No virtual environment active"
fi
echo ""

# --- 4. CUDA ---
echo "4. CUDA"
if command -v nvcc &>/dev/null; then
  CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}')
  check_pass "CUDA: $CUDA_VER"
else
  check_fail "nvcc not found - CUDA toolkit not installed or not in PATH"
fi

if [ -d /usr/local/cuda/lib64 ]; then
  check_pass "/usr/local/cuda/lib64 exists"
else
  check_fail "/usr/local/cuda/lib64 not found"
fi

if [ -f /usr/local/cuda/include/cuda.h ]; then
  check_pass "cuda.h header found"
else
  check_fail "cuda.h not found in /usr/local/cuda/include/"
fi
echo ""

# --- 5. OpenCL ---
echo "5. OpenCL"
if command -v clinfo &>/dev/null; then
  CL_DEV=$(clinfo 2>/dev/null | grep "Device Name" | head -1 || echo "")
  if [ -n "$CL_DEV" ]; then
    check_pass "OpenCL: $CL_DEV"
  else
    check_warn "clinfo found but no OpenCL devices detected"
  fi
else
  check_fail "clinfo not found (run: sudo apt install clinfo ocl-icd-opencl-dev opencl-headers)"
fi

if [ -d /etc/OpenCL/vendors ] && ls /etc/OpenCL/vendors/*.icd &>/dev/null; then
  check_pass "OpenCL ICD files present"
else
  check_warn "No OpenCL ICD files in /etc/OpenCL/vendors/"
fi
echo ""

# --- 6. System Libraries ---
echo "6. System Libraries (apt)"
REQUIRED_PKGS=(
  clang libzmq3-dev libcapnp-dev capnproto libusb-1.0-0-dev
  libssl-dev libffi-dev libsqlite3-dev libeigen3-dev
  ffmpeg libavformat-dev libavcodec-dev libavutil-dev
  opencl-headers ocl-icd-opencl-dev
  libgles2-mesa-dev libglfw3-dev
  portaudio19-dev gcc-arm-none-eabi
  libjpeg-dev libzstd-dev libbz2-dev
)

MISSING_PKGS=()
for pkg in "${REQUIRED_PKGS[@]}"; do
  if dpkg -s "$pkg" &>/dev/null; then
    : # installed
  else
    MISSING_PKGS+=("$pkg")
  fi
done

if [ ${#MISSING_PKGS[@]} -eq 0 ]; then
  check_pass "All ${#REQUIRED_PKGS[@]} required system packages installed"
else
  check_fail "Missing packages: ${MISSING_PKGS[*]}"
fi
echo ""

# --- 7. Storage ---
echo "7. Storage"
FREE_SPACE=$(df -BG /home 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ -n "$FREE_SPACE" ] && [ "$FREE_SPACE" -gt 50 ]; then
  check_pass "Free space: ${FREE_SPACE}GB (> 50GB)"
elif [ -n "$FREE_SPACE" ]; then
  check_warn "Free space: ${FREE_SPACE}GB (< 50GB recommended)"
else
  check_warn "Could not determine free space"
fi
echo ""

# --- 8. RAM ---
echo "8. Memory"
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_RAM" -ge 16 ]; then
  check_pass "Total RAM: ${TOTAL_RAM}GB"
else
  check_warn "Total RAM: ${TOTAL_RAM}GB (16GB+ recommended)"
fi
echo ""

# --- 9. Jetson Power Mode ---
echo "9. Jetson Power Management"
if command -v nvpmodel &>/dev/null; then
  POWER_MODE=$(sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" || echo "unknown")
  check_pass "Power: $POWER_MODE"
else
  check_warn "nvpmodel not found"
fi
echo ""

# --- 10. GPU Info ---
echo "10. GPU"
if [ -f /sys/devices/gpu.0/load ]; then
  GPU_LOAD=$(cat /sys/devices/gpu.0/load 2>/dev/null || echo "0")
  check_pass "GPU sysfs accessible (current load: $((GPU_LOAD / 10))%)"
else
  check_warn "GPU sysfs at /sys/devices/gpu.0/load not found"
fi
echo ""

# --- 11. Third-party Symlinks ---
echo "11. Third-party Symlinks"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -L "$SCRIPT_DIR/third_party/acados/jarch64" ]; then
  check_pass "acados/jarch64 symlink exists"
else
  check_warn "acados/jarch64 symlink missing (create with: cd third_party/acados && ln -sf aarch64 jarch64)"
fi

if [ -L "$SCRIPT_DIR/third_party/libyuv/jarch64" ]; then
  check_pass "libyuv/jarch64 symlink exists"
else
  check_warn "libyuv/jarch64 symlink missing (create with: cd third_party/libyuv && ln -sf aarch64 jarch64)"
fi
echo ""

# --- 12. Thermal Zones ---
echo "12. Thermal Zones"
THERMAL_ZONES=("CPU-therm" "GPU-therm" "Tdiode_tegra" "PMIC-Die")
for tz_name in "${THERMAL_ZONES[@]}"; do
  found=false
  for tz_dir in /sys/devices/virtual/thermal/thermal_zone*; do
    if [ -f "$tz_dir/type" ]; then
      tz_type=$(cat "$tz_dir/type" 2>/dev/null)
      if [ "$tz_type" = "$tz_name" ]; then
        temp=$(cat "$tz_dir/temp" 2>/dev/null || echo "0")
        check_pass "Thermal zone '$tz_name': $((temp / 1000))C"
        found=true
        break
      fi
    fi
  done
  if ! $found; then
    check_warn "Thermal zone '$tz_name' not found"
  fi
done
echo ""

# --- Summary ---
echo "=============================================="
echo " Summary"
echo "=============================================="
echo -e "  ${GREEN}PASS${NC}: $PASS"
echo -e "  ${RED}FAIL${NC}: $FAIL"
echo -e "  ${YELLOW}WARN${NC}: $WARN"
echo ""
if [ $FAIL -eq 0 ]; then
  echo -e "  ${GREEN}Environment is ready for DragonPilot Jetson port!${NC}"
else
  echo -e "  ${RED}$FAIL issue(s) must be fixed before proceeding.${NC}"
fi
echo "=============================================="

exit $FAIL

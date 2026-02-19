#!/usr/bin/env bash
#
# Install TensorRT on Jetson AGX Xavier (JetPack 5.x)
# Activates: TensorRT runner, DLA runner, FP16/INT8 inference
#
# Usage: sudo ./scripts/jetson_install_tensorrt.sh
#

set -e

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: Run as root (sudo)"
  exit 1
fi

if [ ! -f /JETSON ]; then
  echo "ERROR: Not a Jetson device (missing /JETSON marker)"
  exit 1
fi

echo "=== Jetson TensorRT Installation ==="

# Check JetPack version
if [ -f /etc/nv_tegra_release ]; then
  echo "Tegra release: $(head -1 /etc/nv_tegra_release)"
fi

# Step 1: Install TensorRT Python bindings + dev headers
echo ""
echo "[1/7] Installing TensorRT packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3-libnvinfer \
  python3-libnvinfer-dev \
  libnvinfer-bin \
  libnvinfer-samples

# Step 2: Install pycuda for CUDA memory management
echo ""
echo "[2/7] Installing pycuda..."
OPENPILOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "$OPENPILOT_DIR/.venv" ]; then
  "$OPENPILOT_DIR/.venv/bin/pip" install pycuda 2>/dev/null || \
    echo "WARNING: pycuda install failed (may need CUDA_HOME set)"
else
  pip3 install pycuda 2>/dev/null || true
fi

# Step 3: Verify TensorRT installation
echo ""
echo "[3/7] Verifying TensorRT..."
PYTHON="${OPENPILOT_DIR}/.venv/bin/python3"
[ ! -f "$PYTHON" ] && PYTHON="python3"

TRT_VERSION=$($PYTHON -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "FAILED")
if [ "$TRT_VERSION" = "FAILED" ]; then
  # TensorRT Python module may need symlink into venv
  echo "TensorRT not found in venv, creating symlink..."
  SITE_PACKAGES=$($PYTHON -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
  if [ -n "$SITE_PACKAGES" ]; then
    # Find system TensorRT
    for trt_path in /usr/lib/python3/dist-packages/tensorrt* /usr/lib/python3.8/dist-packages/tensorrt*; do
      if [ -e "$trt_path" ]; then
        ln -sf "$trt_path" "$SITE_PACKAGES/" 2>/dev/null || true
      fi
    done
    TRT_VERSION=$($PYTHON -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "FAILED")
  fi
fi

if [ "$TRT_VERSION" = "FAILED" ]; then
  echo "ERROR: TensorRT Python module not available"
  echo "Try: apt-get install python3-libnvinfer"
  exit 1
fi
echo "TensorRT version: $TRT_VERSION"

# Step 4: Verify pycuda
echo ""
echo "[4/7] Verifying pycuda..."
PYCUDA_OK=$($PYTHON -c "import pycuda.driver; pycuda.driver.init(); print('OK')" 2>/dev/null || echo "FAILED")
if [ "$PYCUDA_OK" = "FAILED" ]; then
  echo "WARNING: pycuda not available. TensorRT runner will use numpy fallback."
  echo "To fix: CUDA_HOME=/usr/local/cuda pip install pycuda"
else
  echo "pycuda: OK"
fi

# Step 5: Compile trt_runtime.so (C bridge for Python ctypes)
echo ""
echo "[5/7] Compiling trt_runtime.so..."
TRT_RUNTIME_SRC="$OPENPILOT_DIR/selfdrive/modeld/runners/trt_runtime.cpp"
TRT_RUNTIME_SO="$OPENPILOT_DIR/selfdrive/modeld/runners/trt_runtime.so"
if [ -f "$TRT_RUNTIME_SRC" ]; then
  g++ -shared -fPIC -O2 -I/usr/local/cuda/include \
    -o "$TRT_RUNTIME_SO" "$TRT_RUNTIME_SRC" \
    -lnvinfer -lcudart -L/usr/local/cuda/lib64
  if [ $? -eq 0 ]; then
    echo "trt_runtime.so compiled: $TRT_RUNTIME_SO"
  else
    echo "WARNING: trt_runtime.so compilation failed"
  fi
else
  echo "WARNING: trt_runtime.cpp not found at $TRT_RUNTIME_SRC"
fi

# Step 6: Pre-create engine cache directory
echo ""
echo "[6/7] Setting up engine cache..."
TRT_CACHE="${TRT_ENGINE_CACHE:-/data/trt_engines}"
mkdir -p "$TRT_CACHE"
chmod 777 "$TRT_CACHE"
echo "Engine cache: $TRT_CACHE"

# Step 7: Verify trt_runtime.so loads correctly
echo ""
echo "[7/7] Verifying trt_runtime.so..."
if [ -f "$TRT_RUNTIME_SO" ]; then
  TRT_LOAD=$($PYTHON -c "import ctypes; ctypes.CDLL('$TRT_RUNTIME_SO'); print('OK')" 2>/dev/null || echo "FAILED")
  echo "trt_runtime.so load: $TRT_LOAD"
else
  TRT_LOAD="NOT_BUILT"
  echo "trt_runtime.so: not built"
fi

# Summary
echo ""
echo "=== Installation Complete ==="
echo "TensorRT: $TRT_VERSION"
echo "pycuda: $PYCUDA_OK"
echo "trt_runtime.so: $TRT_LOAD"
echo "Engine cache: $TRT_CACHE"
echo ""
echo "TensorRT runner will activate automatically on next modeld start."
echo "DLA runner will activate for dmonitoring model."
echo ""
echo "To verify:"
echo "  $PYTHON -c \"from selfdrive.modeld.runners.tensorrt_runner import TensorRTRunner; print('OK')\""
echo "  $PYTHON -c \"from selfdrive.modeld.runners.dla_runner import DLARunner; print('OK')\""

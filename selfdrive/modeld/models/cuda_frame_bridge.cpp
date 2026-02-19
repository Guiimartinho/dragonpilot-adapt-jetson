/**
 * CUDA frame bridge for Jetson AGX Xavier.
 * Wraps CudaDrivingModelFrame for Python ctypes access.
 *
 * This bridge enables zero-copy GPU preprocessing:
 *   VisionBuf (host SHM) → cudaMemcpy H2D → CUDA transform → CUDA loadyuv → device ptr
 *   → tinygrad Tensor.from_blob() (zero-copy, stays on GPU)
 *
 * Eliminates the OpenCL→CPU→CUDA roundtrip:
 *   Old: cl_mem → POCL transform → clReadBuffer (D2H) → numpy → Tensor (H2D)
 *   New: host ptr → cudaMemcpy (H2D) → CUDA kernels → device ptr (stays on GPU)
 */

#ifdef __JETSON__

#include <cstring>
#include <cassert>
#include <cstdint>

#include "selfdrive/modeld/models/commonmodel_cuda.h"

static constexpr int MAX_FRAMES = 8;
static CudaDrivingModelFrame* g_frames[MAX_FRAMES] = {};
static int g_next_id = 0;

extern "C" {

int cuda_driving_frame_create(int temporal_skip) {
  int id = g_next_id++;
  assert(id < MAX_FRAMES);
  g_frames[id] = new CudaDrivingModelFrame(temporal_skip);
  return id;
}

void cuda_driving_frame_destroy(int id) {
  if (id >= 0 && id < MAX_FRAMES && g_frames[id]) {
    delete g_frames[id];
    g_frames[id] = nullptr;
  }
}

/**
 * Prepare frame: upload YUV from host, run CUDA transform + loadyuv.
 * Returns CUDA device pointer to preprocessed frame data.
 * The pointer is stable (same address every call) and valid until destroy().
 */
uint64_t cuda_driving_frame_prepare(
    int id,
    const uint8_t* yuv_data,
    int width, int height, int stride, int uv_offset,
    const float* projection)
{
  mat3 m;
  memcpy(m.v, projection, 9 * sizeof(float));
  g_frames[id]->prepare(yuv_data, width, height, stride, uv_offset, m);
  return (uint64_t)(uintptr_t)g_frames[id]->get_device_ptr();
}

uint64_t cuda_driving_frame_get_device_ptr(int id) {
  return (uint64_t)(uintptr_t)g_frames[id]->get_device_ptr();
}

int cuda_driving_frame_get_buf_size(int id) {
  return g_frames[id]->buf_size;
}

void cuda_driving_frame_get_host_buffer(int id, uint8_t* host_dst, int size) {
  uint8_t* src = g_frames[id]->get_host_buffer(size);
  memcpy(host_dst, src, size);
}

}  // extern "C"

#endif  // __JETSON__

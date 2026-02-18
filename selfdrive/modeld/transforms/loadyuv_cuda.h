#pragma once

/**
 * CUDA-native YUV loading declarations for Jetson AGX Xavier.
 * Replaces OpenCL loadys/loaduv/copy kernels with direct CUDA calls.
 *
 * Kernels defined in loadyuv.cu, compiled with nvcc.
 */

#ifdef __JETSON__

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_loadys(const uint8_t* Y, uint8_t* out, int out_offset,
                 int transformed_width, int transformed_height, cudaStream_t stream);

void cuda_loaduv(const uint8_t* in, uint8_t* out, int out_offset,
                 int total_bytes, cudaStream_t stream);

void cuda_copy(const uint8_t* in, uint8_t* out, int in_offset, int out_offset,
               int total_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

/**
 * CUDA-native LoadYUV state (replaces OpenCL LoadYUVState).
 */
struct CudaLoadYUVState {
  int width;
  int height;
  cudaStream_t stream;
  bool owns_stream;

  void init(int w, int h, cudaStream_t ext_stream = nullptr) {
    width = w;
    height = h;
    if (ext_stream) {
      stream = ext_stream;
      owns_stream = false;
    } else {
      cudaStreamCreate(&stream);
      owns_stream = true;
    }
  }

  void destroy() {
    if (owns_stream) {
      cudaStreamDestroy(stream);
    }
  }

  void queue(const uint8_t* d_y, const uint8_t* d_u, const uint8_t* d_v, uint8_t* d_out) {
    int out_offset = 0;

    // Load Y with interleave
    cuda_loadys(d_y, d_out, out_offset, width, height, stream);

    // Load U
    int uv_size = (width / 2) * (height / 2);
    out_offset = width * height;
    cuda_loaduv(d_u, d_out, out_offset, uv_size, stream);

    // Load V
    out_offset += uv_size;
    cuda_loaduv(d_v, d_out, out_offset, uv_size, stream);
  }

  void copy(const uint8_t* d_src, uint8_t* d_dst, int src_off, int dst_off, int size) {
    cuda_copy(d_src, d_dst, src_off, dst_off, size, stream);
  }
};

#endif  // __JETSON__

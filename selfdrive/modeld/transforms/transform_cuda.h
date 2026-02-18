#pragma once

/**
 * CUDA-native transform declarations for Jetson AGX Xavier.
 * Replaces OpenCL warpPerspective with direct CUDA kernel calls,
 * eliminating POCL interop overhead (~12ms saved).
 *
 * Kernels defined in transform.cu, compiled with nvcc.
 */

#ifdef __JETSON__

#include <cuda_runtime.h>
#include <cstdint>

#include "common/mat.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_warp_perspective(
    const uint8_t* src, int src_row_stride, int src_px_stride, int src_offset, int src_rows, int src_cols,
    uint8_t* dst, int dst_row_stride, int dst_offset, int dst_rows, int dst_cols,
    const float* M, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

/**
 * CUDA-native Transform state (replaces OpenCL Transform struct).
 * Uses CUDA device memory and streams instead of cl_mem and cl_command_queue.
 */
struct CudaTransform {
  float* d_m_y;   // Device memory for Y projection matrix (3x3)
  float* d_m_uv;  // Device memory for UV projection matrix (3x3)
  cudaStream_t stream;
  bool owns_stream;

  void init(cudaStream_t ext_stream = nullptr) {
    cudaMalloc(&d_m_y, 9 * sizeof(float));
    cudaMalloc(&d_m_uv, 9 * sizeof(float));
    if (ext_stream) {
      stream = ext_stream;
      owns_stream = false;
    } else {
      cudaStreamCreate(&stream);
      owns_stream = true;
    }
  }

  void destroy() {
    cudaFree(d_m_y);
    cudaFree(d_m_uv);
    if (owns_stream) {
      cudaStreamDestroy(stream);
    }
  }

  void queue(
      const uint8_t* d_in_yuv, int in_width, int in_height, int in_stride, int in_uv_offset,
      uint8_t* d_out_y, uint8_t* d_out_u, uint8_t* d_out_v,
      int out_width, int out_height,
      const mat3& projection) {

    // Y plane transform
    mat3 proj_y = projection;
    cudaMemcpyAsync(d_m_y, proj_y.v, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);

    cuda_warp_perspective(
        d_in_yuv, in_stride, 1, 0, in_height, in_width,
        d_out_y, out_width, 0, out_height, out_width,
        d_m_y, stream);

    // UV plane transform (half resolution)
    mat3 proj_uv = transform_scale_buffer(projection, 0.5);
    cudaMemcpyAsync(d_m_uv, proj_uv.v, 9 * sizeof(float), cudaMemcpyHostToDevice, stream);

    int in_uv_width = in_width / 2;
    int in_uv_height = in_height / 2;
    int out_uv_width = out_width / 2;
    int out_uv_height = out_height / 2;

    // U plane
    cuda_warp_perspective(
        d_in_yuv, in_stride, 2, in_uv_offset, in_uv_height, in_uv_width,
        d_out_u, out_uv_width, 0, out_uv_height, out_uv_width,
        d_m_uv, stream);

    // V plane
    cuda_warp_perspective(
        d_in_yuv, in_stride, 2, in_uv_offset + 1, in_uv_height, in_uv_width,
        d_out_v, out_uv_width, 0, out_uv_height, out_uv_width,
        d_m_uv, stream);
  }
};

#endif  // __JETSON__

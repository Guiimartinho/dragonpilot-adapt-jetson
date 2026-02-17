/**
 * Zero-copy VisionBuf implementation for Jetson AGX Xavier.
 *
 * Uses shared memory (shm) for inter-process VisionIPC communication,
 * combined with CUDA pinned memory (cudaHostRegister) for zero-copy
 * GPU access without explicit memcpy.
 *
 * Memory flow:
 *   shm mmap → cudaHostRegister (page-locked) → CL_MEM_USE_HOST_PTR → GPU
 *   All processes access the same physical pages. GPU accesses are DMA-direct.
 */

#include "msgq/visionipc/visionbuf.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

#ifdef __JETSON__
#include <cuda_runtime.h>
#endif

static std::atomic<int> offset(0);

static void *malloc_with_fd(size_t len, int *fd) {
  char full_path[0x100];
  snprintf(full_path, sizeof(full_path) - 1, "/dev/shm/msgq_visionbuf_%d_%d", getpid(), offset++);

  *fd = open(full_path, O_RDWR | O_CREAT, 0664);
  assert(*fd >= 0);
  unlink(full_path);
  ftruncate(*fd, len);

  void *addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
  assert(addr != MAP_FAILED);
  return addr;
}

void VisionBuf::allocate(size_t length) {
  this->len = length;
  this->mmap_len = this->len + sizeof(uint64_t);
  this->addr = malloc_with_fd(this->mmap_len, &this->fd);
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);

#ifdef __JETSON__
  // Pin the shared memory pages so GPU can DMA directly without staging copies.
  // cudaHostRegister makes existing host memory page-locked (pinned).
  // This enables zero-copy transfers: clEnqueueMapBuffer returns direct pointer.
  cudaError_t err = cudaHostRegister(this->addr, this->mmap_len,
                                     cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    // Non-fatal: falls back to standard buffered transfers
    fprintf(stderr, "visionbuf_jetson: cudaHostRegister failed (%s), using unpinned memory\n",
            cudaGetErrorString(err));
  }
#endif
}

void VisionBuf::import() {
  assert(this->fd >= 0);
  this->addr = mmap(NULL, this->mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
  assert(this->addr != MAP_FAILED);
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);

#ifdef __JETSON__
  // Also pin imported buffers for zero-copy GPU access
  cudaHostRegister(this->addr, this->mmap_len, cudaHostRegisterDefault);
  // Ignore errors: not all imported buffers need GPU access
#endif
}

void VisionBuf::init_cl(cl_device_id device_id, cl_context ctx) {
  int err;
  this->copy_q = clCreateCommandQueue(ctx, device_id, 0, &err);
  assert(err == 0);

  // CL_MEM_USE_HOST_PTR: OpenCL shares the pinned host memory.
  // On Jetson with POCL, this avoids an extra copy for GPU-bound data.
  this->buf_cl = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                this->len, this->addr, &err);
  assert(err == 0);
}

int VisionBuf::sync(int dir) {
  int err = 0;
  if (!this->buf_cl) return 0;

  if (dir == VISIONBUF_SYNC_FROM_DEVICE) {
    err = clEnqueueReadBuffer(this->copy_q, this->buf_cl, CL_FALSE, 0,
                              this->len, this->addr, 0, NULL, NULL);
  } else {
    err = clEnqueueWriteBuffer(this->copy_q, this->buf_cl, CL_FALSE, 0,
                               this->len, this->addr, 0, NULL, NULL);
  }
  if (err == 0) {
    err = clFinish(this->copy_q);
  }
  return err;
}

int VisionBuf::free() {
  int err = 0;

  if (this->buf_cl) {
    err = clReleaseMemObject(this->buf_cl);
    if (err != 0) return err;
    err = clReleaseCommandQueue(this->copy_q);
    if (err != 0) return err;
  }

#ifdef __JETSON__
  // Unpin before unmapping
  if (this->addr != nullptr && this->addr != MAP_FAILED) {
    cudaHostUnregister(this->addr);
  }
#endif

  if (this->addr != nullptr && this->addr != MAP_FAILED) {
    err = munmap(this->addr, this->mmap_len);
    if (err != 0) return err;
  }

  if (this->fd >= 0) {
    err = close(this->fd);
  }
  return err;
}

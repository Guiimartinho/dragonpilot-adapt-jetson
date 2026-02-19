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
  // Pin the shared memory pages with cudaHostRegisterMapped so the GPU can
  // access host memory directly via DMA without explicit cudaMemcpy.
  // cudaHostGetDevicePointer provides the GPU-visible address.
  cudaError_t err = cudaHostRegister(this->addr, this->mmap_len,
                                     cudaHostRegisterMapped);
  if (err != cudaSuccess) {
    // Non-fatal: falls back to standard buffered transfers
    fprintf(stderr, "visionbuf_jetson: cudaHostRegister(Mapped) failed (%s), using unpinned memory\n",
            cudaGetErrorString(err));
  } else {
    // Get device pointer for GPU-direct access (true zero-copy on Jetson unified memory)
    void* d_ptr = nullptr;
    err = cudaHostGetDevicePointer(&d_ptr, this->addr, 0);
    if (err == cudaSuccess && d_ptr != nullptr) {
      this->d_addr = d_ptr;
    }
  }
#endif
}

void VisionBuf::import() {
  assert(this->fd >= 0);
  this->addr = mmap(NULL, this->mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, this->fd, 0);
  assert(this->addr != MAP_FAILED);
  this->frame_id = (uint64_t*)((uint8_t*)this->addr + this->len);

#ifdef __JETSON__
  // Also pin imported buffers for zero-copy GPU access (mapped for GPU-direct)
  cudaError_t err = cudaHostRegister(this->addr, this->mmap_len, cudaHostRegisterMapped);
  if (err == cudaSuccess) {
    void* d_ptr = nullptr;
    if (cudaHostGetDevicePointer(&d_ptr, this->addr, 0) == cudaSuccess && d_ptr != nullptr) {
      this->d_addr = d_ptr;
    }
  }
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
  // Unpin before unmapping (invalidates d_addr)
  if (this->addr != nullptr && this->addr != MAP_FAILED) {
    cudaError_t unreg_err = cudaHostUnregister(this->addr);
    if (unreg_err != cudaSuccess && unreg_err != cudaErrorHostMemoryNotRegistered) {
      fprintf(stderr, "visionbuf_jetson: cudaHostUnregister failed: %s\n", cudaGetErrorString(unreg_err));
    }
    this->d_addr = nullptr;
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

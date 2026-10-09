// gpu_common.h: CUDA error checking, integrated-GPU detection, and paired host/device
// allocation (mapped zero-copy memory on integrated GPUs, pinned host + device memory otherwise).

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Detects whether the given GPU device has a unified (integrated) memory
// architecture (e.g. Jetson Orin) vs. a discrete one (e.g. L40S/A5000).
// Callers store the result in their own local/global -- this function has
// no side effects beyond the query itself.
inline bool detect_integrated_gpu(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return prop.integrated != 0;
}

// Allocates a host+device pointer pair for `count` elements of type T, using
// the platform-appropriate strategy:
//   - integrated GPU: true zero-copy via cudaHostAlloc(cudaHostAllocMapped) +
//     cudaHostGetDevicePointer -- the host buffer IS the device-visible
//     buffer, nothing to transfer.
//   - discrete GPU: a real pinned host allocation (cudaMallocHost) plus a
//     separate device allocation (cudaMalloc); the caller is responsible for
//     the explicit cudaMemcpy(Async) to move data between them.
// `integrated` is passed explicitly (not read from a global) so this
// function has no hidden dependency on any particular translation unit's
// g_integrated variable -- every caller already has its own copy of that
// flag from calling detect_integrated_gpu() once at startup.
template <typename T>
inline void alloc_pair(T** h_ptr, T** d_ptr, size_t count, bool integrated) {
    if (integrated) {
        CUDA_CHECK(cudaHostAlloc(h_ptr, count * sizeof(T), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(d_ptr, *h_ptr, 0));
    } else {
        CUDA_CHECK(cudaMallocHost(h_ptr, count * sizeof(T)));
        CUDA_CHECK(cudaMalloc(d_ptr, count * sizeof(T)));
    }
}

// Variant used where the caller has the data ready up front and wants it
// uploaded immediately (as opposed to alloc_pair's allocate-now-fill-later
// shape, used by per-round buffers). `h_view_out` receives the host-visible
// pointer on integrated GPUs (nullptr on discrete, since there nothing
// aliases the device buffer -- the data lives only on the device after this
// call, matching the original single-shot-upload call sites this replaces).
template <typename T>
inline void alloc_and_upload(T** d_ptr, const std::vector<T>& h_data, T** h_view_out, bool integrated) {
    size_t bytes = h_data.size() * sizeof(T);
    if (integrated) {
        CUDA_CHECK(cudaHostAlloc(h_view_out, bytes, cudaHostAllocMapped));
        memcpy(*h_view_out, h_data.data(), bytes);
        CUDA_CHECK(cudaHostGetDevicePointer(d_ptr, *h_view_out, 0));
    } else {
        *h_view_out = nullptr;
        CUDA_CHECK(cudaMalloc(d_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(*d_ptr, h_data.data(), bytes, cudaMemcpyHostToDevice));
    }
}

// CUDA error checks, integrated-GPU detection, and host/device buffer pairs: zero-copy on
// integrated GPUs (Jetson), pinned host + device memory on discrete ones.

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

inline bool detect_integrated_gpu(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    return prop.integrated != 0;
}

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

// Reset / read back a per-round device counter (a host write on integrated GPUs).

#pragma once
#include "gpu_common.h"

inline void reset_scalar_count(int* d_count_ptr, int* h_count_view, bool integrated) {
    if (integrated) {
        *h_count_view = 0;
    } else {
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_count_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }
}

inline int read_scalar_count_back(int* d_count_ptr, int* h_count_view, bool integrated) {
    if (integrated) return *h_count_view;
    int v = 0;
    CUDA_CHECK(cudaMemcpy(&v, d_count_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    return v;
}

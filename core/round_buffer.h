// round_buffer.h: helpers for per-round GPU buffers (allocation and reading back a
// device-side counter) shared by BFS and coloring.

#pragma once
#include "gpu_common.h"

// Resets a device-resident scalar round-counter to 0 before a round starts.
// Async on discrete GPUs (no need to block the CPU on a 4-byte H2D copy,
// same reasoning as every other round-scoped transfer in this project); a
// plain host write on integrated GPUs (h_count_view aliases d_count_ptr via
// cudaHostGetDevicePointer -- the GPU sees the write with no transfer to
// make async in the first place).
inline void reset_scalar_count(int* d_count_ptr, int* h_count_view, bool integrated) {
    if (integrated) {
        *h_count_view = 0;
    } else {
        int zero = 0;
        CUDA_CHECK(cudaMemcpyAsync(d_count_ptr, &zero, sizeof(int), cudaMemcpyHostToDevice));
    }
}

// Reads that same scalar back after a round's kernels have run. A plain
// host read on integrated GPUs (already visible, no transfer); a real,
// blocking D2H cudaMemcpy on discrete GPUs (the round's next step needs
// this value before it can proceed, so there's nothing to overlap it with).
inline int read_scalar_count_back(int* d_count_ptr, int* h_count_view, bool integrated) {
    if (integrated) return *h_count_view;
    int v = 0;
    CUDA_CHECK(cudaMemcpy(&v, d_count_ptr, sizeof(int), cudaMemcpyDeviceToHost));
    return v;
}

// tools/bfs_ground_truth.cu
//
// GPU level-synchronous BFS ground-truth generator, reading the project's
// .egr binary CSR format directly (same layout FlashGraph.h/ECLgraph.h
// already use). Plays the same role for dynamic BFS that ECL-MIS plays for
// MIS: a trusted, simple reference solver whose output seeds and verifies
// the dynamic repair algorithm (algorithms/bfs/scout_bfs.cu), not a
// performance-critical hot path itself, so it favors a straightforward,
// easy-to-verify level-synchronous design over the more aggressive
// frontier-compaction tricks a hyper-optimized BFS would use.
//
// Level-synchronous shape (one kernel launch per BFS level, each thread
// checks whether its own vertex belongs to the current frontier and if so
// relaxes its neighbors) is adapted from the classical GPU BFS approach
// referenced in mirawara/Parallel-BFS's bfs_gpu kernel, reimplemented here
// against our own CSR layout, since every tool in this project reads .egr natively.
//
// USAGE: bfs_ground_truth <graph.egr> <output.txt> [source_node=0]
// Output: one "<node_id> <level>" line per node (1-indexed node_id, to
// match this project's existing MIS/coloring seed-file convention),
// unreached nodes get level -1.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__global__ void bfs_level_kernel(int nodes, const long long* __restrict__ nindex,
                                  const int* __restrict__ nlist, int* level,
                                  int current_level, int* changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nodes) return;
    if (level[v] != current_level) return;

    long long s = nindex[v], e = nindex[v + 1];
    for (long long i = s; i < e; i++) {
        int u = nlist[i];
        if (u < 0 || u >= nodes) continue;
        if (level[u] == -1) {
            if (atomicCAS(&level[u], -1, current_level + 1) == -1) {
                *changed = 1;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "USAGE: %s graph.egr output.txt [source_node=0]\n", argv[0]);
        return 1;
    }
    const char* in_path = argv[1];
    const char* out_path = argv[2];
    int source_node = (argc >= 4) ? atoi(argv[3]) : 0;

    FILE* f = fopen(in_path, "rb");
    if (!f) { fprintf(stderr, "ERROR: could not open %s\n", in_path); return 1; }

    int nodes, sentinel;
    if (fread(&nodes, sizeof(int), 1, f) != 1) { fprintf(stderr, "ERROR: read nodes\n"); return 1; }
    if (fread(&sentinel, sizeof(int), 1, f) != 1) { fprintf(stderr, "ERROR: read sentinel\n"); return 1; }

    long long edges;
    long long* h_nindex = (long long*)malloc((size_t)(nodes + 1) * sizeof(long long));
    if (sentinel == -1) {
        if (fread(&edges, sizeof(long long), 1, f) != 1) { fprintf(stderr, "ERROR: read edges (wide)\n"); return 1; }
        if (fread(h_nindex, sizeof(long long), (size_t)(nodes + 1), f) != (size_t)(nodes + 1)) {
            fprintf(stderr, "ERROR: read nindex (wide)\n"); return 1;
        }
    } else {
        edges = (long long)sentinel;
        int* tmp = (int*)malloc((size_t)(nodes + 1) * sizeof(int));
        if (fread(tmp, sizeof(int), (size_t)(nodes + 1), f) != (size_t)(nodes + 1)) {
            fprintf(stderr, "ERROR: read nindex (narrow)\n"); return 1;
        }
        for (int i = 0; i <= nodes; i++) h_nindex[i] = (long long)tmp[i];
        free(tmp);
    }
    int* h_nlist = (int*)malloc((size_t)edges * sizeof(int));
    if (fread(h_nlist, sizeof(int), (size_t)edges, f) != (size_t)edges) {
        fprintf(stderr, "ERROR: read nlist\n"); return 1;
    }
    fclose(f);

    if (source_node < 0 || source_node >= nodes) {
        fprintf(stderr, "ERROR: source_node %d out of range [0,%d)\n", source_node, nodes);
        return 1;
    }

    printf("bfs_ground_truth: %d nodes, %lld edges, source=%d\n", nodes, edges, source_node);

    long long *d_nindex; int *d_nlist, *d_level, *d_changed;
    CUDA_CHECK(cudaMalloc(&d_nindex, (size_t)(nodes + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_nlist, (size_t)edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_level, (size_t)nodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_nindex, h_nindex, (size_t)(nodes + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nlist, h_nlist, (size_t)edges * sizeof(int), cudaMemcpyHostToDevice));

    int* h_level = (int*)malloc((size_t)nodes * sizeof(int));
    for (int i = 0; i < nodes; i++) h_level[i] = -1;
    h_level[source_node] = 0;
    CUDA_CHECK(cudaMemcpy(d_level, h_level, (size_t)nodes * sizeof(int), cudaMemcpyHostToDevice));

    auto t0 = std::chrono::high_resolution_clock::now();
    int threads = 256, blocks = (nodes + threads - 1) / threads;
    int current_level = 0;
    int changed = 1;
    while (changed) {
        changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice));
        bfs_level_kernel<<<blocks, threads>>>(nodes, d_nindex, d_nlist, d_level, current_level, d_changed);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        current_level++;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("BFS converged in %d levels, %.3f ms\n", current_level,
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    CUDA_CHECK(cudaMemcpy(h_level, d_level, (size_t)nodes * sizeof(int), cudaMemcpyDeviceToHost));

    long long reached = 0;
    for (int i = 0; i < nodes; i++) if (h_level[i] >= 0) reached++;
    printf("Reached %lld / %d nodes (%.1f%%)\n", reached, nodes, 100.0 * reached / nodes);

    std::ofstream out(out_path);
    if (!out.is_open()) { fprintf(stderr, "ERROR: could not open %s for writing\n", out_path); return 1; }
    for (int i = 0; i < nodes; i++) {
        // 1-indexed node id, matching this project's MIS/coloring seed-file convention.
        out << (i + 1) << " " << h_level[i] << "\n";
    }
    out.close();
    printf("Wrote %s\n", out_path);

    cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_level); cudaFree(d_changed);
    free(h_nindex); free(h_nlist); free(h_level);
    return 0;
}

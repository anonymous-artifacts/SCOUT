// ExCC v1: Connected Components on external memory architecture (batched CSR streaming)
// Implements union-find with init → merge passes → flatten,
// using host-side graph and double-buffered batch H2D.
//
// Compile: nvcc -O3 -std=c++11 -arch=sm_86 excc_v1.cu -o excc

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <set>

using namespace std::chrono;

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

// GPU memory tracking
static double g_baseline_used_mb = -1.0;
static double g_peak_delta_mb    = 0.0;
static double g_gpu_total_mb     = 0.0;

static void gpu_memory_set_baseline() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        g_gpu_total_mb     = (double)total_bytes / (1024.0 * 1024.0);
        g_baseline_used_mb = g_gpu_total_mb - (double)free_bytes / (1024.0 * 1024.0);
    }
}

static void print_gpu_memory_usage(const char* label) {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) return;
    double total_mb = (double)total_bytes / (1024.0 * 1024.0);
    double used_mb  = total_mb - (double)free_bytes / (1024.0 * 1024.0);
    double delta_mb = used_mb - g_baseline_used_mb;
    if (delta_mb < 0.0) delta_mb = 0.0;
    if (delta_mb > g_peak_delta_mb) g_peak_delta_mb = delta_mb;
    g_gpu_total_mb = total_mb;
}

// ---------------------------------------------------------------------------
// Graph (host-side CSR)
// ---------------------------------------------------------------------------
typedef struct {
    int       nodes;
    long long edges;
    long long* nindex;
    int*      nlist;  // pinned
} graph;

static graph readgraph(const char* fname) {
    graph g;
    FILE* f = fopen(fname, "rb");
    if (!f) { fprintf(stderr, "ERROR: could not open file %s\n", fname); exit(1); }
    if (fread(&g.nodes, sizeof(int), 1, f) != 1) { fprintf(stderr, "ERROR: reading nodes\n"); exit(1); }
    int edges_i32;
    if (fread(&edges_i32, sizeof(int), 1, f) != 1) { fprintf(stderr, "ERROR: reading edges\n"); exit(1); }
    if (edges_i32 == -1) {
        if (fread(&g.edges, sizeof(long long), 1, f) != 1) { fprintf(stderr, "ERROR: reading edges (64-bit)\n"); exit(1); }
        g.nindex = (long long*)malloc((size_t)(g.nodes + 1) * sizeof(long long));
        if (cudaMallocHost(&g.nlist, (size_t)g.edges * sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMallocHost for nlist (%.2f GB)\n", (double)g.edges * sizeof(int) / (1024.0*1024.0*1024.0));
            exit(1);
        }
        if (!g.nindex) { fprintf(stderr, "ERROR: malloc\n"); exit(1); }
        if (fread(g.nindex, sizeof(long long), g.nodes + 1, f) != (size_t)(g.nodes + 1)) { fprintf(stderr, "ERROR: reading nindex\n"); exit(1); }
        if (fread(g.nlist, sizeof(int), g.edges, f) != (size_t)g.edges) { fprintf(stderr, "ERROR: reading nlist\n"); exit(1); }
    } else {
        g.edges = edges_i32;
        int* nindex_i32 = (int*)malloc((size_t)(g.nodes + 1) * sizeof(int));
        g.nindex = (long long*)malloc((size_t)(g.nodes + 1) * sizeof(long long));
        if (cudaMallocHost(&g.nlist, (size_t)g.edges * sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaMallocHost for nlist\n"); exit(1);
        }
        if (!nindex_i32 || !g.nindex) { fprintf(stderr, "ERROR: malloc\n"); exit(1); }
        if (fread(nindex_i32, sizeof(int), g.nodes + 1, f) != (size_t)(g.nodes + 1)) { fprintf(stderr, "ERROR: reading nindex\n"); exit(1); }
        for (int i = 0; i <= g.nodes; ++i) g.nindex[i] = nindex_i32[i];
        free(nindex_i32);
        if (fread(g.nlist, sizeof(int), g.edges, f) != (size_t)g.edges) { fprintf(stderr, "ERROR: reading nlist\n"); exit(1); }
    }
    fclose(f);
    return g;
}

static void freegraph(graph g) { free(g.nindex); cudaFreeHost(g.nlist); }

// Symmetrize for undirected CC
static void auto_symmetrize(graph& g) {
    const int n = g.nodes;
    const long long m = g.edges;
    for (int u = 0; u < n; ++u)
        std::sort(g.nlist + (size_t)g.nindex[u], g.nlist + (size_t)g.nindex[u + 1]);
    bool need_sym = false;
    int checks = 0;
    for (int u = 0; u < n && checks < 2000; ++u) {
        for (long long i = g.nindex[u]; i < g.nindex[u + 1] && checks < 2000; ++i) {
            int v = g.nlist[(size_t)i];
            if (v == u) continue;
            if (!std::binary_search(g.nlist + (size_t)g.nindex[v], g.nlist + (size_t)g.nindex[v + 1], u)) {
                need_sym = true;
                break;
            }
            ++checks;
        }
        if (need_sym) break;
    }
    if (!need_sym) {
        printf("Symmetry check: PASSED (sampled %d edges)\n", checks);
        return;
    }
    printf("Symmetry check: FAILED – symmetrizing...\n");
    std::vector<int> extra(n, 0);
    long long total_extra = 0;
    for (int u = 0; u < n; ++u) {
        for (long long i = g.nindex[u]; i < g.nindex[u + 1]; ++i) {
            int v = g.nlist[(size_t)i];
            if (v == u) continue;
            if (!std::binary_search(g.nlist + (size_t)g.nindex[v], g.nlist + (size_t)g.nindex[v + 1], u)) {
                extra[v]++;
                total_extra++;
            }
        }
    }
    long long new_m = m + total_extra;
    long long* new_nindex = (long long*)malloc((size_t)(n + 1) * sizeof(long long));
    int* new_nlist = NULL;
    if (cudaMallocHost(&new_nlist, (size_t)new_m * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMallocHost for symmetrized nlist\n"); exit(1);
    }
    if (!new_nindex) { fprintf(stderr, "ERROR: malloc new_nindex\n"); exit(1); }
    new_nindex[0] = 0;
    for (int u = 0; u < n; ++u)
        new_nindex[u + 1] = new_nindex[u] + (g.nindex[u + 1] - g.nindex[u]) + extra[u];
    std::vector<long long> wpos(n);
    for (int u = 0; u < n; ++u) {
        long long orig = g.nindex[u + 1] - g.nindex[u];
        memcpy(new_nlist + (size_t)new_nindex[u], g.nlist + (size_t)g.nindex[u], (size_t)orig * sizeof(int));
        wpos[u] = new_nindex[u] + orig;
    }
    for (int u = 0; u < n; ++u) {
        for (long long i = g.nindex[u]; i < g.nindex[u + 1]; ++i) {
            int v = g.nlist[(size_t)i];
            if (v == u) continue;
            if (!std::binary_search(g.nlist + (size_t)g.nindex[v], g.nlist + (size_t)g.nindex[v + 1], u))
                new_nlist[(size_t)(wpos[v]++)] = u;
        }
    }
    for (int u = 0; u < n; ++u)
        std::sort(new_nlist + (size_t)new_nindex[u], new_nlist + (size_t)new_nindex[u + 1]);
    free(g.nindex);
    cudaFreeHost(g.nlist);
    g.nindex = new_nindex;
    g.nlist  = new_nlist;
    g.edges  = new_m;
    printf("  Symmetrized: %d nodes, %lld edges\n", g.nodes, g.edges);
}

struct BatchInfo { int row_start, n_rows, edge_count; };

static void compute_local_nindex(const graph& g, const BatchInfo& info, int* pinned_nindex) {
    long long base = g.nindex[info.row_start];
    for (int r = 0; r <= info.n_rows; ++r)
        pinned_nindex[r] = (int)(g.nindex[info.row_start + r] - base);
}

// ---------------------------------------------------------------------------
// Device helper: representative with path compression
// ---------------------------------------------------------------------------
__device__ __forceinline__ int representative_cc(int idx, int* nstat) {
    int curr = nstat[idx];
    if (curr != idx) {
        int next, prev = idx;
        while (curr > (next = nstat[curr])) {
            nstat[prev] = next;
            prev = curr;
            curr = next;
        }
    }
    return curr;
}

// ---------------------------------------------------------------------------
// Init batch: nstat[v] = min(v, min(neighbors)) for v in batch
// ---------------------------------------------------------------------------
__global__ void init_cc_batch(
    const int* __restrict__ nindex,
    const int* __restrict__ nlist,
    int*       __restrict__ d_nstat,
    int        row_offset,
    int        n_rows
) {
    int v_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_local >= n_rows) return;
    int v = row_offset + v_local;
    int m = v;
    int start = nindex[v_local];
    int end   = nindex[v_local + 1];
    for (int i = start; i < end; ++i) {
        int u = nlist[i];
        if (u < m) m = u;
    }
    d_nstat[v] = m;
}

// ---------------------------------------------------------------------------
// Merge batch: for each v in batch, for each neighbor u with v>u, merge components
// ---------------------------------------------------------------------------
__global__ void merge_cc_batch(
    const int* __restrict__ nindex,
    const int* __restrict__ nlist,
    int*       __restrict__ d_nstat,
    int*       __restrict__ d_changed,
    int        row_offset,
    int        n_rows
) {
    int v_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_local >= n_rows) return;
    int v = row_offset + v_local;
    int vstat = representative_cc(v, d_nstat);
    int start = nindex[v_local];
    int end   = nindex[v_local + 1];
    for (int i = start; i < end; ++i) {
        int nli = nlist[i];
        if (v <= nli) continue;
        int ostat = representative_cc(nli, d_nstat);
        bool repeat;
        do {
            repeat = false;
            if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                    ret = atomicCAS(&d_nstat[ostat], ostat, vstat);
                    if (ret != ostat) {
                        ostat = ret;
                        repeat = true;
                    } else {
                        atomicExch(d_changed, 1);
                    }
                } else {
                    ret = atomicCAS(&d_nstat[vstat], vstat, ostat);
                    if (ret != vstat) {
                        vstat = ret;
                        repeat = true;
                    } else {
                        atomicExch(d_changed, 1);
                        vstat = ostat;  // new root is ostat
                    }
                }
            }
        } while (repeat);
    }
}

// ---------------------------------------------------------------------------
// Flatten batch: for each v in batch, follow nstat to root, nstat[v] = root
// ---------------------------------------------------------------------------
__global__ void flatten_cc_batch(
    const int* __restrict__ nindex,
    const int* __restrict__ nlist,
    int*       __restrict__ d_nstat,
    int        row_offset,
    int        n_rows
) {
    (void)nindex;
    (void)nlist;
    int v_local = blockIdx.x * blockDim.x + threadIdx.x;
    if (v_local >= n_rows) return;
    int v = row_offset + v_local;
    int next, vstat = d_nstat[v];
    const int old = vstat;
    while (vstat > (next = d_nstat[vstat])) {
        vstat = next;
    }
    if (old != vstat) d_nstat[v] = vstat;
}

// ---------------------------------------------------------------------------
// Validation (host)
// ---------------------------------------------------------------------------
static bool validate_cc(const graph& g, const std::vector<int>& nstat, int n) {
    for (int u = 0; u < n; ++u) {
        for (long long i = g.nindex[u]; i < g.nindex[u + 1]; ++i) {
            int v = g.nlist[(size_t)i];
            if (nstat[u] != nstat[v]) {
                printf("ERROR: edge (%d,%d) in different components %d vs %d\n", u, v, nstat[u], nstat[v]);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    printf("ExCC v1 (Connected Components, batched CSR streaming)\n");
    fflush(stdout);

    if (argc < 2 || argc > 3) {
        printf("Usage: %s <graph.egr> [batch_budget_mb]\n", argv[0]);
        return 1;
    }
    int user_batch_mb = (argc == 3) ? atoi(argv[2]) : 0;

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaGetDeviceProperties failed\n");
        return 1;
    }
    printf("CUDA device: %s\n", prop.name);
    fflush(stdout);

    printf("Loading graph from %s...\n", argv[1]);
    fflush(stdout);
    graph g = readgraph(argv[1]);
    const int n = g.nodes;
    printf("Graph: %d nodes, %lld edges\n", n, g.edges);
    fflush(stdout);

    // Comment this par if you are are sure, your graph is symmetric
    // This part takes time.
    printf("Checking symmetry...\n");
    fflush(stdout);
    auto_symmetrize(g);

    auto t_e2e_start = high_resolution_clock::now();
    gpu_memory_set_baseline();

    int* d_nstat   = NULL;
    int* d_changed = NULL;
    CUDA_CHECK(cudaMalloc(&d_nstat,   sizeof(int) * (size_t)n));
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t batch_budget = (user_batch_mb > 0)
        ? (size_t)user_batch_mb * 1024ULL * 1024ULL
        : (size_t)((double)free_bytes * 0.70);
    int MAX_ROWS = std::min(n, 1 << 20);
    size_t nindex_cost = 2 * (size_t)(MAX_ROWS + 1) * sizeof(int);
    size_t nlist_budget = (batch_budget > nindex_cost) ? (batch_budget - nindex_cost) : 0;
    long long max_edges_ll = (long long)(nlist_budget / (2 * sizeof(int)));
    long long upper_cap = std::min((long long)(1 << 28), (long long)g.edges);
    int MAX_EDGES = (int)std::min(max_edges_ll, upper_cap);
    double avg_deg = (n > 0) ? (double)g.edges / n : 0.0;
    int max_degree = 0;
    for (int u = 0; u < n; ++u) {
        long long d = g.nindex[u + 1] - g.nindex[u];
        if (d > max_degree) max_degree = (int)(d < INT_MAX ? d : INT_MAX);
    }
    if (MAX_EDGES < max_degree + 1) MAX_EDGES = max_degree + 1;
    if (avg_deg < 8.0) {
        long long sparse_cap = (long long)(MAX_ROWS * avg_deg * 2.0);
        if (sparse_cap < max_degree + 1) sparse_cap = max_degree + 1;
        if (MAX_EDGES > (int)std::min(sparse_cap, (long long)INT_MAX))
            MAX_EDGES = (int)std::min(sparse_cap, (long long)INT_MAX);
    }
    if (MAX_EDGES < (1 << 16)) MAX_EDGES = (int)std::min((long long)(1 << 16), g.edges);
    printf("Batch: MAX_ROWS=%d, MAX_EDGES=%d\n", MAX_ROWS, MAX_EDGES);

    std::vector<BatchInfo> batch_info;
    int row_start = 0;
    while (row_start < n) {
        int row_end = std::min(row_start + MAX_ROWS, n);
        long long edge_end = g.nindex[row_end];
        long long edge_start = g.nindex[row_start];
        int eib = (int)(edge_end - edge_start);
        while (eib > MAX_EDGES && row_end > row_start + 1) {
            row_end--;
            edge_end = g.nindex[row_end];
            eib = (int)(edge_end - edge_start);
        }
        int nr = row_end - row_start;
        if (nr <= 0) break;
        batch_info.push_back({row_start, nr, eib});
        row_start = row_end;
    }
    const int num_batches = (int)batch_info.size();
    printf("Batches: %d\n", num_batches);

    int* d_nindex_batch[2] = {NULL, NULL};
    int* d_nlist_batch[2]  = {NULL, NULL};
    int* h_pinned_nindex[2] = {NULL, NULL};
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&d_nindex_batch[i], sizeof(int) * (size_t)(MAX_ROWS + 1)));
        CUDA_CHECK(cudaMalloc(&d_nlist_batch[i],  sizeof(int) * (size_t)MAX_EDGES));
        CUDA_CHECK(cudaMallocHost(&h_pinned_nindex[i], sizeof(int) * (size_t)(MAX_ROWS + 1)));
    }
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));
    print_gpu_memory_usage("after allocs");

    const int BLK = 256;
    auto t_compute_start = high_resolution_clock::now();

    // ---------- Phase 1: Init (one pass over all batches) ----------
    printf("Phase 1: Init...\n");
    fflush(stdout);
    for (int i = 0; i < num_batches; ++i) {
        const BatchInfo& info = batch_info[i];
        compute_local_nindex(g, info, h_pinned_nindex[0]);
        CUDA_CHECK(cudaMemcpy(d_nindex_batch[0], h_pinned_nindex[0],
                   sizeof(int) * (size_t)(info.n_rows + 1), cudaMemcpyHostToDevice));
        long long nlist_off = g.nindex[info.row_start];
        CUDA_CHECK(cudaMemcpy(d_nlist_batch[0], g.nlist + nlist_off,
                   sizeof(int) * (size_t)info.edge_count, cudaMemcpyHostToDevice));
        int blks = (info.n_rows + BLK - 1) / BLK;
        init_cc_batch<<<blks, BLK>>>(d_nindex_batch[0], d_nlist_batch[0], d_nstat, info.row_start, info.n_rows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- Phase 2: Merge until convergence ----------
    printf("Phase 2: Merge (until no change)...\n");
    fflush(stdout);
    const int max_merge_passes = 100;
    int merge_pass = 0;
    int changed = 1;
    while (changed != 0 && merge_pass < max_merge_passes) {
        merge_pass++;
        changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice));

        int buf = 0;
        for (int i = 0; i < num_batches; ++i) {
            const BatchInfo& info = batch_info[i];

            if (i > 0) {
                int pb = buf ^ 1;
                const BatchInfo& prev = batch_info[i - 1];
                int blks = (prev.n_rows + BLK - 1) / BLK;
                merge_cc_batch<<<blks, BLK, 0, streams[pb]>>>(
                    d_nindex_batch[pb], d_nlist_batch[pb],
                    d_nstat, d_changed,
                    prev.row_start, prev.n_rows);
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaStreamSynchronize(streams[buf]));
            compute_local_nindex(g, info, h_pinned_nindex[buf]);
            CUDA_CHECK(cudaMemcpyAsync(d_nindex_batch[buf], h_pinned_nindex[buf],
                       sizeof(int) * (size_t)(info.n_rows + 1), cudaMemcpyHostToDevice, streams[buf]));
            long long nlist_off = g.nindex[info.row_start];
            CUDA_CHECK(cudaMemcpyAsync(d_nlist_batch[buf], g.nlist + nlist_off,
                       sizeof(int) * (size_t)info.edge_count, cudaMemcpyHostToDevice, streams[buf]));
            buf ^= 1;
        }
        if (num_batches > 0) {
            int lb = buf ^ 1;
            const BatchInfo& last = batch_info[num_batches - 1];
            int blks = (last.n_rows + BLK - 1) / BLK;
            merge_cc_batch<<<blks, BLK, 0, streams[lb]>>>(
                d_nindex_batch[lb], d_nlist_batch[lb],
                d_nstat, d_changed, last.row_start, last.n_rows);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        if (merge_pass <= 3 || merge_pass % 10 == 0 || changed == 0)
            printf("  Merge pass %d: changed=%d\n", merge_pass, changed);
    }
    if (merge_pass >= max_merge_passes && changed != 0)
        printf("  Warning: reached max merge passes (%d)\n", max_merge_passes);

    // ---------- Phase 3: Flatten (one pass over all batches) ----------
    printf("Phase 3: Flatten...\n");
    fflush(stdout);
    for (int iter = 0; iter < 3; ++iter) {  // multiple flatten passes to propagate roots
        for (int i = 0; i < num_batches; ++i) {
            const BatchInfo& info = batch_info[i];
            compute_local_nindex(g, info, h_pinned_nindex[0]);
            CUDA_CHECK(cudaMemcpy(d_nindex_batch[0], h_pinned_nindex[0],
                       sizeof(int) * (size_t)(info.n_rows + 1), cudaMemcpyHostToDevice));
            long long nlist_off = g.nindex[info.row_start];
            CUDA_CHECK(cudaMemcpy(d_nlist_batch[0], g.nlist + nlist_off,
                       sizeof(int) * (size_t)info.edge_count, cudaMemcpyHostToDevice));
            int blks = (info.n_rows + BLK - 1) / BLK;
            flatten_cc_batch<<<blks, BLK>>>(d_nindex_batch[0], d_nlist_batch[0], d_nstat, info.row_start, info.n_rows);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    print_gpu_memory_usage("after compute");
    auto t_compute_end = high_resolution_clock::now();
    double compute_time = duration_cast<duration<double>>(t_compute_end - t_compute_start).count();

    std::vector<int> nstat_host(n);
    CUDA_CHECK(cudaMemcpy(nstat_host.data(), d_nstat, sizeof(int) * n, cudaMemcpyDeviceToHost));

    std::set<int> roots;
    for (int i = 0; i < n; ++i) roots.insert(nstat_host[i]);
    printf("Number of connected components: %zu\n", roots.size());
    printf("Compute time: %.6f s\n", compute_time);
    printf("Throughput: %.3f Mnodes/s\n", n * 0.000001 / compute_time);
    printf("Throughput: %.3f Medges/s\n", (double)g.edges * 0.000001 / compute_time);

    bool valid = validate_cc(g, nstat_host, n);
    printf("Validation: %s\n", valid ? "PASS" : "FAIL");

    // Write the component array to disk: one component id per line,
    // 0-indexed node order -- matches this project's own seed-file
    // convention (algorithms/coloring/scout_coloring.cu's ECL-GC seed reader
    // uses the same shape). Added because upstream ExCC otherwise only
    // prints summary stats and discards nstat_host after this point.
    {
        char out_path[4096];
        snprintf(out_path, sizeof(out_path), "%s.components.txt", argv[1]);
        FILE* outf = fopen(out_path, "w");
        if (outf) {
            for (int v = 0; v < n; v++) fprintf(outf, "%d\n", nstat_host[v]);
            fclose(outf);
            printf("components written to %s\n", out_path);
        } else {
            fprintf(stderr, "WARNING: could not open %s for writing\n", out_path);
        }
    }

    auto t_e2e_end = high_resolution_clock::now();
    double total_sec = duration_cast<duration<double>>(t_e2e_end - t_e2e_start).count();
    printf("Total CC time: %.6f s\n", total_sec);

    double peak_pct = (g_gpu_total_mb > 0.0) ? (g_peak_delta_mb / g_gpu_total_mb * 100.0) : 0.0;
    printf("Peak program GPU memory: %.2f MB (%.1f%% of %.2f MB total)\n",
           g_peak_delta_mb, peak_pct, g_gpu_total_mb);

    CUDA_CHECK(cudaFree(d_nstat));
    CUDA_CHECK(cudaFree(d_changed));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaFree(d_nindex_batch[i]));
        CUDA_CHECK(cudaFree(d_nlist_batch[i]));
        CUDA_CHECK(cudaFreeHost(h_pinned_nindex[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    freegraph(g);
    return 0;
}

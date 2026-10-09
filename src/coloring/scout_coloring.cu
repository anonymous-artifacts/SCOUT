// SCOUT: dynamic graph coloring (LCL problem, repair radius K = 1) on a storage-resident graph.
//
// An inserted edge whose endpoints share a color marks the higher-id endpoint for recoloring;
// deletions never invalidate a proper coloring. Marked vertices fetch their neighborhoods on
// the CPU and are recolored on the GPU in speculative propose/detect/apply rounds until no
// conflict remains.
//
// Usage: scout_coloring <graph.egr> <initial_colors.txt> <batch_dir> <num_batches> <gpu_id>
//   initial_colors.txt: one color per line (vertex order)
// Environment: SCOUT_NO_MEMO, SCOUT_NO_MADV_RANDOM, SCOUT_DUMP_TOUCHED=<file>.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <unordered_set>

#include "../../core/FlashGraph.h"
#include "../../core/gpu_common.h"
#include "../../core/cli_setup.h"
#include "../../core/round_buffer.h"
#include "../../core/reporting.h"

using namespace std;
// CUDA_CHECK now comes from core/gpu_common.h -- no longer defined here.

#define THREADS_PER_BLOCK 256
#define MAX_ROUND_VERTS 200000
#define MAX_ROUND_CSR_ELEMS 20000000
#define UNMARKED (-2)

static bool g_integrated = false;

// ---------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------

// Pass 1: propose a color for every round vertex, reading only the
// pre-round color snapshot (d_color is not written by this kernel).
// Publishes the proposal into d_round_marker[my_id] for pass 2 to see.
__global__ void propose_color_kernel(const int* ids, int num_ids,
                                      const long long* offsets, const int* elems,
                                      const int* d_color, int* d_round_marker,
                                      int* proposed_out) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ids) return;
    long long s = offsets[tid], e = offsets[tid + 1];
    int c = 0;
    bool found = false;
    while (!found) {
        found = true;
        for (long long i = s; i < e; i++) {
            if (d_color[elems[i]] == c) { found = false; break; }
        }
        if (!found) c++;
    }
    proposed_out[tid] = c;
    d_round_marker[ids[tid]] = c;
}

// Pass 2: detect same-round mutual-adjacency clashes. A round vertex loses
// (must retry) only if a NEIGHBOR that is ALSO a round vertex proposed the
// identical color AND has a lower id (deterministic, guarantees progress).
__global__ void detect_conflict_kernel(const int* ids, int num_ids,
                                        const long long* offsets, const int* elems,
                                        const int* d_round_marker, const int* proposed,
                                        int* retry_out, int* retry_count, int max_retry) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ids) return;
    int my_id = ids[tid];
    int my_color = proposed[tid];
    long long s = offsets[tid], e = offsets[tid + 1];
    bool lose = false;
    for (long long i = s; i < e; i++) {
        int n = elems[i];
        if (n < my_id && d_round_marker[n] == my_color) { lose = true; break; }
    }
    if (lose) {
        int pos = atomicAdd(retry_count, 1);
        if (pos < max_retry) retry_out[pos] = my_id;
    }
}

__global__ void apply_color_kernel(const int* ids, const int* proposed, int num_ids, int* d_color) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ids) return;
    d_color[ids[tid]] = proposed[tid];
}

// Clears this round's marker entries back to UNMARKED for the next round's
// (possibly different) vertex set -- touches only what this round touched.
__global__ void clear_marker_kernel(const int* ids, int num_ids, int* d_round_marker) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_ids) d_round_marker[ids[tid]] = UNMARKED;
}

struct ColoringRoundBuffers {
    int* h_ids;
    long long* h_offsets;
    int* h_elems;
    int* h_proposed;
    int* h_retry_ids;
    int* h_retry_count;

    int* d_ids;
    long long* d_offsets;
    int* d_elems;
    int* d_proposed;
    int* d_retry_ids;
    int* d_retry_count;

    int* d_color;
    int* h_color_view;   // aliases d_color when integrated; nullptr otherwise
    int* d_round_marker; // persistent, size num_nodes, init UNMARKED
    int num_nodes;

    template <typename T>
    void alloc_pair(T** h_ptr, T** d_ptr, size_t count) {
        ::alloc_pair(h_ptr, d_ptr, count, g_integrated); // core/gpu_common.h
    }

    void init(int n) {
        num_nodes = n;
        alloc_pair(&h_ids, &d_ids, (size_t)MAX_ROUND_VERTS);
        alloc_pair(&h_offsets, &d_offsets, (size_t)(MAX_ROUND_VERTS + 1));
        alloc_pair(&h_elems, &d_elems, (size_t)MAX_ROUND_CSR_ELEMS);
        alloc_pair(&h_proposed, &d_proposed, (size_t)MAX_ROUND_VERTS);
        alloc_pair(&h_retry_ids, &d_retry_ids, (size_t)MAX_ROUND_VERTS);
        alloc_pair(&h_retry_count, &d_retry_count, (size_t)1);

        if (g_integrated) {
            CUDA_CHECK(cudaHostAlloc(&h_color_view, (size_t)num_nodes * sizeof(int), cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_color, h_color_view, 0));
        } else {
            h_color_view = nullptr;
            CUDA_CHECK(cudaMalloc(&d_color, (size_t)num_nodes * sizeof(int)));
        }
        CUDA_CHECK(cudaMalloc(&d_round_marker, (size_t)num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_round_marker, 0xFF, (size_t)num_nodes * sizeof(int))); // -1 pattern; fixed up below
        vector<int> init_marker(num_nodes, UNMARKED);
        CUDA_CHECK(cudaMemcpy(d_round_marker, init_marker.data(), (size_t)num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    }

    // core/round_buffer.h -- shared with BFS's next_count round-trip.
    void reset_retry_count() { reset_scalar_count(d_retry_count, h_retry_count, g_integrated); }
    int read_retry_count() { return read_scalar_count_back(d_retry_count, h_retry_count, g_integrated); }
};

static double g_transfer_ms = 0, g_kernel_ms = 0;
static double g_fetch_ms = 0, g_commit_ms = 0;  // eval stage breakdown (core/reporting.h print_stage_times)
static void time_event_pair(cudaEvent_t a, cudaEvent_t b, double& accum) {
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); accum += ms;
}

struct Stats {
    long long insertions = 0, insertions_noop = 0, insertions_conflict = 0;
    long long deletions = 0; // always noop for coloring, tracked for symmetry with other algorithms
    long long total_rounds = 0, max_rounds_single_batch = 0;
    double total_batch_ms = 0;
};

struct Edit { int u, v; bool isInsertion; };

// One speculative propose/detect/apply round over `round_verts`. Returns
// the retry set (vertices that lost a same-round clash). Folds every
// applied color back into the host `color[]` mirror.
static vector<int> gpu_color_round(HybridGraph& graph, vector<int>& color,
                                    ColoringRoundBuffers& buf, vector<int>& round_verts) {
    if (round_verts.empty()) return {};
    vector<int> scratch;
    int num_v = 0;
    buf.h_offsets[0] = 0;
    long long elem_count = 0;
    auto fetch_t0 = chrono::high_resolution_clock::now();
    for (int v : round_verts) {
        if (num_v >= MAX_ROUND_VERTS) break;
        graph.neighbors_into(v, scratch);
        if (elem_count + (long long)scratch.size() > MAX_ROUND_CSR_ELEMS) break;
        buf.h_ids[num_v] = v;
        for (int u : scratch) buf.h_elems[elem_count++] = u;
        num_v++;
        buf.h_offsets[num_v] = elem_count;
    }
    g_fetch_ms += chrono::duration<double, milli>(chrono::high_resolution_clock::now() - fetch_t0).count();
    if (num_v == 0) return {};

    cudaEvent_t t0, t1, t2, t3;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2)); CUDA_CHECK(cudaEventCreate(&t3));

    CUDA_CHECK(cudaEventRecord(t0));
    if (!g_integrated) {
        CUDA_CHECK(cudaMemcpyAsync(buf.d_ids, buf.h_ids, num_v * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_offsets, buf.h_offsets, (num_v + 1) * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_elems, buf.h_elems, elem_count * sizeof(int), cudaMemcpyHostToDevice));
    }
    buf.reset_retry_count();
    CUDA_CHECK(cudaEventRecord(t1));

    int blocks = (num_v + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    propose_color_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        buf.d_ids, num_v, buf.d_offsets, buf.d_elems, buf.d_color, buf.d_round_marker, buf.d_proposed);
    detect_conflict_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        buf.d_ids, num_v, buf.d_offsets, buf.d_elems, buf.d_round_marker, buf.d_proposed,
        buf.d_retry_ids, buf.d_retry_count, MAX_ROUND_VERTS);
    apply_color_kernel<<<blocks, THREADS_PER_BLOCK>>>(buf.d_ids, buf.d_proposed, num_v, buf.d_color);
    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fold applied colors back into the host mirror.
    if (!g_integrated) {
        CUDA_CHECK(cudaMemcpy(buf.h_proposed, buf.d_proposed, num_v * sizeof(int), cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < num_v; i++) color[buf.h_ids[i]] = buf.h_proposed[i];

    int retry_count = buf.read_retry_count();
    if (retry_count > MAX_ROUND_VERTS) retry_count = MAX_ROUND_VERTS;
    vector<int> retry_verts;
    if (retry_count > 0) {
        if (!g_integrated) {
            CUDA_CHECK(cudaMemcpy(buf.h_retry_ids, buf.d_retry_ids, retry_count * sizeof(int), cudaMemcpyDeviceToHost));
        }
        retry_verts.assign(buf.h_retry_ids, buf.h_retry_ids + retry_count);
    }

    clear_marker_kernel<<<blocks, THREADS_PER_BLOCK>>>(buf.d_ids, num_v, buf.d_round_marker);
    CUDA_CHECK(cudaEventRecord(t3));
    CUDA_CHECK(cudaEventSynchronize(t3));
    time_event_pair(t0, t1, g_transfer_ms);
    time_event_pair(t1, t2, g_kernel_ms);
    time_event_pair(t2, t3, g_commit_ms);
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(t2)); CUDA_CHECK(cudaEventDestroy(t3));

    return retry_verts;
}

static void processBatch(vector<Edit>& batch, HybridGraph& graph, vector<int>& color,
                          Stats& stats, ColoringRoundBuffers& buf) {
    graph.memo_clear();   // batch-local memo
    auto t0 = chrono::high_resolution_clock::now();

    // Deletions never need repair (removing a constraint can't break a
    // proper coloring) -- apply the graph edit and move on.
    unordered_set<int> needs_recolor;
    for (auto& e : batch) {
        int u = e.u, v = e.v;
        if (e.isInsertion) {
            graph.insertEdge(u, v);
            stats.insertions++;
            if (color[u] != color[v]) { stats.insertions_noop++; continue; }
            stats.insertions_conflict++;
            int loser = (u > v) ? u : v;  // deterministic: higher id recolors
            needs_recolor.insert(loser);
        } else {
            graph.removeEdge(u, v);
            stats.deletions++;
        }
    }

    vector<int> round_verts(needs_recolor.begin(), needs_recolor.end());
    long long rounds_this_batch = 0;
    while (!round_verts.empty()) {
        rounds_this_batch++;
        round_verts = gpu_color_round(graph, color, buf, round_verts);
    }
    stats.total_rounds += rounds_this_batch;
    if (rounds_this_batch > stats.max_rounds_single_batch) stats.max_rounds_single_batch = rounds_this_batch;

    auto t1 = chrono::high_resolution_clock::now();
    stats.total_batch_ms += chrono::duration<double, milli>(t1 - t0).count();
}

int main(int argc, char* argv[]) {
    ScoutCliArgs args = parse_scout_cli(argc, argv, "coloring"); // core/cli_setup.h
    g_integrated = gpu_warmup_and_device_setup(args.gpu_id); // core/cli_setup.h
    if (getenv("SCOUT_FORCE_COPY")) g_integrated = false;  // integrated GPU: use device buffers + copies instead of mapped host memory

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.gpu_id));

    auto setup_t0 = chrono::high_resolution_clock::now();

    HybridGraph graph;
    graph.open(args.egr_path.c_str());
    graph.enable_memo();
    int num_nodes = graph.num_nodes;

    long long total_degree = graph.base.edges;   // no updates applied yet: sum of degrees = stored edges
    double avg_degree = num_nodes > 0 ? (double)total_degree / num_nodes : 0.0;
    print_run_header("coloring", args.egr_path, num_nodes, total_degree, avg_degree, prop.name, g_integrated); // core/reporting.h

    vector<int> color(num_nodes, 0);
    int initial_colors_used = 0;
    {
        vector<long long> vals = read_ints_fast(args.context_path); // core/cli_setup.h
        for (int idx = 0; idx < num_nodes && idx < (int)vals.size(); idx++) color[idx] = (int)vals[idx];
        int max_c = -1;
        for (int v = 0; v < num_nodes; v++) if (color[v] > max_c) max_c = color[v];
        initial_colors_used = max_c + 1;
    }

    ColoringRoundBuffers buf;
    buf.init(num_nodes);
    if (g_integrated) {
        memcpy(buf.h_color_view, color.data(), (size_t)num_nodes * sizeof(int));
    } else {
        CUDA_CHECK(cudaMemcpy(buf.d_color, color.data(), (size_t)num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    }

    auto setup_t1 = chrono::high_resolution_clock::now();
    double setup_ms = chrono::duration<double, milli>(setup_t1 - setup_t0).count();

    vector<filesystem::path> batch_paths = list_batch_files_sorted(args.batch_dir); // core/cli_setup.h

    Stats stats;
    int processed = 0;
    for (const auto& path : batch_paths) {
        if (processed >= args.num_batches) break;
        vector<EdgeUpdate> raw_batch = parse_batch_file(path, graph, num_nodes); // core/cli_setup.h
        vector<Edit> batch;
        batch.reserve(raw_batch.size());
        for (const auto& e : raw_batch) batch.push_back({e.u, e.v, e.is_insertion});
        Stats before = stats;
        double f0 = g_fetch_ms, x0 = g_transfer_ms, k0 = g_kernel_ms, c0 = g_commit_ms;
        processBatch(batch, graph, color, stats, buf);
        double this_batch_ms = stats.total_batch_ms - before.total_batch_ms;
        print_batch_result(processed + 1, args.num_batches, (int)batch.size(), this_batch_ms, // core/reporting.h
            {{"insertions", to_string(stats.insertions - before.insertions)},
             {"conflicts", to_string(stats.insertions_conflict - before.insertions_conflict)},
             {"rounds", to_string(stats.total_rounds - before.total_rounds)}});
        {
            double f = g_fetch_ms - f0, x = g_transfer_ms - x0, k = g_kernel_ms - k0, c = g_commit_ms - c0;
            print_stage_times(processed + 1, f, max(0.0, this_batch_ms - f - x - k - c), x, k, c); // core/reporting.h
        }
        processed++;
    }

    int colors_used;
    {
        int max_color = -1;
        for (int v = 0; v < num_nodes; v++) if (color[v] > max_color) max_color = color[v];
        colors_used = max_color + 1;
    }

    cout << "Insertions: " << stats.insertions << " (noop=" << stats.insertions_noop
         << ", conflict=" << stats.insertions_conflict << ")" << endl;
    cout << "Deletions:  " << stats.deletions << " (always a structural noop for coloring)" << endl;
    cout << "Speculative rounds: total=" << stats.total_rounds
         << " max_in_single_batch=" << stats.max_rounds_single_batch << endl;
    cout << "  GPU kernel: " << fixed << setprecision(3) << g_kernel_ms
         << " ms | transfer: " << g_transfer_ms << " ms" << endl;

    double e2e_ms = setup_ms + stats.total_batch_ms;
    print_final_summary(setup_ms, stats.total_batch_ms, e2e_ms, // core/reporting.h
                         "colors used", initial_colors_used,
                         "colors used", colors_used);

    // Results are printed; skip freeing pinned/mapped buffers and CUDA context teardown
    // (the OS reclaims them), which otherwise adds seconds after the work is done.
    if (!g_touched.empty()) dump_touched_at_exit();   // SCOUT_DUMP_TOUCHED: _Exit skips atexit handlers
    std::cout.flush();
    fflush(stdout);
    std::_Exit(0);

}

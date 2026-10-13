// SCOUT BFS (non-LCL) on the NL driver. Each round expands the vertices whose level changed;
// a deletion past the budget is deferred to one full traversal after the stream.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <omp.h>

#include "../../core/FlashGraph.h"
#include "../../core/gpu_common.h"
#include "../../core/cli_setup.h"
#include "../../core/round_buffer.h"
#include "../../core/local_repair.h"
#include "../../core/reporting.h"
#include "../../core/fetch.h"
#include "../../core/nl_driver.h"

using namespace std;

const int UNREACHED = -1;
#define THREADS_PER_BLOCK 256
#define MAX_ROUND_FRONTIER 200000
#define MAX_ROUND_CSR_ELEMS 20000000

static bool g_integrated = false;
static bool g_defer_fallback = true;

__device__ inline bool gpu_relax(int* level_ptr, int candidate) {
    int old = *level_ptr;
    while (old == -1 || old > candidate) {
        int prev = atomicCAS(level_ptr, old, candidate);
        if (prev == old) return true;
        old = prev;
    }
    return false;
}

__global__ void relax_frontier_kernel(const int* frontier_ids, const int* frontier_levels,
                                       int num_frontier, const long long* offsets,
                                       const int* elems, int* d_level,
                                       int* next_out, int* next_count, int max_next) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_frontier) return;
    int lv = frontier_levels[tid];
    long long s = offsets[tid], e = offsets[tid + 1];
    for (long long i = s; i < e; i++) {
        int u = elems[i];
        if (gpu_relax(&d_level[u], lv + 1)) {
            int pos = atomicAdd(next_count, 1);
            if (pos < max_next) next_out[pos] = u;
        }
    }
}

__global__ void deletion_compute_kernel(const int* worklist_ids, const int* worklist_old_levels,
                                         int num_worklist, const long long* offsets,
                                         const int* elems, const int* d_level,
                                         int* new_level_out, int* next_out, int* next_count, int max_next) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_worklist) return;
    int old_level = worklist_old_levels[tid];
    long long s = offsets[tid], e = offsets[tid + 1];
    int best = UNREACHED;
    for (long long i = s; i < e; i++) {
        int ln = d_level[elems[i]];
        if (ln != UNREACHED && (best == UNREACHED || ln < best)) best = ln;
    }
    int new_level = (best == UNREACHED) ? UNREACHED : best + 1;
    new_level_out[tid] = new_level;
    if (new_level != old_level) {
        for (long long i = s; i < e; i++) {
            int n = elems[i];
            int ln = d_level[n];
            if (ln != UNREACHED && old_level != UNREACHED && ln == old_level + 1) {
                int pos = atomicAdd(next_count, 1);
                if (pos < max_next) next_out[pos] = n;
            }
        }
    }
}

__global__ void deletion_apply_kernel(const int* worklist_ids, const int* new_levels,
                                       int num_worklist, int* d_level) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_worklist) return;
    d_level[worklist_ids[tid]] = new_levels[tid];
}

__global__ void gather_levels_kernel(const int* ids, int count, const int* d_level, int* out_levels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) out_levels[tid] = d_level[ids[tid]];
}

struct BfsRoundBuffers {
    int* h_ids;
    int* h_aux_levels;
    long long* h_offsets;
    int* h_elems;
    int* h_next_ids;
    int* h_next_levels;

    int* d_ids;
    int* d_aux_levels;
    long long* d_offsets;
    int* d_elems;
    int* d_new_levels;
    int* d_next_ids;
    int* d_next_count;
    int* d_gather_out;

    int* d_level;
    int* h_level_view;
    int num_nodes;

    int* h_next_count;

    template <typename T>
    void alloc_pair(T** h_ptr, T** d_ptr, size_t count) {
        ::alloc_pair(h_ptr, d_ptr, count, g_integrated);
    }

    void init(int n) {
        num_nodes = n;
        alloc_pair(&h_ids, &d_ids, (size_t)MAX_ROUND_FRONTIER);
        alloc_pair(&h_aux_levels, &d_aux_levels, (size_t)MAX_ROUND_FRONTIER);
        alloc_pair(&h_offsets, &d_offsets, (size_t)(MAX_ROUND_FRONTIER + 1));
        alloc_pair(&h_elems, &d_elems, (size_t)MAX_ROUND_CSR_ELEMS);
        alloc_pair(&h_next_ids, &d_next_ids, (size_t)MAX_ROUND_FRONTIER);
        alloc_pair(&h_next_levels, &d_gather_out, (size_t)MAX_ROUND_FRONTIER);
        int* dummy_h;
        alloc_pair(&dummy_h, &d_new_levels, (size_t)MAX_ROUND_FRONTIER);
        int* dummy_h2;
        alloc_pair(&dummy_h2, &d_next_count, (size_t)1);
        h_next_count = dummy_h2;

        if (g_integrated) {
            CUDA_CHECK(cudaHostAlloc(&h_level_view, (size_t)num_nodes * sizeof(int), cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_level, h_level_view, 0));
        } else {
            h_level_view = nullptr;
            CUDA_CHECK(cudaMalloc(&d_level, (size_t)num_nodes * sizeof(int)));
        }
    }

    void reset_next_count() { reset_scalar_count(d_next_count, h_next_count, g_integrated); }
    int read_next_count() { return read_scalar_count_back(d_next_count, h_next_count, g_integrated); }
};

static double g_transfer_ms = 0, g_kernel_ms = 0;
static double g_commit_ms = 0;

static void time_event_pair(cudaEvent_t a, cudaEvent_t b, double& accum) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    accum += ms;
}

// One GPU round over a frontier chunk whose lists came from FetchFrontier.
static vector<int> gpu_relax_round(vector<int>& level, BfsRoundBuffers& buf,
                                    const vector<int>& frontier, const FetchResult& fetched,
                                    int* raw_next_count_out = nullptr) {
    if (raw_next_count_out) *raw_next_count_out = 0;
    int num_f = (int)frontier.size();
    if (num_f == 0) return {};

    buf.h_offsets[0] = 0;
    long long elem_count = 0;
    for (int i = 0; i < num_f; i++) {
        int v = frontier[i];
        buf.h_ids[i] = v;
        buf.h_aux_levels[i] = level[v];
        for (int u : fetched[i]) buf.h_elems[elem_count++] = u;
        buf.h_offsets[i + 1] = elem_count;
    }

    cudaEvent_t t0, t1, t2, t3;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2)); CUDA_CHECK(cudaEventCreate(&t3));

    CUDA_CHECK(cudaEventRecord(t0));
    if (!g_integrated) {
        CUDA_CHECK(cudaMemcpyAsync(buf.d_ids, buf.h_ids, num_f * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_aux_levels, buf.h_aux_levels, num_f * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_offsets, buf.h_offsets, (num_f + 1) * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_elems, buf.h_elems, elem_count * sizeof(int), cudaMemcpyHostToDevice));
    }
    buf.reset_next_count();
    CUDA_CHECK(cudaEventRecord(t1));

    int blocks = (num_f + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    relax_frontier_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        buf.d_ids, buf.d_aux_levels, num_f, buf.d_offsets, buf.d_elems,
        buf.d_level, buf.d_next_ids, buf.d_next_count, MAX_ROUND_FRONTIER);
    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaDeviceSynchronize());

    int next_count = buf.read_next_count();
    if (raw_next_count_out) *raw_next_count_out = next_count;
    if (next_count > MAX_ROUND_FRONTIER) next_count = MAX_ROUND_FRONTIER;

    vector<int> next_frontier;
    if (next_count > 0) {
        int gblocks = (next_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        gather_levels_kernel<<<gblocks, THREADS_PER_BLOCK>>>(buf.d_next_ids, next_count, buf.d_level, buf.d_gather_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!g_integrated) {
            CUDA_CHECK(cudaMemcpy(buf.h_next_ids, buf.d_next_ids, next_count * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(buf.h_next_levels, buf.d_gather_out, next_count * sizeof(int), cudaMemcpyDeviceToHost));
        }

        next_frontier.reserve(next_count);
        for (int i = 0; i < next_count; i++) {
            int id = buf.h_next_ids[i];
            level[id] = buf.h_next_levels[i];
            next_frontier.push_back(id);
        }
    }
    CUDA_CHECK(cudaEventRecord(t3));
    CUDA_CHECK(cudaEventSynchronize(t3));
    time_event_pair(t0, t1, g_transfer_ms);
    time_event_pair(t1, t2, g_kernel_ms);
    time_event_pair(t2, t3, g_commit_ms);
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(t2)); CUDA_CHECK(cudaEventDestroy(t3));

    return next_frontier;
}

// Full recompute: traversal from the source (memo bypassed).
static void gpu_full_bfs(HybridGraph& graph, vector<int>& level, BfsRoundBuffers& buf, int source) {
    fill(level.begin(), level.end(), UNREACHED);
    level[source] = 0;

    if (g_integrated) {
        memset(buf.h_level_view, 0xFF, (size_t)buf.num_nodes * sizeof(int));
        buf.h_level_view[source] = 0;
    } else {
        CUDA_CHECK(cudaMemset(buf.d_level, 0xFF, (size_t)buf.num_nodes * sizeof(int)));
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(&buf.d_level[source], &zero, sizeof(int), cudaMemcpyHostToDevice));
    }

    vector<int> frontier = {source};
    FetchResult fr;
    while (!frontier.empty()) {
        vector<int> next_frontier;
        size_t i = 0;
        while (i < frontier.size()) {
            vector<int> chunk;
            long long chunk_degree_sum = 0;
            while (i < frontier.size()) {
                long long d = (long long)graph.degree(frontier[i]);
                if (!chunk.empty() && (chunk_degree_sum + d > MAX_ROUND_FRONTIER ||
                                        chunk.size() >= (size_t)MAX_ROUND_FRONTIER)) break;
                chunk.push_back(frontier[i]);
                chunk_degree_sum += d;
                i++;
            }
            if (chunk.empty()) {
                fprintf(stderr, "gpu_full_bfs: vertex %d has degree > MAX_ROUND_FRONTIER (%d), skipping it\n",
                        frontier[i], MAX_ROUND_FRONTIER);
                i++;
                continue;
            }
            int raw_next = 0;
            fetch_frontier(graph, chunk, fr, false);
            vector<int> chunk_next = gpu_relax_round(level, buf, chunk, fr, &raw_next);
            if (raw_next > MAX_ROUND_FRONTIER) {

                fprintf(stderr, "gpu_full_bfs: INTERNAL ERROR -- degree-bounded chunk still overflowed (raw_next=%d) -- results may be wrong\n", raw_next);
            }
            next_frontier.insert(next_frontier.end(), chunk_next.begin(), chunk_next.end());
        }
        frontier = std::move(next_frontier);
    }
}

// One deletion round: each vertex takes 1 + its smallest neighbor level; changed
// vertices pass their children on.
static vector<int> gpu_deletion_round(vector<int>& level, BfsRoundBuffers& buf,
                                       const vector<int>& worklist, const FetchResult& fetched,
                                       int* raw_next_count_out = nullptr) {
    if (raw_next_count_out) *raw_next_count_out = 0;
    int num_w = (int)worklist.size();
    if (num_w == 0) return {};

    buf.h_offsets[0] = 0;
    long long elem_count = 0;
    for (int i = 0; i < num_w; i++) {
        int w = worklist[i];
        buf.h_ids[i] = w;
        buf.h_aux_levels[i] = level[w];
        for (int n : fetched[i]) buf.h_elems[elem_count++] = n;
        buf.h_offsets[i + 1] = elem_count;
    }

    cudaEvent_t t0, t1, t2, t3;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2)); CUDA_CHECK(cudaEventCreate(&t3));

    CUDA_CHECK(cudaEventRecord(t0));
    if (!g_integrated) {
        CUDA_CHECK(cudaMemcpyAsync(buf.d_ids, buf.h_ids, num_w * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_aux_levels, buf.h_aux_levels, num_w * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_offsets, buf.h_offsets, (num_w + 1) * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(buf.d_elems, buf.h_elems, elem_count * sizeof(int), cudaMemcpyHostToDevice));
    }
    buf.reset_next_count();
    CUDA_CHECK(cudaEventRecord(t1));

    int blocks = (num_w + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    deletion_compute_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        buf.d_ids, buf.d_aux_levels, num_w, buf.d_offsets, buf.d_elems,
        buf.d_level, buf.d_new_levels, buf.d_next_ids, buf.d_next_count, MAX_ROUND_FRONTIER);
    deletion_apply_kernel<<<blocks, THREADS_PER_BLOCK>>>(buf.d_ids, buf.d_new_levels, num_w, buf.d_level);
    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> h_new_levels(num_w);
    if (g_integrated) {
        CUDA_CHECK(cudaMemcpy(h_new_levels.data(), buf.d_new_levels, num_w * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(h_new_levels.data(), buf.d_new_levels, num_w * sizeof(int), cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < num_w; i++) level[buf.h_ids[i]] = h_new_levels[i];

    int next_count = buf.read_next_count();
    if (raw_next_count_out) *raw_next_count_out = next_count;
    if (next_count > MAX_ROUND_FRONTIER) next_count = MAX_ROUND_FRONTIER;

    vector<int> next_worklist;
    if (next_count > 0) {
        if (!g_integrated) {
            CUDA_CHECK(cudaMemcpy(buf.h_next_ids, buf.d_next_ids, next_count * sizeof(int), cudaMemcpyDeviceToHost));
        }
        next_worklist.assign(buf.h_next_ids, buf.h_next_ids + next_count);
    }
    CUDA_CHECK(cudaEventRecord(t3));
    CUDA_CHECK(cudaEventSynchronize(t3));
    time_event_pair(t0, t1, g_transfer_ms);
    time_event_pair(t1, t2, g_kernel_ms);
    time_event_pair(t2, t3, g_commit_ms);
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(t2)); CUDA_CHECK(cudaEventDestroy(t3));

    return next_worklist;
}

struct Stats {
    long long insertions = 0, insertions_noop = 0, insertions_repaired = 0;
    long long deletions = 0, deletions_noop = 0, deletions_local = 0, deletions_fallback = 0;
    long long deletions_fallback_deferred = 0;
    long long total_affected_vertices = 0;
    double total_batch_ms = 0;
    double fallback_ms = 0;
};

static inline void sync_level_to_device(BfsRoundBuffers& buf, int idx, int value) {
    if (g_integrated) {
        buf.h_level_view[idx] = value;
    } else {
        CUDA_CHECK(cudaMemcpy(&buf.d_level[idx], &value, sizeof(int), cudaMemcpyHostToDevice));
    }
}

// The next part of a frontier that fits one GPU round.
static vector<int> next_chunk(HybridGraph& graph, const vector<int>& frontier, size_t pos) {
    vector<int> chunk;
    long long deg_sum = 0;
    for (size_t i = pos; i < frontier.size(); i++) {
        long long d = (long long)graph.degree(frontier[i]);
        if (!chunk.empty() && (deg_sum + d > MAX_ROUND_FRONTIER || chunk.size() >= (size_t)MAX_ROUND_FRONTIER)) break;
        chunk.push_back(frontier[i]);
        deg_sum += d;
    }
    return chunk;
}

struct BfsApp {
    HybridGraph& graph;
    vector<int>& level;
    BfsRoundBuffers& buf;
    int source;
    Stats& stats;
    Budget deletion_budget;

    struct Search {
        bool deletion = false;
        vector<int> frontier, next;
        size_t pos = 0;
        long long touched = 0;
    };

    Budget budget_for(const Update& e) { return e.ins ? Budget::unbounded() : deletion_budget; }

    // seed: the improved endpoint of an insertion, the child of a deleted tree edge.
    bool seed(const Update& e, Search& R, Budget&) {
        int u = e.u, v = e.v;
        if (e.ins) {
            stats.insertions++;
            bool improve_v = (level[u] != UNREACHED) && (level[v] == UNREACHED || level[u] + 1 < level[v]);
            bool improve_u = (level[v] != UNREACHED) && (level[u] == UNREACHED || level[v] + 1 < level[u]);
            if (!improve_u && !improve_v) { stats.insertions_noop++; return false; }
            if (improve_v) { level[v] = level[u] + 1; sync_level_to_device(buf, v, level[v]); R.frontier.push_back(v); }
            if (improve_u) { level[u] = level[v] + 1; sync_level_to_device(buf, u, level[u]); R.frontier.push_back(u); }
            R.touched = (long long)R.frontier.size();
            return true;
        }
        stats.deletions++;
        bool was_tree_edge = (level[u] != UNREACHED && level[v] == level[u] + 1) ||
                             (level[v] != UNREACHED && level[u] == level[v] + 1);
        if (!was_tree_edge) { stats.deletions_noop++; return false; }
        R.deletion = true;
        R.frontier.push_back((level[u] > level[v]) ? u : v);
        return true;
    }

    bool resolved(const Search& R) const { return R.frontier.empty(); }

    // select: a deletion's search is charged |level| when the level starts.
    vector<int> select(Search& R, Budget& bud) {
        if (R.deletion && R.pos == 0) bud.vertices += (long long)R.frontier.size();
        return next_chunk(graph, R.frontier, R.pos);
    }

    void expand(Search& R, const vector<int>& Y, const FetchResult& fr, Budget&) {
        int raw_next = 0;
        vector<int> out = R.deletion ? gpu_deletion_round(level, buf, Y, fr, &raw_next)
                                     : gpu_relax_round(level, buf, Y, fr, &raw_next);
        if (raw_next > (int)MAX_ROUND_FRONTIER)
            fprintf(stderr, "bfs round: INTERNAL ERROR -- output truncated (raw_next_count=%d > %d)\n",
                    raw_next, MAX_ROUND_FRONTIER);
        R.next.insert(R.next.end(), out.begin(), out.end());
        R.touched += (long long)Y.size();
        R.pos += Y.size();
        if (R.pos >= R.frontier.size()) { R.frontier = std::move(R.next); R.next.clear(); R.pos = 0; }
    }

    void commit(const Search& R) {
        if (R.deletion) stats.deletions_local++;
        else { stats.insertions_repaired++; stats.total_affected_vertices += R.touched; }
    }

    void fallback(const Search&) {
        stats.deletions_fallback++;
        if (g_defer_fallback) { stats.deletions_fallback_deferred++; return; }
        auto fb0 = chrono::high_resolution_clock::now();
        gpu_full_bfs(graph, level, buf, source);
        stats.fallback_ms += chrono::duration<double, milli>(chrono::high_resolution_clock::now() - fb0).count();
    }
};

int main(int argc, char* argv[]) {
    ScoutCliArgs args = parse_scout_cli(argc, argv, "bfs");
    g_integrated = gpu_warmup_and_device_setup(args.gpu_id);
    if (getenv("SCOUT_FORCE_COPY")) g_integrated = false;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.gpu_id));

    g_defer_fallback = read_defer_fallback_env(true);
    cout << (g_defer_fallback
             ? "[Deferred fallback: ON (default)] Cascades past the safety bound are deferred; "
               "resolved by ONE catch-up gpu_full_bfs after all batches, instead of one call each."
             : "[Deferred fallback: OFF (SCOUT_DEFER_FALLBACK=0)] Each cascade past the safety "
               "bound triggers its own immediate gpu_full_bfs call.") << endl;

    auto setup_t0 = chrono::high_resolution_clock::now();

    HybridGraph graph;
    graph.open(args.egr_path.c_str());
    graph.enable_memo();
    int num_nodes = graph.num_nodes;

    long long total_degree = 0;
    total_degree = graph.base.edges;
    double avg_degree = num_nodes > 0 ? (double)total_degree / num_nodes : 0.0;
    print_run_header("bfs", args.egr_path, num_nodes, total_degree, avg_degree, prop.name, g_integrated);

    vector<int> level(num_nodes, UNREACHED);
    int source = 0;
    int initial_reached = 0;
    {
        vector<long long> vals = read_ints_fast(args.context_path);
        for (size_t k = 0; k + 1 < vals.size(); k += 2) {
            int idx = (int)vals[k] - 1, lv = (int)vals[k + 1];
            if (idx >= 0 && idx < num_nodes) { level[idx] = lv; if (lv >= 0) initial_reached++; }
        }
    }

    BfsRoundBuffers buf;
    buf.init(num_nodes);
    if (g_integrated) {
        memcpy(buf.h_level_view, level.data(), (size_t)num_nodes * sizeof(int));
    } else {
        CUDA_CHECK(cudaMemcpy(buf.d_level, level.data(), (size_t)num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    }

    auto setup_t1 = chrono::high_resolution_clock::now();
    double setup_ms = chrono::duration<double, milli>(setup_t1 - setup_t0).count();

    vector<filesystem::path> batch_paths = list_batch_files_sorted(args.batch_dir);

    Stats stats;
    BfsApp app{graph, level, buf, source, stats,
               Budget::from_env(getenv("SCOUT_BFS_BOUND") ? atoll(getenv("SCOUT_BFS_BOUND")) : 200)};
    cout << "[NL driver] deletion budget: " << app.deletion_budget.describe()
         << ", memo " << (graph.memo_on ? "on" : "off (SCOUT_NO_MEMO)") << endl;
    int processed = 0;
    for (const auto& path : batch_paths) {
        if (processed >= args.num_batches) break;
        vector<EdgeUpdate> raw_batch = parse_batch_file(path, graph, num_nodes);
        vector<Update> batch;
        batch.reserve(raw_batch.size());
        for (const auto& e : raw_batch) batch.push_back({e.u, e.v, e.is_insertion});
        Stats before = stats;
        FetchStats fs0 = g_fetch;
        long long mf0 = process_major_faults();
        double x0 = g_transfer_ms, k0 = g_kernel_ms, c0 = g_commit_ms;
        auto bt0 = chrono::high_resolution_clock::now();
        nl_process_batch(app, graph, batch);
        double this_batch_ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - bt0).count();
        stats.total_batch_ms += this_batch_ms;
        print_batch_result(processed + 1, args.num_batches, (int)batch.size(), this_batch_ms,
            {{"insertions", to_string(stats.insertions - before.insertions)},
             {"deletions", to_string(stats.deletions - before.deletions)},
             {"fallback", to_string(stats.deletions_fallback - before.deletions_fallback)}});
        {
            double f = g_fetch.ms - fs0.ms, x = g_transfer_ms - x0, k = g_kernel_ms - k0, c = g_commit_ms - c0;
            print_stage_times(processed + 1, f, max(0.0, this_batch_ms - f - x - k - c), x, k, c);
            print_fetch_stats(processed + 1, g_fetch.pages - fs0.pages, process_major_faults() - mf0, g_fetch.lists - fs0.lists,
                              g_fetch.memo_hits - fs0.memo_hits);
        }
        processed++;
    }

    double deferred_fallback_ms = 0;
    if (g_defer_fallback && stats.deletions_fallback_deferred > 0) {
        cout << "\n[DEFERRED FALLBACK] resolving " << stats.deletions_fallback_deferred
             << " deferred cascades with one gpu_full_bfs call..." << endl;
        auto fb0 = chrono::high_resolution_clock::now();
        gpu_full_bfs(graph, level, buf, source);
        auto fb1 = chrono::high_resolution_clock::now();
        deferred_fallback_ms = chrono::duration<double, milli>(fb1 - fb0).count();
        cout << "[DEFERRED FALLBACK] done in " << fixed << setprecision(3) << deferred_fallback_ms << " ms" << endl;
    }

    long long final_reached = 0;
    for (int i = 0; i < num_nodes; i++) if (level[i] != UNREACHED) final_reached++;

    cout << "Pages requested (Alg. 1): " << g_fetch.pages << " | lists reconciled: " << g_fetch.lists
         << " | memo hits: " << g_fetch.memo_hits << endl;
    cout << "Insertions: " << stats.insertions << " (noop=" << stats.insertions_noop
         << ", repaired=" << stats.insertions_repaired << ")" << endl;
    cout << "Deletions:  " << stats.deletions << " (noop=" << stats.deletions_noop
         << ", local=" << stats.deletions_local << ", fallback=" << stats.deletions_fallback
         << ", deferred=" << stats.deletions_fallback_deferred << ")" << endl;
    cout << "  fallback (gpu_full_bfs, immediate): " << fixed << setprecision(3) << stats.fallback_ms << " ms"
         << " | deferred catch-up: " << deferred_fallback_ms << " ms"
         << " | GPU kernel: " << g_kernel_ms << " ms | transfer: " << g_transfer_ms << " ms" << endl;

    double processing_ms = stats.total_batch_ms + deferred_fallback_ms;
    double e2e_ms = setup_ms + processing_ms;
    print_final_summary(setup_ms, processing_ms, e2e_ms,
                         "reached vertices", initial_reached,
                         "reached vertices", final_reached);

    if (!g_touched.empty()) dump_touched_at_exit();
    std::cout.flush();
    fflush(stdout);
    std::_Exit(0);

}

// SCOUT connected components (non-LCL) on the NL driver. Labels with a union-find over
// labels; a deletion runs a bidirectional search, and a split side gets a fresh label.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <unordered_set>
#include <omp.h>

#include "../../core/FlashGraph.h"
#include "../../core/gpu_common.h"
#include "../../core/cli_setup.h"
#include "../../core/local_repair.h"
#include "../../core/reporting.h"
#include "../../core/fetch.h"
#include "../../core/nl_driver.h"

using namespace std;

__global__ void cc_hook_kernel(int num_nodes, const long long* nindex, const int* nlist,
                                int* parent, int* changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    int pv = parent[v];
    long long s = nindex[v], e = nindex[v + 1];
    for (long long i = s; i < e; i++) {
        int n = nlist[i];
        int pn = parent[n];
        if (pv == pn) continue;
        int hi = pv > pn ? pv : pn;
        int lo = pv > pn ? pn : pv;
        if (parent[hi] == hi) {
            parent[hi] = lo;
            *changed = 1;
        }
    }
}

__global__ void cc_shortcut_kernel(int num_nodes, int* parent) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    parent[v] = parent[parent[v]];
}

#define CC_THREADS_PER_BLOCK 256

static bool g_integrated = false;
static bool g_defer_fallback = false;
static bool g_use_exhaustive_search = false;
static long long g_fallback_bound = 5000;
static const double LOW_DEGREE_THRESHOLD = 10.0;

template <typename T>
static void cc_alloc_and_upload(T** d_ptr, const vector<T>& h_data, T** h_view_out) {
    ::alloc_and_upload(d_ptr, h_data, h_view_out, g_integrated);
}

static int find_root_atomic(int* p, int v) {
    while (true) {
        int q = __atomic_load_n(&p[v], __ATOMIC_RELAXED);
        if (q == v) return v;
        int r = __atomic_load_n(&p[q], __ATOMIC_RELAXED);
        if (r != q) __atomic_compare_exchange_n(&p[v], &q, r, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        v = q;
    }
}

// Host recompute, used when the graph does not fit in free GPU memory.
static void full_recompute_cc_streamed(HybridGraph& graph, vector<int>& parent, int num_nodes) {
    auto t0 = chrono::high_resolution_clock::now();
    int* p = parent.data();
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < num_nodes; v++) p[v] = v;
    #pragma omp parallel
    {
        vector<int> scratch;
        #pragma omp for schedule(dynamic, 4096)
        for (int v = 0; v < num_nodes; v++) {
            graph.neighbors_direct(v, scratch);
            for (int u : scratch) {
                if (u <= v) continue;
                int a = v, b = u;
                while (true) {
                    a = find_root_atomic(p, a);
                    b = find_root_atomic(p, b);
                    if (a == b) break;
                    int lo = a < b ? a : b, hi = a < b ? b : a;
                    int expect = hi;
                    if (__atomic_compare_exchange_n(&p[hi], &expect, lo, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
                }
            }
        }
    }
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < num_nodes; v++) p[v] = find_root_atomic(p, v);
    fprintf(stderr, "[CC-FALLBACK-DIAG] streamed host recompute (graph exceeds free GPU memory): %.1fms\n",
            chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count());
    fflush(stderr);
}

// Full recompute: one sequential pass over the current graph, streamed to the GPU.
static void full_recompute_cc_gpu(HybridGraph& graph, vector<int>& parent, int num_nodes) {
    if (!g_integrated) {
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        size_t need = (size_t)graph.base.edges * sizeof(int) + (size_t)(num_nodes + 1) * sizeof(long long)
                    + (size_t)num_nodes * sizeof(int);
        if (need > free_b * 0.9) { full_recompute_cc_streamed(graph, parent, num_nodes); return; }
    }
    auto t_pack0 = chrono::high_resolution_clock::now();

    vector<long long> h_nindex(num_nodes + 1);
    vector<int> deg(num_nodes);
    #pragma omp parallel for schedule(dynamic, 4096)
    for (int v = 0; v < num_nodes; v++) {
        deg[v] = graph.degree(v);
    }

    h_nindex[0] = 0;
    for (int v = 0; v < num_nodes; v++) h_nindex[v + 1] = h_nindex[v] + deg[v];
    long long total_edges = h_nindex[num_nodes];

    int* d_nlist; int* h_nlist_view = nullptr;
    if (g_integrated) {
        CUDA_CHECK(cudaHostAlloc(&h_nlist_view, (size_t)total_edges * sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&d_nlist, h_nlist_view, 0));
        #pragma omp parallel
        {
            vector<int> scratch;
            #pragma omp for schedule(dynamic, 4096)
            for (int v = 0; v < num_nodes; v++) {
                graph.neighbors_direct(v, scratch);
                long long off = h_nindex[v];
                for (size_t i = 0; i < scratch.size(); i++) h_nlist_view[off + i] = scratch[i];
            }
        }
    } else {
        CUDA_CHECK(cudaMalloc(&d_nlist, (size_t)total_edges * sizeof(int)));

        const int NUM_CHUNKS = 16;
        long long target_per_chunk = (total_edges + NUM_CHUNKS - 1) / NUM_CHUNKS;
        vector<int> chunk_vstart = {0};
        long long acc = 0;
        for (int v = 0; v < num_nodes; v++) {
            acc += (h_nindex[v + 1] - h_nindex[v]);
            if (acc >= target_per_chunk && (int)chunk_vstart.size() < NUM_CHUNKS) {
                chunk_vstart.push_back(v + 1);
                acc = 0;
            }
        }
        chunk_vstart.push_back(num_nodes);
        chunk_vstart.erase(unique(chunk_vstart.begin(), chunk_vstart.end()), chunk_vstart.end());
        int n_chunks = (int)chunk_vstart.size() - 1;

        long long max_chunk_edges = 0;
        for (int c = 0; c < n_chunks; c++) {
            long long e = h_nindex[chunk_vstart[c + 1]] - h_nindex[chunk_vstart[c]];
            if (e > max_chunk_edges) max_chunk_edges = e;
        }
        if (max_chunk_edges == 0) max_chunk_edges = 1;

        int* h_stage[2];
        cudaStream_t stream[2];
        CUDA_CHECK(cudaMallocHost(&h_stage[0], (size_t)max_chunk_edges * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_stage[1], (size_t)max_chunk_edges * sizeof(int)));
        CUDA_CHECK(cudaStreamCreate(&stream[0]));
        CUDA_CHECK(cudaStreamCreate(&stream[1]));

        for (int c = 0; c < n_chunks; c++) {
            int buf = c % 2;
            if (c >= 2) CUDA_CHECK(cudaStreamSynchronize(stream[buf]));

            int v_start = chunk_vstart[c], v_end = chunk_vstart[c + 1];
            long long base_off = h_nindex[v_start];
            #pragma omp parallel
            {
                vector<int> scratch;
                #pragma omp for schedule(dynamic, 4096)
                for (int v = v_start; v < v_end; v++) {
                    graph.neighbors_direct(v, scratch);
                    long long local_off = h_nindex[v] - base_off;
                    for (size_t i = 0; i < scratch.size(); i++) h_stage[buf][local_off + i] = scratch[i];
                }
            }
            long long chunk_edges = h_nindex[v_end] - base_off;
            CUDA_CHECK(cudaMemcpyAsync(d_nlist + base_off, h_stage[buf], (size_t)chunk_edges * sizeof(int),
                                        cudaMemcpyHostToDevice, stream[buf]));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream[0]));
        CUDA_CHECK(cudaStreamSynchronize(stream[1]));
        cudaStreamDestroy(stream[0]);
        cudaStreamDestroy(stream[1]);
        cudaFreeHost(h_stage[0]);
        cudaFreeHost(h_stage[1]);
    }

    auto t_pack1 = chrono::high_resolution_clock::now();

    long long* d_nindex; int* d_parent; int* d_changed;
    long long* h_nindex_view; int* h_parent_view; int* h_changed_view;
    cc_alloc_and_upload(&d_nindex, h_nindex, &h_nindex_view);

    vector<int> h_init(num_nodes);
    for (int v = 0; v < num_nodes; v++) h_init[v] = v;
    cc_alloc_and_upload(&d_parent, h_init, &h_parent_view);

    vector<int> zero_vec(1, 0);
    cc_alloc_and_upload(&d_changed, zero_vec, &h_changed_view);

    auto t_transfer1 = chrono::high_resolution_clock::now();

    int blocks = (num_nodes + CC_THREADS_PER_BLOCK - 1) / CC_THREADS_PER_BLOCK;
    int changed = 1;
    int rounds = 0;
    while (changed) {
        if (g_integrated) {
            *h_changed_view = 0;
        } else {
            changed = 0;
            CUDA_CHECK(cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice));
        }
        cc_hook_kernel<<<blocks, CC_THREADS_PER_BLOCK>>>(num_nodes, d_nindex, d_nlist, d_parent, d_changed);
        cc_shortcut_kernel<<<blocks, CC_THREADS_PER_BLOCK>>>(num_nodes, d_parent);
        if (g_integrated) {
            CUDA_CHECK(cudaDeviceSynchronize());
            changed = *h_changed_view;
        } else {
            CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        }
        rounds++;
    }

    for (int i = 0; i < 8; i++) cc_shortcut_kernel<<<blocks, CC_THREADS_PER_BLOCK>>>(num_nodes, d_parent);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_kernel1 = chrono::high_resolution_clock::now();

    if (g_integrated) {
        memcpy(parent.data(), h_parent_view, (size_t)num_nodes * sizeof(int));
        cudaFreeHost(h_nindex_view); cudaFreeHost(h_nlist_view);
        cudaFreeHost(h_parent_view); cudaFreeHost(h_changed_view);
    } else {
        CUDA_CHECK(cudaMemcpy(parent.data(), d_parent, (size_t)num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

        cudaFree(d_nindex); cudaFree(d_nlist); cudaFree(d_parent); cudaFree(d_changed);
    }

    auto t_end = chrono::high_resolution_clock::now();
    fprintf(stderr, "[CC-FALLBACK-DIAG] pack=%.1fms transfer=%.1fms kernel(%d rounds)=%.1fms d2h+free=%.1fms total_edges=%lld\n",
            chrono::duration<double, milli>(t_pack1 - t_pack0).count(),
            chrono::duration<double, milli>(t_transfer1 - t_pack1).count(),
            rounds, chrono::duration<double, milli>(t_kernel1 - t_transfer1).count(),
            chrono::duration<double, milli>(t_end - t_kernel1).count(),
            total_edges);
    fflush(stderr);
}

static int find_root(vector<int>& parent, int v) {
    int root = v;
    while (parent[root] != root) root = parent[root];
    while (parent[v] != root) {
        int next = parent[v];
        parent[v] = root;
        v = next;
    }
    return root;
}

struct Stats {
    long long insertions = 0, insertions_noop = 0, insertions_merged = 0;
    long long deletions = 0, deletions_noop = 0;
    long long deletions_still_connected = 0, deletions_definitive_split = 0, deletions_fallback = 0;
    long long deletions_fallback_deferred = 0;
    long long split_side_total_size = 0, split_side_max_size = 0;
    double total_batch_ms = 0;
    double fallback_ms = 0;
};

// Union-find over labels, never over vertices, so a split never reroutes other vertices.
struct CCLabels {
    vector<int> label;
    vector<int> lp;
    int find(int v) { return find_root(lp, label[v]); }
    bool unite_v(int u, int v) {
        int ru = find(u), rv = find(v);
        if (ru == rv) return false;
        if (ru < rv) lp[rv] = ru; else lp[ru] = rv;
        return true;
    }
    void split(const vector<int>& side) {
        int fresh = (int)lp.size();
        lp.push_back(fresh);
        for (int w : side) label[w] = fresh;
    }
    void reset_from_roots(const vector<int>& root) {
        label = root;
        lp.resize(label.size());
        for (size_t i = 0; i < lp.size(); i++) lp[i] = (int)i;
    }
    int count_components() {
        vector<char> seen(lp.size(), 0);
        int c = 0;
        for (size_t v = 0; v < label.size(); v++) {
            int r = find((int)v);
            if (!seen[r]) { seen[r] = 1; c++; }
        }
        return c;
    }
};

#ifndef CC_SELECT_CHUNK
#define CC_SELECT_CHUNK 4096
#endif

struct CcApp {
    HybridGraph& graph;
    CCLabels& cc;
    Stats& stats;
    Budget deletion_budget;

    struct Search {
        unordered_set<int> visited[2];
        vector<int> frontier[2], next;
        int side = 0;
        size_t pos = 0;
        bool met = false;
    };

    Budget budget_for(const Update&) { return deletion_budget; }

    // seed: an insertion unions labels (no search); a deletion searches from both endpoints.
    bool seed(const Update& e, Search& R, Budget& bud) {
        int u = e.u, v = e.v;
        if (e.ins) {
            stats.insertions++;
            if (cc.unite_v(u, v)) stats.insertions_merged++;
            else stats.insertions_noop++;
            return false;
        }
        stats.deletions++;
        if (cc.find(u) != cc.find(v)) { stats.deletions_noop++; return false; }
        R.visited[0].insert(u); R.frontier[0].push_back(u);
        R.visited[1].insert(v); R.frontier[1].push_back(v);
        if (u == v) R.met = true;
        bud.vertices = 2;
        return true;
    }

    bool resolved(const Search& R) const { return R.met || R.frontier[0].empty() || R.frontier[1].empty(); }

    // select: the side with the smaller frontier, chosen per level.
    vector<int> select(Search& R, Budget&) {
        if (R.pos == 0) R.side = (R.frontier[0].size() <= R.frontier[1].size()) ? 0 : 1;
        const vector<int>& F = R.frontier[R.side];
        size_t end = min(F.size(), R.pos + (size_t)CC_SELECT_CHUNK);
        return vector<int>(F.begin() + R.pos, F.begin() + end);
    }

    void expand(Search& R, const vector<int>& Y, const FetchResult& fr, Budget& bud) {
        unordered_set<int>& mine = R.visited[R.side];
        const unordered_set<int>& other = R.visited[1 - R.side];
        for (size_t i = 0; i < Y.size(); i++) {
            for (int n : fr[i]) {
                if (other.count(n)) { R.met = true; return; }
                if (!mine.insert(n).second) continue;
                R.next.push_back(n);
                bud.vertices++;
                if (bud.exceeded()) return;
            }
        }
        R.pos += Y.size();
        if (R.pos >= R.frontier[R.side].size()) {
            R.frontier[R.side] = std::move(R.next);
            R.next.clear();
            R.pos = 0;
        }
    }

    void commit(const Search& R) {
        if (R.met) { stats.deletions_still_connected++; return; }
        const unordered_set<int>& side = R.frontier[0].empty() ? R.visited[0] : R.visited[1];
        vector<int> touched(side.begin(), side.end());
        stats.deletions_definitive_split++;
        stats.split_side_total_size += (long long)touched.size();
        if ((long long)touched.size() > stats.split_side_max_size) stats.split_side_max_size = (long long)touched.size();
        cc.split(touched);
    }

    void fallback(const Search&) {
        if (g_defer_fallback) { stats.deletions_fallback_deferred++; return; }
        auto fb0 = chrono::high_resolution_clock::now();
        vector<int> roots(cc.label.size());
        full_recompute_cc_gpu(graph, roots, (int)roots.size());
        cc.reset_from_roots(roots);
        stats.fallback_ms += chrono::duration<double, milli>(chrono::high_resolution_clock::now() - fb0).count();
        stats.deletions_fallback++;
    }
};

int main(int argc, char* argv[]) {
    ScoutCliArgs args = parse_scout_cli(argc, argv, "cc");
    g_integrated = gpu_warmup_and_device_setup(args.gpu_id);
    if (getenv("SCOUT_FORCE_COPY")) g_integrated = false;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.gpu_id));

    g_defer_fallback = read_defer_fallback_env(true);
    cout << (g_defer_fallback
             ? "[Deferred fallback: ON (default)] Inconclusive deletions deferred; resolved by ONE "
               "catch-up full recompute after all batches, instead of one call each."
             : "[Deferred fallback: OFF (SCOUT_DEFER_FALLBACK=0)] Each inconclusive deletion "
               "triggers its own immediate full recompute.") << endl;

    auto setup_t0 = chrono::high_resolution_clock::now();

    HybridGraph graph;
    graph.open(args.egr_path.c_str());
    graph.enable_memo();
    int num_nodes = graph.num_nodes;

    long long total_degree = 0;
    total_degree = graph.base.edges;
    double avg_degree = num_nodes > 0 ? (double)total_degree / num_nodes : 0.0;
    g_use_exhaustive_search = (avg_degree < LOW_DEGREE_THRESHOLD);
    g_fallback_bound = make_fallback_bound(5000, LOW_DEGREE_THRESHOLD, avg_degree, (long long)num_nodes);
    print_run_header("cc", args.egr_path, num_nodes, total_degree, avg_degree, prop.name, g_integrated);
    cout << (g_use_exhaustive_search ? "EXHAUSTIVE search mode (no cap)" : "CAPPED search mode (bound=5000)")
         << " (avg_degree=" << fixed << setprecision(2) << avg_degree << " vs threshold=" << LOW_DEGREE_THRESHOLD << ")" << endl;

    CCLabels cc;
    int initial_components = 0;
    {
        vector<int> parent(num_nodes);
        for (int v = 0; v < num_nodes; v++) parent[v] = v;
        vector<long long> vals = read_ints_fast(args.context_path);
        for (int idx = 0; idx < num_nodes && idx < (int)vals.size(); idx++) parent[idx] = (int)vals[idx];
        vector<int> roots(num_nodes);
        for (int v = 0; v < num_nodes; v++) roots[v] = find_root(parent, v);
        cc.reset_from_roots(roots);
        initial_components = cc.count_components();
    }

    auto setup_t1 = chrono::high_resolution_clock::now();
    double setup_ms = chrono::duration<double, milli>(setup_t1 - setup_t0).count();

    vector<filesystem::path> batch_paths = list_batch_files_sorted(args.batch_dir);

    Stats stats;
    CcApp app{graph, cc, stats, Budget::from_env(g_fallback_bound)};
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
        auto bt0 = chrono::high_resolution_clock::now();
        nl_process_batch(app, graph, batch);
        double this_batch_ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - bt0).count();
        stats.total_batch_ms += this_batch_ms;
        print_batch_result(processed + 1, args.num_batches, (int)batch.size(), this_batch_ms,
            {{"insertions", to_string(stats.insertions - before.insertions)},
             {"deletions", to_string(stats.deletions - before.deletions)},
             {"fallback", to_string(stats.deletions_fallback - before.deletions_fallback)}});
        {

            double f = g_fetch.ms - fs0.ms;
            print_stage_times(processed + 1, f, 0.0, 0.0, max(0.0, this_batch_ms - f), 0.0);
            print_fetch_stats(processed + 1, g_fetch.pages - fs0.pages, process_major_faults() - mf0, g_fetch.lists - fs0.lists,
                              g_fetch.memo_hits - fs0.memo_hits);
        }
        processed++;
    }

    double deferred_fallback_ms = 0;
    if (g_defer_fallback && stats.deletions_fallback_deferred > 0) {
        cout << "\n[DEFERRED FALLBACK] resolving " << stats.deletions_fallback_deferred
             << " deferred inconclusive deletions with one recompute..." << endl;
        auto fb0 = chrono::high_resolution_clock::now();
        vector<int> roots(num_nodes);
        full_recompute_cc_gpu(graph, roots, num_nodes);
        cc.reset_from_roots(roots);
        auto fb1 = chrono::high_resolution_clock::now();
        deferred_fallback_ms = chrono::duration<double, milli>(fb1 - fb0).count();
        cout << "[DEFERRED FALLBACK] done in " << fixed << setprecision(3) << deferred_fallback_ms << " ms" << endl;
    }

    int num_components = cc.count_components();

    cout << "Pages requested (Alg. 1): " << g_fetch.pages << " | lists reconciled: " << g_fetch.lists
         << " | memo hits: " << g_fetch.memo_hits << endl;
    cout << "Insertions: " << stats.insertions << " (noop=" << stats.insertions_noop
         << ", merged=" << stats.insertions_merged << ")" << endl;
    cout << "Deletions:  " << stats.deletions << " (noop=" << stats.deletions_noop
         << ", still_connected=" << stats.deletions_still_connected
         << ", definitive_split=" << stats.deletions_definitive_split
         << ", fallback=" << stats.deletions_fallback
         << ", deferred=" << stats.deletions_fallback_deferred << ")" << endl;
    cout << "  fallback (full_recompute_cc, immediate): " << fixed << setprecision(3) << stats.fallback_ms << " ms"
         << " | deferred catch-up: " << deferred_fallback_ms << " ms" << endl;

    double processing_ms = stats.total_batch_ms + deferred_fallback_ms;
    double e2e_ms = setup_ms + processing_ms;
    print_final_summary(setup_ms, processing_ms, e2e_ms,
                         "components", initial_components,
                         "components", num_components);

    if (!g_touched.empty()) dump_touched_at_exit();
    std::cout.flush();
    fflush(stdout);
    std::_Exit(0);

}

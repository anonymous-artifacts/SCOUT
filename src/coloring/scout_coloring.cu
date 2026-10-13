// SCOUT graph coloring (LCL, K = 1) on the LCL driver. Seed = the higher-id endpoint of an
// inserted edge whose endpoints share a color; deletions never break a proper coloring.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <unordered_set>
#include <omp.h>

#include "../../core/FlashGraph.h"
#include "../../core/gpu_common.h"
#include "../../core/cli_setup.h"
#include "../../core/reporting.h"
#include "../../core/fetch.h"
#include "../../core/lcl_driver.h"

using namespace std;

#define THREADS_PER_BLOCK 256

static bool g_integrated = false;

// Repair: one thread per region; seeds in increasing id, each taking the smallest
// color absent from its neighborhood.
__global__ void gc_region_repair_kernel(const int* region_offsets, int num_regions,
                                        const int* seed_rows, const int* row_vertex,
                                        const long long* row_offsets, const int* row_elems,
                                        int* d_color, int* new_color_out) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_regions) return;
    for (int k = region_offsets[r]; k < region_offsets[r + 1]; k++) {
        int row = seed_rows[k];
        long long s = row_offsets[row], e = row_offsets[row + 1];
        int c = 0;
        bool found = false;
        while (!found) {
            found = true;
            for (long long i = s; i < e; i++) {
                if (d_color[row_elems[i]] == c) { found = false; break; }
            }
            if (!found) c++;
        }
        d_color[row_vertex[row]] = c;
        new_color_out[row] = c;
    }
}

template <typename T>
static void grow_pair(T** h, T** d, size_t& cap, size_t need) {
    if (need <= cap) return;
    size_t nc = max(need, (size_t)(cap * 1.5) + 1);
    if (*h) CUDA_CHECK(cudaFreeHost(*h));
    if (!g_integrated && *d) CUDA_CHECK(cudaFree(*d));
    alloc_pair(h, d, nc, g_integrated);
    cap = nc;
}

struct Stats {
    long long insertions = 0, insertions_noop = 0, insertions_conflict = 0;
    long long deletions = 0;
};

struct GcApp {
    static constexpr int K = 1;
    bool seed_needs_state = true;

    HybridGraph& g;
    vector<int>& color;
    Stats& stats;
    int* d_color = nullptr;
    int* h_color_view = nullptr;
    cudaStream_t stream;
    cudaEvent_t t0, t1, t2, t3;

    int *h_rowv = nullptr, *d_rowv = nullptr;         size_t rowv_cap = 0;
    long long *h_off = nullptr, *d_off = nullptr;     size_t off_cap = 0;
    int *h_elems = nullptr, *d_elems = nullptr;       size_t elems_cap = 0;
    int *h_roff = nullptr, *d_roff = nullptr;         size_t roff_cap = 0;
    int *h_srows = nullptr, *d_srows = nullptr;       size_t srows_cap = 0;
    int *h_newc = nullptr, *d_newc = nullptr;         size_t newc_cap = 0;
    int pending_rows = 0;
    double pack_ms = 0;
    float h2d_ms = 0, repair_ms = 0, commit_ms = 0;

    GcApp(HybridGraph& g_, vector<int>& color_, Stats& s) : g(g_), color(color_), stats(s) {
        int n = (int)color.size();
        if (g_integrated) {
            CUDA_CHECK(cudaHostAlloc(&h_color_view, (size_t)n * sizeof(int), cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(&d_color, h_color_view, 0));
            memcpy(h_color_view, color.data(), (size_t)n * sizeof(int));
        } else {
            CUDA_CHECK(cudaMalloc(&d_color, (size_t)n * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_color, color.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventCreate(&t2)); CUDA_CHECK(cudaEventCreate(&t3));
        const size_t V0 = (size_t)1 << 17;
        grow_pair(&h_rowv, &d_rowv, rowv_cap, V0);
        grow_pair(&h_off, &d_off, off_cap, V0 + 1);
        grow_pair(&h_elems, &d_elems, elems_cap, (size_t)1 << 23);
        grow_pair(&h_roff, &d_roff, roff_cap, V0 + 1);
        grow_pair(&h_srows, &d_srows, srows_cap, V0);
        grow_pair(&h_newc, &d_newc, newc_cap, V0);
    }

    void seed(const vector<Update>& B, vector<int>& S) {
        unordered_set<int> seen;
        for (const Update& e : B) {
            if (!e.ins) { stats.deletions++; continue; }
            stats.insertions++;
            if (color[e.u] != color[e.v]) { stats.insertions_noop++; continue; }
            stats.insertions_conflict++;
            int loser = max(e.u, e.v);
            if (seen.insert(loser).second) S.push_back(loser);
        }
    }

    // Before the previous batch is committed: endpoints that conflict under the current colors.
    void prefetch_hint(const vector<Update>& B, vector<int>& H) {
        unordered_set<int> seen;
        for (const Update& e : B)
            if (e.ins && color[e.u] == color[e.v] && seen.insert(max(e.u, e.v)).second) H.push_back(max(e.u, e.v));
    }
    void link(const vector<Update>&, vector<pair<int, int>>&) {}

    void launch(int, const vector<Update>&, LclPlan& plan) {
        auto ts = lcl_clock::now();
        const size_t n = plan.seeds.size();
        pending_rows = (int)n;
        if (n == 0) { pack_ms = lcl_ms_since(ts); return; }
        FetchResult fr;
        fetch_frontier(g, plan.seeds, fr);
        long long total = 0;
        for (size_t i = 0; i < n; i++) total += (long long)fr[i].size();
        int R = plan.num_regions;
        grow_pair(&h_rowv, &d_rowv, rowv_cap, n);
        grow_pair(&h_off, &d_off, off_cap, n + 1);
        grow_pair(&h_elems, &d_elems, elems_cap, (size_t)max(total, 1LL));
        grow_pair(&h_roff, &d_roff, roff_cap, (size_t)R + 1);
        grow_pair(&h_srows, &d_srows, srows_cap, n);
        grow_pair(&h_newc, &d_newc, newc_cap, n);
        long long o = 0;
        h_off[0] = 0;
        for (size_t i = 0; i < n; i++) {
            h_rowv[i] = plan.seeds[i];
            memcpy(h_elems + o, fr[i].data(), fr[i].size() * sizeof(int));
            o += (long long)fr[i].size();
            h_off[i + 1] = o;
        }

        for (int r = 0; r <= R; r++) h_roff[r] = plan.region_offsets[r];
        for (int r = 0; r < R; r++) {
            int a = plan.region_offsets[r], b = plan.region_offsets[r + 1];
            for (int k = a; k < b; k++) h_srows[k] = plan.region_seeds[k];
            sort(h_srows + a, h_srows + b, [&](int x, int y) { return plan.seeds[x] < plan.seeds[y]; });
        }
        pack_ms = lcl_ms_since(ts);

        CUDA_CHECK(cudaEventRecord(t0, stream));
        if (!g_integrated) {
            CUDA_CHECK(cudaMemcpyAsync(d_rowv, h_rowv, n * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_off, h_off, (n + 1) * sizeof(long long), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_elems, h_elems, (size_t)o * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_roff, h_roff, (size_t)(R + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_srows, h_srows, n * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaEventRecord(t1, stream));
        gc_region_repair_kernel<<<(R + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(
            d_roff, R, d_srows, d_rowv, d_off, d_elems, d_color, d_newc);
        CUDA_CHECK(cudaEventRecord(t2, stream));
        if (!g_integrated)
            CUDA_CHECK(cudaMemcpyAsync(h_newc, d_newc, n * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(t3, stream));
    }

    // Copy the seeds' new colors back to the host copy of the coloring.
    void collect(int) {
        h2d_ms = repair_ms = commit_ms = 0;
        if (pending_rows == 0) return;
        CUDA_CHECK(cudaEventSynchronize(t3));
        for (int i = 0; i < pending_rows; i++) color[h_rowv[i]] = h_newc[i];
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));
        CUDA_CHECK(cudaEventElapsedTime(&repair_ms, t1, t2));
        CUDA_CHECK(cudaEventElapsedTime(&commit_ms, t2, t3));
        pending_rows = 0;
    }
};

int main(int argc, char* argv[]) {
    ScoutCliArgs args = parse_scout_cli(argc, argv, "coloring");
    g_integrated = gpu_warmup_and_device_setup(args.gpu_id);
    if (getenv("SCOUT_FORCE_COPY")) g_integrated = false;
    bool no_overlap = getenv("SCOUT_NO_OVERLAP") != nullptr;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.gpu_id));

    auto setup_t0 = chrono::high_resolution_clock::now();

    HybridGraph graph;
    graph.open(args.egr_path.c_str());
    graph.enable_memo();
    int num_nodes = graph.num_nodes;

    long long total_degree = graph.base.edges;
    double avg_degree = num_nodes > 0 ? (double)total_degree / num_nodes : 0.0;
    print_run_header("coloring", args.egr_path, num_nodes, total_degree, avg_degree, prop.name, g_integrated);

    vector<int> color(num_nodes, 0);
    int initial_colors_used = 0;
    {
        vector<long long> vals = read_ints_fast(args.context_path);
        for (int idx = 0; idx < num_nodes && idx < (int)vals.size(); idx++) color[idx] = (int)vals[idx];
        int max_c = -1;
        for (int v = 0; v < num_nodes; v++) if (color[v] > max_c) max_c = color[v];
        initial_colors_used = max_c + 1;
    }

    vector<filesystem::path> batch_paths = list_batch_files_sorted(args.batch_dir);
    int nb = min(args.num_batches, (int)batch_paths.size());

    auto load = [&](int k) {
        vector<EdgeUpdate> raw = parse_batch_file(batch_paths[k], graph, num_nodes);
        vector<Update> b;
        b.reserve(raw.size());
        for (const auto& e : raw) b.push_back({e.u, e.v, e.is_insertion});
        return b;
    };

    Stats stats;
    GcApp app(graph, color, stats);

    double setup_ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - setup_t0).count();

    LclDriver<GcApp> driver(graph, app);
    driver.overlap = !no_overlap;
    cout << "[LCL driver] K=" << GcApp::K << ", overlap " << (driver.overlap ? "on" : "off (SCOUT_NO_OVERLAP)")
         << ", memo " << (graph.memo_on ? "on" : "off (SCOUT_NO_MEMO)") << endl;

    Stats before = stats;
    double processing_ms = driver.run(nb, load, [&](int k, double wall_ms) {
        const LclBatchTimes& t = driver.times[k];
        print_batch_result(k + 1, nb, (int)driver.batches[k].size(), wall_ms,
            {{"insertions", to_string(stats.insertions - before.insertions)},
             {"conflicts", to_string(stats.insertions_conflict - before.insertions_conflict)},
             {"regions", to_string(t.regions)}});
        print_stage_times(k + 1, t.fetch_ms, t.record_ms + t.discover_ms + t.merge_ms + app.pack_ms,
                          app.h2d_ms, app.repair_ms, app.commit_ms);
        print_fetch_stats(k + 1, t.pages, t.faulted, t.lists, t.memo_hits);
        before = stats;
    });

    int colors_used;
    {
        int max_color = -1;
        for (int v = 0; v < num_nodes; v++) if (color[v] > max_color) max_color = color[v];
        colors_used = max_color + 1;
    }

    // Full check of the final coloring against the final graph (not timed).
    if (getenv("SCOUT_VALIDATE")) {
        long long conflicts = 0;
        #pragma omp parallel reduction(+:conflicts)
        {
            vector<int> nb;
            #pragma omp for schedule(dynamic, 4096)
            for (int v = 0; v < num_nodes; v++) {
                graph.neighbors_direct(v, nb);
                for (int u : nb) if (u > v && color[u] == color[v]) conflicts++;
            }
        }
        cout << "[VALIDATE] coloring conflicts=" << conflicts << endl;
    }

    cout << "Insertions: " << stats.insertions << " (noop=" << stats.insertions_noop
         << ", conflict=" << stats.insertions_conflict << ")" << endl;
    cout << "Deletions:  " << stats.deletions << " (always a structural noop for coloring)" << endl;
    cout << "Pages requested (Alg. 1): " << g_fetch.pages << " | lists reconciled: " << g_fetch.lists
         << " | memo hits: " << g_fetch.memo_hits << endl;

    double e2e_ms = setup_ms + processing_ms;
    print_final_summary(setup_ms, processing_ms, e2e_ms,
                        "colors used", initial_colors_used,
                        "colors used", colors_used);

    if (!g_touched.empty()) dump_touched_at_exit();
    std::cout.flush();
    fflush(stdout);
    std::_Exit(0);
}

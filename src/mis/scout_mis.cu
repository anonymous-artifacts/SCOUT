// SCOUT MIS (LCL, K = 2) on the LCL driver. Seed = update endpoints; each vertex-disjoint
// 2-hop region is repaired by one GPU thread, applying its updates in batch order.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <omp.h>
#include <malloc.h>

#include "../../core/FlashGraph.h"
#include "../../core/flash_compactor.h"
#include "../../core/gpu_common.h"
#include "../../core/cli_setup.h"
#include "../../core/reporting.h"
#include "../../core/fetch.h"
#include "../../core/lcl_driver.h"

using namespace std;

#define THREADS_PER_BLOCK 256

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

static void gpu_memory_sample() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
    const double total_mb = (double)total_bytes / (1024.0 * 1024.0);
    const double used_mb  = total_mb - (double)free_bytes / (1024.0 * 1024.0);
    double delta_mb = used_mb - g_baseline_used_mb;
    if (delta_mb < 0.0) delta_mb = 0.0;
    if (delta_mb > g_peak_delta_mb) g_peak_delta_mb = delta_mb;
    g_gpu_total_mb = total_mb;
}

struct __align__(16) DeviceEdge {
    int source;
    int destination;
    bool isInsertion;
};

__global__ void build_vertex_map_kernel(const int* row_vertices, int num_rows, int* vertex_to_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) vertex_to_row[row_vertices[i]] = i;
}

__global__ void clear_vertex_map_kernel(const int* row_vertices, int num_rows, int* vertex_to_row) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) vertex_to_row[row_vertices[i]] = -1;
}

// Repair: one thread per region; no atomics, since regions are vertex-disjoint.
// A seed's row is its 2-hop region; a depth-1 vertex's row is its adjacency.
__global__ void mis_region_repair_kernel(const DeviceEdge* edges, const int* region_offsets,
                                         const int* region_edges, int num_regions, char* member,
                                         const long long* row_offsets, const int* row_elems,
                                         const int* vertex_to_row) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_regions) return;
    for (int k = region_offsets[r]; k < region_offsets[r + 1]; k++) {
        const DeviceEdge& edge = edges[region_edges[k]];
        int u = edge.source;
        int v = edge.destination;
        bool u_in_mis = member[u];
        bool v_in_mis = member[v];

        if (edge.isInsertion) {
            if (u_in_mis && v_in_mis) {
                int to_remove = (u > v) ? u : v;
                member[to_remove] = 0;
                int idx = vertex_to_row[to_remove];
                if (idx != -1) {
                    for (long long j = row_offsets[idx]; j < row_offsets[idx + 1]; j++) {
                        int cand = row_elems[j];
                        if (member[cand]) continue;
                        int cidx = vertex_to_row[cand];
                        if (cidx == -1) continue;
                        bool can_add = true;
                        for (long long q = row_offsets[cidx]; q < row_offsets[cidx + 1]; q++) {
                            if (member[row_elems[q]]) { can_add = false; break; }
                        }
                        if (can_add) { member[cand] = 1; break; }
                    }
                }
            }
        } else {
            if (u_in_mis != v_in_mis) {
                int in = u_in_mis ? u : v;
                int out = u_in_mis ? v : u;
                bool can_add = true;
                int idx = vertex_to_row[out];
                if (idx != -1) {
                    for (long long j = row_offsets[idx]; j < row_offsets[idx + 1]; j++) {
                        int nb = row_elems[j];
                        if (nb != in && member[nb]) { can_add = false; break; }
                    }
                }
                if (can_add) member[out] = 1;
            }
        }
    }
}

template <typename T>
static void grow_pair(T** h, T** d, size_t& cap, size_t need) {
    if (need <= cap) return;
    size_t nc = max(need, (size_t)(cap * 1.5) + 1);
    if (*h) CUDA_CHECK(cudaFreeHost(*h));
    if (*d) CUDA_CHECK(cudaFree(*d));
    CUDA_CHECK(cudaMallocHost(h, nc * sizeof(T)));
    CUDA_CHECK(cudaMalloc(d, nc * sizeof(T)));
    cap = nc;
}

// Two channels: batch k+1 is packed into one while batch k is repaired from the other.
struct MisApp {
    static constexpr int K = 2;
    bool seed_needs_state = false;

    HybridGraph& g;
    int n;
    char* d_member = nullptr;
    int* d_vertex_to_row = nullptr;
    cudaStream_t stream;

    struct Channel {
        DeviceEdge *h_edges = nullptr, *d_edges = nullptr;  size_t edges_cap = 0;
        int *h_rows = nullptr, *d_rows = nullptr;           size_t rows_cap = 0;
        long long *h_off = nullptr, *d_off = nullptr;       size_t off_cap = 0;
        int *h_elems = nullptr, *d_elems = nullptr;         size_t elems_cap = 0;
        int *h_roff = nullptr, *d_roff = nullptr;           size_t roff_cap = 0;
        int *h_redges = nullptr, *d_redges = nullptr;       size_t redges_cap = 0;
        cudaEvent_t done, t0, t1, t2;
        int num_rows = 0;
    } ch[2];

    struct BatchGpu { int channel = 0; double pack_ms = 0; float h2d_ms = 0, repair_ms = 0; long long elems = 0; };
    vector<BatchGpu> info;

    MisApp(HybridGraph& g_, int n_, long long initial_elems, int num_batches) : g(g_), n(n_) {
        CUDA_CHECK(cudaMalloc(&d_member, (size_t)n));
        CUDA_CHECK(cudaMalloc(&d_vertex_to_row, (size_t)n * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_vertex_to_row, 0xFF, (size_t)n * sizeof(int)));
        CUDA_CHECK(cudaStreamCreate(&stream));
        for (auto& c : ch) {
            grow_pair(&c.h_elems, &c.d_elems, c.elems_cap, (size_t)initial_elems);
            grow_pair(&c.h_edges, &c.d_edges, c.edges_cap, (size_t)1 << 17);
            grow_pair(&c.h_rows, &c.d_rows, c.rows_cap, (size_t)1 << 20);
            grow_pair(&c.h_off, &c.d_off, c.off_cap, ((size_t)1 << 20) + 1);
            grow_pair(&c.h_roff, &c.d_roff, c.roff_cap, ((size_t)1 << 17) + 1);
            grow_pair(&c.h_redges, &c.d_redges, c.redges_cap, (size_t)1 << 17);
            CUDA_CHECK(cudaEventCreate(&c.done)); CUDA_CHECK(cudaEventCreate(&c.t0));
            CUDA_CHECK(cudaEventCreate(&c.t1)); CUDA_CHECK(cudaEventCreate(&c.t2));
            CUDA_CHECK(cudaEventRecord(c.done, stream));
        }
        info.assign(num_batches, BatchGpu{});
    }

    void set_mis(const vector<char>& mis) {
        CUDA_CHECK(cudaMemcpy(d_member, mis.data(), (size_t)n, cudaMemcpyHostToDevice));
    }
    vector<char> get_mis() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        vector<char> mis(n);
        CUDA_CHECK(cudaMemcpy(mis.data(), d_member, (size_t)n, cudaMemcpyDeviceToHost));
        return mis;
    }

    void seed(const vector<Update>& B, vector<int>& S) {
        unordered_set<int> seen;
        for (const Update& e : B) {
            if (seen.insert(e.u).second) S.push_back(e.u);
            if (seen.insert(e.v).second) S.push_back(e.v);
        }
    }
    void prefetch_hint(const vector<Update>& B, vector<int>& H) { seed(B, H); }

    // An update reads and writes both endpoints, so it ties their regions together.
    void link(const vector<Update>& B, vector<pair<int, int>>& L) {
        for (const Update& e : B) L.emplace_back(e.u, e.v);
    }

    // Pack rows and per-region update lists, transfer, and launch the repair.
    void launch(int k, const vector<Update>& B, LclPlan& plan) {
        auto t0 = lcl_clock::now();
        int c_id = k % 2;
        Channel& c = ch[c_id];
        CUDA_CHECK(cudaEventSynchronize(c.done));

        vector<int> extra;
        for (int v : plan.inner) if (!plan.seed_index.count(v)) extra.push_back(v);
        FetchResult fr;
        fetch_frontier(g, extra, fr);

        size_t num_rows = plan.seeds.size() + extra.size();
        long long total = 0;
        for (auto& b : plan.balls) total += (long long)b.size();
        for (size_t i = 0; i < extra.size(); i++) total += (long long)fr[i].size();
        grow_pair(&c.h_rows, &c.d_rows, c.rows_cap, num_rows);
        grow_pair(&c.h_off, &c.d_off, c.off_cap, num_rows + 1);
        grow_pair(&c.h_elems, &c.d_elems, c.elems_cap, (size_t)max(total, 1LL));
        long long o = 0;
        size_t r = 0;
        c.h_off[0] = 0;
        auto put_row = [&](int v, const vector<int>& row) {
            c.h_rows[r] = v;
            memcpy(c.h_elems + o, row.data(), row.size() * sizeof(int));
            o += (long long)row.size();
            c.h_off[++r] = o;
        };
        for (size_t i = 0; i < plan.seeds.size(); i++) put_row(plan.seeds[i], plan.balls[i]);
        for (size_t i = 0; i < extra.size(); i++) put_row(extra[i], fr[i]);
        c.num_rows = (int)num_rows;

        size_t m = B.size();
        int R = plan.num_regions;
        grow_pair(&c.h_edges, &c.d_edges, c.edges_cap, max(m, (size_t)1));
        grow_pair(&c.h_roff, &c.d_roff, c.roff_cap, (size_t)R + 1);
        grow_pair(&c.h_redges, &c.d_redges, c.redges_cap, max(m, (size_t)1));
        vector<int> region_of_edge(m);
        for (int i = 0; i <= R; i++) c.h_roff[i] = 0;
        for (size_t j = 0; j < m; j++) {
            c.h_edges[j] = DeviceEdge{B[j].u, B[j].v, B[j].ins};
            region_of_edge[j] = plan.region_of[plan.seed_index.at(B[j].u)];
            c.h_roff[region_of_edge[j] + 1]++;
        }
        for (int i = 0; i < R; i++) c.h_roff[i + 1] += c.h_roff[i];
        vector<int> fill(c.h_roff, c.h_roff + R);
        for (size_t j = 0; j < m; j++) c.h_redges[fill[region_of_edge[j]]++] = (int)j;
        info[k].channel = c_id;
        info[k].elems = o;
        info[k].pack_ms = lcl_ms_since(t0);

        CUDA_CHECK(cudaEventRecord(c.t0, stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_edges, c.h_edges, m * sizeof(DeviceEdge), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_rows, c.h_rows, num_rows * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_off, c.h_off, (num_rows + 1) * sizeof(long long), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_elems, c.h_elems, (size_t)o * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_roff, c.h_roff, (size_t)(R + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(c.d_redges, c.h_redges, m * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(c.t1, stream));
        int bt = THREADS_PER_BLOCK;
        int brows = ((int)num_rows + bt - 1) / bt;
        if (num_rows > 0) build_vertex_map_kernel<<<brows, bt, 0, stream>>>(c.d_rows, (int)num_rows, d_vertex_to_row);
        if (R > 0) mis_region_repair_kernel<<<(R + bt - 1) / bt, bt, 0, stream>>>(
            c.d_edges, c.d_roff, c.d_redges, R, d_member, c.d_off, c.d_elems, d_vertex_to_row);
        if (num_rows > 0) clear_vertex_map_kernel<<<brows, bt, 0, stream>>>(c.d_rows, (int)num_rows, d_vertex_to_row);
        CUDA_CHECK(cudaEventRecord(c.t2, stream));
        CUDA_CHECK(cudaEventRecord(c.done, stream));
    }

    void collect(int k) {
        Channel& c = ch[info[k].channel];
        CUDA_CHECK(cudaEventSynchronize(c.done));
        CUDA_CHECK(cudaEventElapsedTime(&info[k].h2d_ms, c.t0, c.t1));
        CUDA_CHECK(cudaEventElapsedTime(&info[k].repair_ms, c.t1, c.t2));
    }
};

int main(int argc, char* argv[]) {
    #ifdef M_MMAP_MAX
    mallopt(M_MMAP_MAX, 0);
    #endif

    ScoutCliArgs args = parse_scout_cli(argc, argv, "mis");
    auto setup_t0 = chrono::high_resolution_clock::now();
    cout << "=========================================START=====================================================" << endl;
    cout << "======================================= SCOUT MIS =========================================" << endl;

    string graph_path = args.egr_path;
    if (graph_path.size() < 4 || graph_path.substr(graph_path.size() - 4) != ".egr") {
        cerr << "[Warning] " << graph_path << " does not end in .egr — this binary expects a pre-built binary CSR file." << endl;
        cerr << "          If this fails to open, run: python3 tools/efficient_converter.py <graph.mtx> <graph.egr>" << endl;
    }

    bool no_overlap = getenv("SCOUT_NO_OVERLAP") != nullptr;
    bool integrated = gpu_warmup_and_device_setup(args.gpu_id);
    gpu_memory_set_baseline();

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, args.gpu_id));

    HybridGraph graph;
    graph.open(args.egr_path.c_str());
    graph.enable_memo();
    int num_nodes = graph.num_nodes;

    double avg_degree = num_nodes > 0 ? (double)graph.base.edges / num_nodes : 1.0;
    print_run_header("mis", args.egr_path, num_nodes, graph.base.edges, avg_degree, deviceProp.name, integrated);

    vector<filesystem::path> batch_paths = list_batch_files_sorted(args.batch_dir);
    vector<vector<Update>> batches;
    for (const auto& path : batch_paths) {
        if ((int)batches.size() >= args.num_batches) break;
        vector<EdgeUpdate> raw = parse_batch_file(path, graph, num_nodes);
        vector<Update> b;
        b.reserve(raw.size());
        for (const auto& e : raw) b.push_back({e.u, e.v, e.is_insertion});
        batches.push_back(move(b));
    }
    int nb = (int)batches.size();

    long long initial_elems = (long long)(2.0 * avg_degree * avg_degree * 4096);
    initial_elems = max(1000000LL, min(initial_elems, 64000000LL));
    MisApp app(graph, num_nodes, initial_elems, nb);
    gpu_memory_sample();

    int initial_card = 0;
    vector<char> mis(num_nodes, 0);
    for (long long id : read_ints_fast(args.context_path)) {
        int node_id = (int)id - 1;
        if (node_id >= 0 && node_id < num_nodes && !mis[node_id]) { mis[node_id] = 1; initial_card++; }
    }
    app.set_mis(mis);

    double setup_ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - setup_t0).count();

    LclDriver<MisApp> driver(graph, app);
    driver.overlap = !no_overlap;
    driver.guarded_record = true;
    cout << "[LCL driver] K=" << MisApp::K << ", overlap " << (driver.overlap ? "on" : "off (SCOUT_NO_OVERLAP)")
         << ", memo " << (graph.memo_on ? "on" : "off (SCOUT_NO_MEMO)") << endl;

    double processing_ms = driver.run(nb, [&](int k) { return move(batches[k]); }, [&](int k, double wall_ms) {
        const LclBatchTimes& t = driver.times[k];
        const MisApp::BatchGpu& gi = app.info[k];
        print_batch_result(k + 1, nb, (int)driver.batches[k].size(), wall_ms,
                           {{"regions", to_string(t.regions)}, {"packed_elems", to_string(gi.elems)}});
        print_stage_times(k + 1, t.fetch_ms, t.record_ms + t.discover_ms + t.merge_ms + gi.pack_ms,
                          gi.h2d_ms, gi.repair_ms, 0.0);
        print_fetch_stats(k + 1, t.pages, t.faulted, t.lists, t.memo_hits);
        gpu_memory_sample();
    });

    vector<char> final_mis = app.get_mis();
    int updated_card = 0;
    for (int i = 0; i < num_nodes; i++) if (final_mis[i]) updated_card++;

    // Full check of the final MIS against the final graph (not timed).
    if (getenv("SCOUT_VALIDATE")) {
        long long not_independent = 0, not_maximal = 0;
        #pragma omp parallel reduction(+:not_independent, not_maximal)
        {
            vector<int> nb;
            #pragma omp for schedule(dynamic, 4096)
            for (int v = 0; v < num_nodes; v++) {
                graph.neighbors_direct(v, nb);
                bool dominated = false;
                for (int u : nb) {
                    if (final_mis[u]) dominated = true;
                    if (final_mis[v] && final_mis[u] && u > v) not_independent++;
                }
                if (!final_mis[v] && !dominated) not_maximal++;
            }
        }
        cout << "[VALIDATE] independence violations=" << not_independent
             << " maximality violations=" << not_maximal << endl;
    }

    {
        string out_dir = "datasets/Final_MISs";
        filesystem::create_directories(out_dir);
        string graph_name = filesystem::path(graph_path).stem().string();
        string final_mis_file = out_dir + "/" + graph_name + "_flash_final_mis.txt";
        ofstream out_file(final_mis_file);
        if (out_file.is_open()) {
            for (int i = 0; i < num_nodes; i++) if (final_mis[i]) out_file << (i + 1) << "\n";
            cout << "Saved final MIS to " << final_mis_file << endl;
        } else {
            cerr << "Warning: Could not open file to save final MIS: " << final_mis_file << endl;
        }
    }

    bool skip_compaction = false;
    {
        const char* skip_env = getenv("SCOUT_SKIP_COMPACTION");
        skip_compaction = (skip_env != nullptr) && (string(skip_env) == "1");
    }
    double compact_time_ms = 0.0;
    if (!skip_compaction) {
        auto compact_start = chrono::high_resolution_clock::now();
        compact_flash_graph(graph.base, graph.delta, graph.flash_path);
        graph.base.remap(graph.flash_path.c_str());
        graph.delta.clear();
        compact_time_ms = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - compact_start).count();
    } else {
        cout << "\n[SCOUT_SKIP_COMPACTION=1] Skipping end-of-cycle compaction — on-disk .egr left untouched." << endl;
    }

    int processed = max(nb, 1);
    cout << "\n" << string(80, '=') << endl;
    cout << "=== FINAL RESULTS ===" << endl;
    cout << "  Initial MIS cardinality: " << initial_card << endl;
    cout << "  Final MIS cardinality:   " << updated_card << endl;
    cout << "  Number of batches:       " << nb << endl;
    cout << fixed << setprecision(3);
    cout << "  Total processing time:   " << processing_ms << " milliseconds" << endl;
    cout << "  Average time per batch:  " << processing_ms / processed << " milliseconds" << endl;
    cout << "  Pages requested (Alg. 1): " << g_fetch.pages << " | lists reconciled: " << g_fetch.lists
         << " | memo hits: " << g_fetch.memo_hits << endl;
    cout << "\nEnd-of-cycle compaction time (NOT counted in per-batch numbers above): " << compact_time_ms << " ms" << endl;
    gpu_memory_sample();
    double peak_pct = (g_gpu_total_mb > 0.0) ? (g_peak_delta_mb / g_gpu_total_mb * 100.0) : 0.0;
    cout << "Peak program GPU memory: " << setprecision(2) << g_peak_delta_mb << " MB ("
         << setprecision(1) << peak_pct << "% of " << setprecision(2) << g_gpu_total_mb << " MB total)" << endl;
    cout << string(80, '=') << endl;

    double e2e_ms = setup_ms + processing_ms + compact_time_ms;
    print_final_summary(setup_ms, processing_ms, e2e_ms,
                        "MIS size", initial_card,
                        "MIS size", updated_card);

    if (!g_touched.empty()) dump_touched_at_exit();
    std::cout.flush();
    fflush(stdout);
    std::_Exit(0);
}

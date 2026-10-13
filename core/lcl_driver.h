// Maintenance for locally checkable problems (fixed repair radius K), and the CPU-GPU pipeline.

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <chrono>
#include <cstdint>
#include <omp.h>

#include "FlashGraph.h"
#include "fetch.h"
#include "problem.h"

using lcl_clock = std::chrono::high_resolution_clock;
inline double lcl_ms_since(lcl_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(lcl_clock::now() - t0).count();
}

// What the runtime hands to the application for one batch: seeds, their regions
// the K-hop region of each seed, and the vertex-disjoint grouping of those regions.
struct LclPlan {
    std::vector<int> seeds;
    std::unordered_map<int, int> seed_index;
    std::vector<std::vector<int>> balls;
    std::vector<int> inner;
    std::vector<int> region_of;
    int num_regions = 0;
    std::vector<int> region_offsets;
    std::vector<int> region_seeds;
};

// Per-vertex scratch, allocated once and reset through the touched entries only.
struct LclScratch {
    std::vector<int> pos;
    std::vector<int> owner;
    std::vector<std::vector<uint64_t>> marks;
    void ensure(int n) {
        if ((int)pos.size() != n) { pos.assign(n, -1); owner.assign(n, -1); marks.clear(); }
    }

    void ensure_marks(int T, int n) {
        if ((int)marks.size() < T) marks.resize(T);
        for (int t = 0; t < T; t++)
            if (marks[t].empty()) marks[t].assign(((size_t)n + 63) / 64, 0);
    }
};

// K rounds, each one fetch_frontier call over the union of all seeds' frontiers;
// then each seed's region is assembled from the fetched lists.
inline void lcl_discover(HybridGraph& g, int K, LclPlan& plan, LclScratch& sc) {
    const int n = g.num_nodes;
    const size_t ns = plan.seeds.size();
    sc.ensure(n);
    plan.balls.assign(ns, {});
    plan.inner.clear();
    std::vector<const std::vector<int>*> lists;
    std::vector<FetchResult> rounds(K);
    std::vector<int> frontier;
    for (int x : plan.seeds)
        if (sc.pos[x] < 0) { sc.pos[x] = (int)plan.inner.size(); plan.inner.push_back(x); frontier.push_back(x); }
    for (int d = 1; d <= K && !frontier.empty(); d++) {
        FetchResult& fr = rounds[d - 1];
        fetch_frontier(g, frontier, fr, true, d == 1);
        for (size_t i = 0; i < frontier.size(); i++) lists.push_back(&fr[i]);
        if (d == K) break;
        std::vector<int> next;
        for (size_t i = 0; i < frontier.size(); i++)
            for (int nb : fr[i])
                if (sc.pos[nb] < 0) { sc.pos[nb] = (int)plan.inner.size(); plan.inner.push_back(nb); next.push_back(nb); }
        frontier = std::move(next);
    }
    lists.resize(plan.inner.size(), nullptr);

    const int T = team_size((long long)ns, 8);
    sc.ensure_marks(T, n);
    #pragma omp parallel num_threads(T) if(T > 1)
    {
        std::vector<uint64_t>& mark = sc.marks[omp_get_thread_num()];
        auto test_set = [&](int v) {
            uint64_t bit = 1ULL << (v & 63);
            if (mark[v >> 6] & bit) return false;
            mark[v >> 6] |= bit;
            return true;
        };
        std::vector<int> F, next;
        #pragma omp for schedule(dynamic, 1)
        for (long long i = 0; i < (long long)ns; i++) {
            std::vector<int>& ball = plan.balls[i];
            int x = plan.seeds[i];
            test_set(x);
            ball.push_back(x);
            F.assign(1, x);
            for (int d = 1; d <= K; d++) {
                next.clear();
                for (int y : F)
                    for (int nb : *lists[sc.pos[y]])
                        if (test_set(nb)) { ball.push_back(nb); next.push_back(nb); }
                F.swap(next);
            }
            for (int v : ball) mark[v >> 6] &= ~(1ULL << (v & 63));
        }
    }
    for (int v : plan.inner) sc.pos[v] = -1;
}

// Regions that share a vertex, or that one update ties
// together, are merged, so the result is vertex-disjoint.
inline void lcl_merge_overlap(LclPlan& plan, const std::vector<std::pair<int, int>>& links, LclScratch& sc) {
    const int n = (int)plan.seeds.size();
    std::vector<int> parent(n);
    for (int i = 0; i < n; i++) parent[i] = i;
    auto find = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    auto unite = [&](int a, int b) {
        a = find(a); b = find(b);
        if (a != b) { if (a < b) parent[b] = a; else parent[a] = b; }
    };
    for (int i = 0; i < n; i++)
        for (int v : plan.balls[i]) {
            int& o = sc.owner[v];
            if (o < 0) o = i; else unite(i, o);
        }
    for (auto& b : plan.balls) for (int v : b) sc.owner[v] = -1;
    for (auto& l : links) {
        auto a = plan.seed_index.find(l.first), b = plan.seed_index.find(l.second);
        if (a != plan.seed_index.end() && b != plan.seed_index.end()) unite(a->second, b->second);
    }
    plan.region_of.assign(n, -1);
    std::vector<int> rid_of_root(n, -1);
    int R = 0;
    for (int i = 0; i < n; i++) {
        int r = find(i);
        if (rid_of_root[r] < 0) rid_of_root[r] = R++;
        plan.region_of[i] = rid_of_root[r];
    }
    plan.num_regions = R;
    plan.region_offsets.assign(R + 1, 0);
    for (int i = 0; i < n; i++) plan.region_offsets[plan.region_of[i] + 1]++;
    for (int r = 0; r < R; r++) plan.region_offsets[r + 1] += plan.region_offsets[r];
    plan.region_seeds.assign(n, 0);
    std::vector<int> fill(plan.region_offsets.begin(), plan.region_offsets.end() - 1);
    for (int i = 0; i < n; i++) plan.region_seeds[fill[plan.region_of[i]]++] = i;
}

struct LclBatchTimes {
    double record_ms = 0, fetch_ms = 0, discover_ms = 0, merge_ms = 0, launch_ms = 0;
    long long pages = 0, faulted = 0, lists = 0, memo_hits = 0;
    int regions = 0;
};

template <class App>
// While the GPU repairs batch k, the CPU records batch k+1 and fetches its neighborhoods;
// a seed step that reads the solution waits until batch k is committed.
struct LclDriver {
    HybridGraph& g;
    App& app;
    bool overlap = true;
    bool guarded_record = false;
    std::vector<LclBatchTimes> times;
    std::vector<std::vector<Update>> batches;
    LclScratch scratch;

    LclDriver(HybridGraph& g_, App& app_) : g(g_), app(app_) {}

    // Record the batch in the scratchpad (CPU only).
    void record(const std::vector<Update>& B, LclBatchTimes& t) {
        auto t0 = lcl_clock::now();
        g.memo_clear();
        for (const Update& e : B) {
            if (guarded_record) {
                bool adj = g.isAdjacent(e.u, e.v);
                if (e.ins && !adj) g.insertEdge(e.u, e.v);
                else if (!e.ins && adj) g.removeEdge(e.u, e.v);
            } else {
                if (e.ins) g.insertEdge(e.u, e.v); else g.removeEdge(e.u, e.v);
            }
        }
        t.record_ms += lcl_ms_since(t0);
    }

    template <class Fn>
    void with_fetch_accounting(LclBatchTimes& t, Fn fn) {
        FetchStats f0 = g_fetch;
        fn();
        t.fetch_ms += g_fetch.ms - f0.ms;
        t.pages += g_fetch.pages - f0.pages;
        t.lists += g_fetch.lists - f0.lists;
        t.memo_hits += g_fetch.memo_hits - f0.memo_hits;
    }

    // Fetch for batch k+1 that does not depend on the solution, while batch k is on the GPU.
    void prefetch(int k, const std::vector<Update>& B) {
        std::vector<int> H;
        app.prefetch_hint(B, H);
        if (H.empty()) return;
        with_fetch_accounting(times[k], [&] { FetchResult fr; fetch_frontier(g, H, fr); });
    }

    // Seed, discover, merge, then pack and launch the asynchronous GPU repair.
    void prepare(int k, const std::vector<Update>& B) {
        LclBatchTimes& t = times[k];
        LclPlan plan;
        auto t0 = lcl_clock::now();
        app.seed(B, plan.seeds);
        for (size_t i = 0; i < plan.seeds.size(); i++) plan.seed_index.emplace(plan.seeds[i], (int)i);
        double fetch_before = t.fetch_ms;
        with_fetch_accounting(t, [&] { lcl_discover(g, App::K, plan, scratch); });
        t.discover_ms += lcl_ms_since(t0) - (t.fetch_ms - fetch_before);
        auto t1 = lcl_clock::now();
        std::vector<std::pair<int, int>> links;
        app.link(B, links);
        lcl_merge_overlap(plan, links, scratch);
        t.merge_ms += lcl_ms_since(t1);
        t.regions = plan.num_regions;
        auto t2 = lcl_clock::now();
        fetch_before = t.fetch_ms;
        with_fetch_accounting(t, [&] { app.launch(k, B, plan); });
        t.launch_ms += lcl_ms_since(t2) - (t.fetch_ms - fetch_before);
    }

    template <class Load, class OnDone>
    // load(k) parses batch k against the graph as of batch k-1; parsing is not timed.
    double run(int n, Load load, OnDone on_done) {
        batches.assign(n, {});
        times.assign(n, LclBatchTimes{});
        std::vector<char> loaded(n, 0), recorded(n, 0), launched(n, 0);
        scratch.ensure(g.num_nodes);
        scratch.ensure_marks(fetch_workers(), g.num_nodes);
        double untimed = 0, untimed_total = 0;
        auto ensure = [&](int k) {
            if (loaded[k]) return;
            auto t0 = lcl_clock::now();
            batches[k] = load(k);
            loaded[k] = 1;
            double ms = lcl_ms_since(t0);
            untimed += ms; untimed_total += ms;
        };
        auto start = lcl_clock::now();
        auto last = start;
        long long mf_last = process_major_faults();
        for (int k = 0; k < n; k++) {
            if (!launched[k]) {
                if (!recorded[k]) { ensure(k); record(batches[k], times[k]); recorded[k] = 1; }
                prepare(k, batches[k]);
                launched[k] = 1;
            }
            if (overlap && k + 1 < n) {
                ensure(k + 1);
                record(batches[k + 1], times[k + 1]);
                recorded[k + 1] = 1;
                if (!app.seed_needs_state) { prepare(k + 1, batches[k + 1]); launched[k + 1] = 1; }
                else prefetch(k + 1, batches[k + 1]);
            }
            app.collect(k);
            long long mf = process_major_faults();
            times[k].faulted = mf - mf_last;
            mf_last = mf;
            auto now = lcl_clock::now();
            double wall = std::chrono::duration<double, std::milli>(now - last).count() - untimed;
            last = now;
            untimed = 0;
            on_done(k, wall);
        }
        return std::chrono::duration<double, std::milli>(lcl_clock::now() - start).count() - untimed_total;
    }
};

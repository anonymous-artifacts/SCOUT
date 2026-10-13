// Page-granular fetch engine: every neighborhood request of the runtime goes through here.

#pragma once

#include <vector>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <cstdlib>
#include <sys/resource.h>
#include <omp.h>

#include "FlashGraph.h"

struct FetchStats {
    long long requests = 0;
    long long memo_hits = 0;
    long long lists = 0;
    long long pages = 0;
    long long list_bytes = 0;
    double ms = 0;
};
inline FetchStats g_fetch;

// N(F), aligned with the request. Valid until the next edge change or memo clear.
struct FetchResult {
    std::vector<const std::vector<int>*> lists;
    std::vector<std::vector<int>> owned;
    const std::vector<int>& operator[](size_t i) const { return *lists[i]; }
    size_t size() const { return lists.size(); }
};

// T, the number of fetch workers (SCOUT_FETCH_THREADS, default min(cores, 64)).
inline int fetch_workers() {
    static int T = [] {
        int t = std::min(omp_get_max_threads(), 64);
        if (const char* e = std::getenv("SCOUT_FETCH_THREADS")) t = std::max(1, std::atoi(e));
        return t;
    }();
    return T;
}

inline int team_size(long long work, long long grain) {
    long long t = (work + grain - 1) / grain;
    return (int)std::max(1LL, std::min<long long>(t, fetch_workers()));
}

// Buffers reused across calls, so small requests allocate nothing.
struct FetchScratch {
    std::vector<int> Fm;
    std::vector<size_t> off_pages, adj_pages, to_fault, cold;
    std::vector<unsigned char> vec;
    std::unordered_set<int> seen;
};
inline FetchScratch& fetch_scratch() { static thread_local FetchScratch s; return s; }

// Page sets this small are faulted by the caller; a worker team cannot pay off.
#define SCOUT_SERIAL_PAGES 8

// Pages read from storage so far (major faults); drivers take per-batch differences.
inline long long process_major_faults() {
    rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_majflt;
}

// Workers fault disjoint, contiguous slices of the sorted page set.
// Pages already resident (mincore) are skipped.
inline void fault_pages(const FlashGraph& base, const std::vector<size_t>& pages, FetchScratch& sc) {
    if (pages.size() <= SCOUT_SERIAL_PAGES) {
        for (size_t p : pages) base.touch_page(p);
        return;
    }
    std::vector<size_t>& cold = sc.cold;
    cold.clear();
    for (size_t i = 0; i < pages.size();) {
        size_t j = i + 1;
        while (j < pages.size() && pages[j] == pages[j - 1] + 1) j++;
        sc.vec.resize(j - i);
        if (base.resident(pages[i], j - i, sc.vec.data())) {
            for (size_t k = i; k < j; k++) if (!(sc.vec[k - i] & 1)) cold.push_back(pages[k]);
        } else {
            cold.insert(cold.end(), pages.begin() + i, pages.begin() + j);
        }
        i = j;
    }
    const long long n = (long long)cold.size();
    const int T = team_size(n, 1);
    if (T <= 1) {
        for (long long i = 0; i < n; i++) base.touch_page(cold[i]);
    } else {
        #pragma omp parallel for schedule(static) num_threads(T)
        for (long long i = 0; i < n; i++) base.touch_page(cold[i]);
    }
}

// allow_evict=false keeps lists returned by earlier calls valid.
inline void fetch_frontier(HybridGraph& g, const std::vector<int>& F, FetchResult& out,
                           bool use_memo = true, bool allow_evict = true) {
    auto t0 = std::chrono::high_resolution_clock::now();
    use_memo = use_memo && g.memo_on;
    if (use_memo && allow_evict && g.memo_over_budget()) g.memo_clear();
    const FlashGraph& base = g.base;
    const size_t P = base.page_bytes();
    FetchScratch& sc = fetch_scratch();

    std::vector<int>& Fm = sc.Fm;
    // Requested vertices that are not in the memo, without duplicates.
    Fm.clear();
    const bool small = F.size() <= 32;
    if (!small) { sc.seen.clear(); sc.seen.reserve(F.size() * 2); }
    for (int y : F) {
        if (small ? std::find(Fm.begin(), Fm.end(), y) != Fm.end() : !sc.seen.insert(y).second) continue;
        if (use_memo && g.memo_find(y)) { g_fetch.memo_hits++; continue; }
        Fm.push_back(y);
    }

    std::vector<size_t>& off_pages = sc.off_pages;
    // Offset pages first, since a list's byte range is known only after its offsets are read.
    off_pages.clear();
    const size_t w = base.offset_entry_width();
    for (int y : Fm) {
        off_pages.push_back(base.offset_entry_byte(y) / P);
        off_pages.push_back((base.offset_entry_byte(y + 1) + w - 1) / P);
    }
    std::sort(off_pages.begin(), off_pages.end());
    off_pages.erase(std::unique(off_pages.begin(), off_pages.end()), off_pages.end());
    fault_pages(base, off_pages, sc);

    std::vector<size_t>& adj_pages = sc.adj_pages;
    // Adjacency pages of every list; the union removes pages shared by several lists.
    adj_pages.clear();
    {
        const long long m = (long long)Fm.size();
        const int T = team_size(m, 1024);
        if (T <= 1) {
            for (int y : Fm) {
                size_t b = base.list_byte_begin(y), e = base.list_byte_end(y);
                if (e > b) for (size_t p = b / P; p <= (e - 1) / P; p++) adj_pages.push_back(p);
            }
        } else {
            std::vector<std::vector<size_t>> local(T);
            #pragma omp parallel num_threads(T)
            {
                std::vector<size_t>& mine = local[omp_get_thread_num()];
                #pragma omp for schedule(static)
                for (long long i = 0; i < m; i++) {
                    int y = Fm[i];
                    size_t b = base.list_byte_begin(y), e = base.list_byte_end(y);
                    if (e <= b) continue;
                    for (size_t p = b / P; p <= (e - 1) / P; p++) mine.push_back(p);
                }
            }
            for (auto& v : local) adj_pages.insert(adj_pages.end(), v.begin(), v.end());
        }
    }
    std::sort(adj_pages.begin(), adj_pages.end());
    adj_pages.erase(std::unique(adj_pages.begin(), adj_pages.end()), adj_pages.end());

    std::vector<size_t>& to_fault = sc.to_fault;
    to_fault.clear();
    std::set_difference(adj_pages.begin(), adj_pages.end(), off_pages.begin(), off_pages.end(),
                        std::back_inserter(to_fault));
    // Fault the adjacency pages not already faulted with the offsets.
    fault_pages(base, to_fault, sc);
    g_fetch.pages += (long long)(off_pages.size() + to_fault.size());

    // Merge each list with the scratchpad in parallel and memoize it.
    out.lists.assign(F.size(), nullptr);
    if (use_memo) {
        const long long m = (long long)Fm.size();
        const int T = team_size(m, 64);
        #pragma omp parallel for schedule(dynamic, 16) num_threads(T) if(T > 1)
        for (long long i = 0; i < m; i++) {
            std::vector<int> list;
            g.neighbors_direct(Fm[i], list);
            g.memo_put(Fm[i], std::move(list));
        }
        for (size_t i = 0; i < F.size(); i++) out.lists[i] = g.memo_find(F[i]);
    } else {
        out.owned.resize(F.size());
        const long long n = (long long)F.size();
        const int T = team_size(n, 64);
        #pragma omp parallel for schedule(dynamic, 16) num_threads(T) if(T > 1)
        for (long long i = 0; i < n; i++) g.neighbors_direct(F[i], out.owned[i]);
        for (size_t i = 0; i < F.size(); i++) out.lists[i] = &out.owned[i];
    }

    long long bytes = 0;
    for (auto* l : out.lists) bytes += (long long)l->size() * (long long)sizeof(int);
    g_fetch.requests += (long long)F.size();
    g_fetch.lists += (long long)Fm.size();
    g_fetch.list_bytes += bytes;
    g_fetch.ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

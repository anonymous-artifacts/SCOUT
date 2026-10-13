// Storage-resident graph: the .egr CSR is mmap'd read-only; updates live in an in-memory
// scratchpad and are merged into a vertex's neighbor list on demand.

#ifndef FLASH_GRAPH_H
#define FLASH_GRAPH_H

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <unordered_set>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef unsigned long long ull;

// Read-only view of the .egr file. Pages become resident only when touched.
class FlashGraph {
public:
    int nodes = 0;
    ull edges = 0;

    FlashGraph() = default;
    FlashGraph(const FlashGraph&) = delete;
    FlashGraph& operator=(const FlashGraph&) = delete;

    ~FlashGraph() { close_mapping(); }

    void open(const char* fname) {
        Mapping m = map_file(fname);
        adopt(m);
    }

    // Swap to a new file: map it first, then unmap the old one.
    void remap(const char* new_fname) {
        Mapping m = map_file(new_fname);
        int old_fd = fd;
        void* old_map = map_base;
        size_t old_size = file_size;

        adopt(m);

        if (old_map && old_map != MAP_FAILED) munmap(old_map, old_size);
        if (old_fd >= 0) ::close(old_fd);
    }

    inline long long nindex_at(int v) const {
        return wide_nindex ? (long long)(((const ull*)nindex_ptr)[v])
                            : (long long)(((const int32_t*)nindex_ptr)[v]);
    }

    inline int degree(int v) const {
        return (int)(nindex_at(v + 1) - nindex_at(v));
    }

    inline const int* neighbors_begin(int v) const {
        return nlist_ptr + nindex_at(v);
    }

    inline const int* neighbors_end(int v) const {
        return nlist_ptr + nindex_at(v + 1);
    }

    // Page-level view used by the fetch engine (core/fetch.h).
    inline size_t page_bytes() const { return page_size; }
    inline size_t offset_entry_byte(int v) const {
        return nindex_byte_offset + (size_t)v * (wide_nindex ? sizeof(ull) : sizeof(int32_t));
    }
    inline size_t offset_entry_width() const { return wide_nindex ? sizeof(ull) : sizeof(int32_t); }
    inline size_t list_byte_begin(int v) const { return nlist_byte_offset + (size_t)nindex_at(v) * sizeof(int); }
    inline size_t list_byte_end(int v) const { return nlist_byte_offset + (size_t)nindex_at(v + 1) * sizeof(int); }

    inline bool resident(size_t first, size_t count, unsigned char* vec) const {
        if ((first + count) * page_size > file_size)
            count = (file_size + page_size - 1) / page_size - first;
        return mincore((char*)map_base + first * page_size, count * page_size, vec) == 0;
    }
    inline void touch_page(size_t page) const {
        size_t off = page * page_size;
        if (off < file_size) (void)*(volatile const char*)((const char*)map_base + off);
    }

private:
    int fd = -1;
    void* map_base = nullptr;
    size_t file_size = 0;
    bool wide_nindex = false;
    size_t nindex_byte_offset = 0;
    size_t nlist_byte_offset = 0;
    const char* nindex_ptr = nullptr;
    const int* nlist_ptr = nullptr;
    size_t page_size = 4096;

    struct Mapping {
        int fd;
        void* map_base;
        size_t file_size;
        int nodes;
        ull edges;
        bool wide_nindex;
        size_t nindex_byte_offset;
        size_t nlist_byte_offset;
    };

    void adopt(const Mapping& m) {
        fd = m.fd;
        map_base = m.map_base;
        file_size = m.file_size;
        nodes = m.nodes;
        edges = m.edges;
        wide_nindex = m.wide_nindex;
        nindex_byte_offset = m.nindex_byte_offset;
        nlist_byte_offset = m.nlist_byte_offset;
        nindex_ptr = (const char*)map_base + nindex_byte_offset;
        nlist_ptr = (const int*)((const char*)map_base + nlist_byte_offset);
        long ps = sysconf(_SC_PAGESIZE);
        page_size = ps > 0 ? (size_t)ps : 4096;
    }

    void close_mapping() {
        if (map_base && map_base != MAP_FAILED) munmap(map_base, file_size);
        if (fd >= 0) ::close(fd);
        map_base = nullptr;
        fd = -1;
        file_size = 0;
    }

    static Mapping map_file(const char* fname) {
        FILE* f = fopen(fname, "rb");
        if (!f) { fprintf(stderr, "ERROR: FlashGraph could not open %s\n", fname); exit(1); }

        int nodes_local = 0;
        if (fread(&nodes_local, sizeof(nodes_local), 1, f) != 1) {
            fprintf(stderr, "ERROR: FlashGraph failed to read nodes from %s\n", fname); exit(1);
        }
        int edges_i32 = 0;
        if (fread(&edges_i32, sizeof(edges_i32), 1, f) != 1) {
            fprintf(stderr, "ERROR: FlashGraph failed to read edges from %s\n", fname); exit(1);
        }

        ull edges_local;
        bool wide;
        if (edges_i32 == -1) {
            ull edges64 = 0;
            if (fread(&edges64, sizeof(edges64), 1, f) != 1) {
                fprintf(stderr, "ERROR: FlashGraph failed to read 64-bit edge count from %s\n", fname); exit(1);
            }
            edges_local = edges64;
            wide = true;
        } else {
            if (nodes_local < 1 || edges_i32 < 0) {
                fprintf(stderr, "ERROR: FlashGraph found invalid header in %s\n", fname); exit(1);
            }
            edges_local = (ull)edges_i32;
            wide = false;
        }

        long header_end = ftell(f);
        fclose(f);
        if (header_end < 0) { fprintf(stderr, "ERROR: ftell failed for %s\n", fname); exit(1); }

        int fdesc = ::open(fname, O_RDONLY);
        if (fdesc < 0) { fprintf(stderr, "ERROR: open() failed for %s\n", fname); exit(1); }

        struct stat st;
        if (fstat(fdesc, &st) != 0) {
            fprintf(stderr, "ERROR: fstat failed for %s\n", fname); exit(1);
        }
        size_t sz = (size_t)st.st_size;

        void* mapped = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fdesc, 0);
        if (mapped == MAP_FAILED) {
            fprintf(stderr, "ERROR: mmap failed for %s\n", fname); exit(1);
        }

        // Accesses are scattered, so readahead would only load unneeded pages.
        if (!std::getenv("SCOUT_NO_MADV_RANDOM")) madvise(mapped, sz, MADV_RANDOM);

        Mapping m;
        m.fd = fdesc;
        m.map_base = mapped;
        m.file_size = sz;
        m.nodes = nodes_local;
        m.edges = edges_local;
        m.wide_nindex = wide;
        m.nindex_byte_offset = (size_t)header_end;
        size_t stride = wide ? sizeof(ull) : sizeof(int32_t);
        m.nlist_byte_offset = m.nindex_byte_offset + (size_t)(nodes_local + 1) * stride;
        return m;
    }
};

// Update scratchpad: inserted and deleted neighbors per vertex.
class DeltaOverlay {
public:
    std::unordered_map<int, std::vector<int>> added_adj;
    std::unordered_map<int, std::unordered_set<int>> removed_adj;
    long long delta_edge_count = 0;

    bool isAdjacentDelta(int u, int v, bool& definitive) const {
        auto ait = added_adj.find(u);
        if (ait != added_adj.end()) {
            const auto& vec = ait->second;
            if (std::find(vec.begin(), vec.end(), v) != vec.end()) {
                definitive = true;
                return true;
            }
        }
        auto rit = removed_adj.find(u);
        if (rit != removed_adj.end() && rit->second.count(v)) {
            definitive = true;
            return false;
        }
        definitive = false;
        return false;
    }

    void insertEdge(int u, int v) {
        bool untombstoned = un_tombstone(u, v);
        if (!untombstoned) {
            added_adj[u].push_back(v);
            added_adj[v].push_back(u);
        }
        delta_edge_count++;
    }

    void removeEdge(int u, int v) {
        bool forgotten = forget_delta_edge(u, v);
        if (!forgotten) {
            removed_adj[u].insert(v);
            removed_adj[v].insert(u);
        }
        delta_edge_count++;
    }

    // Current neighbors: base list minus deleted neighbors, plus inserted ones.
    void neighbors_into(int v, const FlashGraph& base, std::vector<int>& out) const {
        out.clear();
        const int* b = base.neighbors_begin(v);
        const int* e = base.neighbors_end(v);

        auto rit = removed_adj.find(v);
        if (rit == removed_adj.end() || rit->second.empty()) {
            out.insert(out.end(), b, e);
        } else {
            const auto& tomb = rit->second;
            for (const int* p = b; p != e; ++p) {
                if (!tomb.count(*p)) out.push_back(*p);
            }
        }

        auto ait = added_adj.find(v);
        if (ait != added_adj.end()) {
            out.insert(out.end(), ait->second.begin(), ait->second.end());
        }
    }

    int degree(int v, const FlashGraph& base) const {
        int d = base.degree(v);
        auto rit = removed_adj.find(v);
        if (rit != removed_adj.end()) d -= (int)rit->second.size();
        auto ait = added_adj.find(v);
        if (ait != added_adj.end()) d += (int)ait->second.size();
        return d;
    }

    void clear() {
        added_adj.clear();
        removed_adj.clear();
        delta_edge_count = 0;
    }

private:

    bool un_tombstone(int u, int v) {
        auto rit_u = removed_adj.find(u);
        if (rit_u == removed_adj.end() || !rit_u->second.count(v)) return false;
        rit_u->second.erase(v);
        auto rit_v = removed_adj.find(v);
        if (rit_v != removed_adj.end()) rit_v->second.erase(u);
        return true;
    }

    bool forget_delta_edge(int u, int v) {
        auto ait_u = added_adj.find(u);
        if (ait_u == added_adj.end()) return false;
        auto& vec_u = ait_u->second;
        auto it_u = std::find(vec_u.begin(), vec_u.end(), v);
        if (it_u == vec_u.end()) return false;
        vec_u.erase(it_u);

        auto ait_v = added_adj.find(v);
        if (ait_v != added_adj.end()) {
            auto& vec_v = ait_v->second;
            auto it_v = std::find(vec_v.begin(), vec_v.end(), u);
            if (it_v != vec_v.end()) vec_v.erase(it_v);
        }
        return true;
    }
};

// SCOUT_DUMP_TOUCHED: writes the ids of all fetched vertices at exit.
inline std::vector<char> g_touched;
inline std::string g_touched_path;
inline void dump_touched_at_exit() {
    FILE* f = fopen(g_touched_path.c_str(), "wb");
    if (!f) return;
    for (size_t v = 0; v < g_touched.size(); v++)
        if (g_touched[v]) { int x = (int)v; fwrite(&x, sizeof(int), 1, f); }
    fclose(f);
}

// Base graph + scratchpad + a batch-local memo of fetched lists (sharded, so the parallel
// merge step does not serialize on one lock).
class HybridGraph {
public:

    static constexpr int MEMO_SHARDS = 64;
    struct MemoShard {
        std::mutex mu;
        std::unordered_map<int, std::vector<int>> map;
    };
    mutable MemoShard memo_shards[MEMO_SHARDS];
    mutable std::atomic<long long> memo_bytes{0};
    long long memo_max_bytes = 0;
    bool memo_on = false;
    FlashGraph base;
    DeltaOverlay delta;
    int num_nodes = 0;
    std::string flash_path;

    void enable_memo() {
        memo_on = (std::getenv("SCOUT_NO_MEMO") == nullptr);
        if (const char* m = std::getenv("SCOUT_MEMO_MAX_BYTES")) memo_max_bytes = std::atoll(m);
    }

    void open(const char* fname) {
        base.open(fname);
        num_nodes = base.nodes;
        flash_path = fname;
        if (const char* t = std::getenv("SCOUT_DUMP_TOUCHED")) {
            g_touched_path = t;
            g_touched.assign(num_nodes, 0);
            std::atexit(dump_touched_at_exit);
        }
    }

    bool isAdjacent(int u, int v) const {
        bool definitive;
        bool result = delta.isAdjacentDelta(u, v, definitive);
        if (definitive) return result;
        const int* b = base.neighbors_begin(u);
        const int* e = base.neighbors_end(u);
        for (const int* p = b; p != e; ++p) {
            if (*p == v) return true;
        }
        return false;
    }

    MemoShard& shard_of(int v) const { return memo_shards[(unsigned)v % MEMO_SHARDS]; }

    const std::vector<int>* memo_find(int v) const {
        MemoShard& sh = shard_of(v);
        std::lock_guard<std::mutex> g(sh.mu);
        auto it = sh.map.find(v);
        return it == sh.map.end() ? nullptr : &it->second;
    }
    const std::vector<int>* memo_put(int v, std::vector<int>&& list) const {
        MemoShard& sh = shard_of(v);
        std::lock_guard<std::mutex> g(sh.mu);
        auto res = sh.map.emplace(v, std::move(list));
        if (res.second) memo_bytes += (long long)res.first->second.size() * (long long)sizeof(int);
        return &res.first->second;
    }
    void memo_clear() {
        for (auto& sh : memo_shards) { std::lock_guard<std::mutex> g(sh.mu); sh.map.clear(); }
        memo_bytes = 0;
    }
    bool memo_over_budget() const { return memo_max_bytes > 0 && memo_bytes.load() > memo_max_bytes; }
    void memo_erase(int v) {
        MemoShard& sh = shard_of(v);
        std::lock_guard<std::mutex> g(sh.mu);
        auto it = sh.map.find(v);
        if (it != sh.map.end()) {
            memo_bytes -= (long long)it->second.size() * (long long)sizeof(int);
            sh.map.erase(it);
        }
    }
    // An edge change makes both endpoints' memoized lists stale.
    void memo_invalidate(int u, int v) {
        if (!memo_on) return;
        memo_erase(u); memo_erase(v);
    }

    void insertEdge(int u, int v) { delta.insertEdge(u, v); memo_invalidate(u, v); }
    void removeEdge(int u, int v) { delta.removeEdge(u, v); memo_invalidate(u, v); }

    // Bypasses the memo: used by the fetch engine and by full sequential passes.
    void neighbors_direct(int v, std::vector<int>& out) const {
        if (!g_touched.empty()) g_touched[v] = 1;
        delta.neighbors_into(v, base, out);
    }

    void neighbors_into(int v, std::vector<int>& out) const {
        if (memo_on) {
            if (const std::vector<int>* m = memo_find(v)) { out = *m; return; }
        }
        neighbors_direct(v, out);
        if (memo_on) memo_put(v, std::vector<int>(out));
    }

    int degree(int v) const { return delta.degree(v, base); }
};

#endif

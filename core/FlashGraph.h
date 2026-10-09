// FlashGraph.h: storage-resident graph access for SCOUT.
//
// FlashGraph memory-maps a .egr CSR file read-only (MADV_RANDOM by default) so only the pages
// that are touched become resident. DeltaOverlay keeps inserted/deleted edges per vertex in
// memory, and HybridGraph combines the two: neighbors_into(v) returns the logical neighborhood
// (base adjacency minus deletions plus insertions), with an optional batch-local memo.
//
// .egr format: int32 nodes, int32 edges (or -1 followed by uint64 edges for the wide format),
// nindex[nodes+1] (int32, or int64 when wide), nlist[edges] (int32).

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
#include <unordered_set>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef unsigned long long ull;

// ---------------------------------------------------------------------------
// FlashGraph: read-only, mmap-backed access to a .egr CSR file.
// ---------------------------------------------------------------------------
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

    // Build the new mapping first, THEN swap pointers, THEN unmap the old
    // one — safe under POSIX semantics because unlink/rename doesn't
    // invalidate an fd/mmap that's still open on the old inode, so any
    // in-flight reader using the old pointers keeps working correctly
    // right up until this function returns.
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

private:
    int fd = -1;
    void* map_base = nullptr;
    size_t file_size = 0;
    bool wide_nindex = false;
    size_t nindex_byte_offset = 0;
    size_t nlist_byte_offset = 0;
    const char* nindex_ptr = nullptr;
    const int* nlist_ptr = nullptr;

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
    }

    void close_mapping() {
        if (map_base && map_base != MAP_FAILED) munmap(map_base, file_size);
        if (fd >= 0) ::close(fd);
        map_base = nullptr;
        fd = -1;
        file_size = 0;
    }

    // Mirrors readECLgraph()'s header parsing (ECLgraph.h) but stops after
    // the header — never reads nindex/nlist into RAM, only computes their
    // byte offsets so they can be mmap'd instead.
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
        // 2-hop BFS access is scattered, not sequential — suppress readahead.
        // SCOUT_NO_MADV_RANDOM (evaluation ablation) keeps the kernel's default readahead.
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

// ---------------------------------------------------------------------------
// DeltaOverlay: in-memory edges inserted/removed since the base .egr was
// written. Never touches disk — the base file is only rewritten by the
// end-of-cycle compaction step (tools/flash_compactor.h).
// ---------------------------------------------------------------------------
class DeltaOverlay {
public:
    std::unordered_map<int, std::vector<int>> added_adj;
    std::unordered_map<int, std::unordered_set<int>> removed_adj;
    long long delta_edge_count = 0;

    // Returns whether the delta has a definitive opinion on (u,v)'s
    // adjacency; if not (definitive=false), caller must fall through to
    // the base graph.
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

    // Merges base.neighbors(v) (minus tombstones) with added_adj[v] into out.
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
    // If (u,v) was a tombstoned base edge, un-tombstone it and return true.
    bool un_tombstone(int u, int v) {
        auto rit_u = removed_adj.find(u);
        if (rit_u == removed_adj.end() || !rit_u->second.count(v)) return false;
        rit_u->second.erase(v);
        auto rit_v = removed_adj.find(v);
        if (rit_v != removed_adj.end()) rit_v->second.erase(u);
        return true;
    }

    // If (u,v) was a delta-only edge (never touched base), forget it and
    // return true.
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

// ---------------------------------------------------------------------------
// HybridGraph: base FlashGraph + DeltaOverlay, exposing the same call shape
// the existing in-RAM Graph class used (isAdjacent/insertEdge/removeEdge),
// plus neighbors_into/degree for the 2-hop BFS fetch path.
// ---------------------------------------------------------------------------
// SCOUT_DUMP_TOUCHED=<path> (evaluation, experiment 3b): records every vertex
// whose adjacency list is fetched, and writes them as int32 at exit, so the
// Eq. (4) page bound can be computed for the run. Off by default.
inline std::vector<char> g_touched;
inline std::string g_touched_path;
inline void dump_touched_at_exit() {
    FILE* f = fopen(g_touched_path.c_str(), "wb");
    if (!f) return;
    for (size_t v = 0; v < g_touched.size(); v++)
        if (g_touched[v]) { int x = (int)v; fwrite(&x, sizeof(int), 1, f); }
    fclose(f);
}

class HybridGraph {
public:
    // Batch-local memo of reconciled adjacency lists (the memo of Alg. 1). Cleared at the start of
    // each batch, invalidated for both endpoints of every edge change. SCOUT_NO_MEMO disables it.
    mutable std::unordered_map<int, std::vector<int>> memo;
    mutable std::mutex memo_mu;
    bool memo_on = false;   // enabled by GC/BFS/CC via enable_memo(); MIS keeps its own hop-1 reuse
    void enable_memo() { memo_on = (std::getenv("SCOUT_NO_MEMO") == nullptr); }
    void memo_clear() { std::lock_guard<std::mutex> g(memo_mu); memo.clear(); }
    FlashGraph base;
    DeltaOverlay delta;
    int num_nodes = 0;
    std::string flash_path;

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

    void memo_invalidate(int u, int v) {
        if (!memo_on) return;
        std::lock_guard<std::mutex> g(memo_mu);
        memo.erase(u); memo.erase(v);
    }
    void insertEdge(int u, int v) { delta.insertEdge(u, v); memo_invalidate(u, v); }
    void removeEdge(int u, int v) { delta.removeEdge(u, v); memo_invalidate(u, v); }

    void neighbors_into(int v, std::vector<int>& out) const {
        if (memo_on) {
            std::lock_guard<std::mutex> g(memo_mu);
            auto it = memo.find(v);
            if (it != memo.end()) { out = it->second; return; }
        }
        if (!g_touched.empty()) g_touched[v] = 1;   // benign race: every writer stores 1
        delta.neighbors_into(v, base, out);
        if (memo_on) { std::lock_guard<std::mutex> g(memo_mu); memo.emplace(v, out); }
    }

    int degree(int v) const { return delta.degree(v, base); }
};

#endif // FLASH_GRAPH_H

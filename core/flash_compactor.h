// flash_compactor.h: merges a FlashGraph base and its DeltaOverlay into a new .egr file
// (scratchpad compaction), run outside the per-batch update path.

#ifndef FLASH_COMPACTOR_H
#define FLASH_COMPACTOR_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>
#include <fcntl.h>

#include "FlashGraph.h"

// Native C++ port of tools/efficient_converter.py's two-pass streaming
// approach (Pass 1: degree cumsum -> nindex; Pass 2: stream nlist), but
// merging base .egr + delta instead of a raw .mtx. Chosen over reusing
// the Python script because this needs to run in-process, without
// forking an interpreter, on every cycle's finalization.
inline void compact_flash_graph(const FlashGraph& base, const DeltaOverlay& delta, const std::string& out_path) {
    const int nodes = base.nodes;

    // ---- Pass 1: degree merge -> new_nindex (always written as the
    // 64-bit sentinel format, regardless of the input's original width —
    // simpler than tracking format across repeated compactions, and both
    // readECLgraph() and FlashGraph already support it). ----
    std::vector<ull> new_nindex((size_t)nodes + 1);
    new_nindex[0] = 0;
    for (int v = 0; v < nodes; v++) {
        long long d = base.degree(v);
        auto rit = delta.removed_adj.find(v);
        if (rit != delta.removed_adj.end()) d -= (long long)rit->second.size();
        auto ait = delta.added_adj.find(v);
        if (ait != delta.added_adj.end()) d += (long long)ait->second.size();
        if (d < 0) {
            fprintf(stderr, "ERROR: flash_compactor computed negative degree (%lld) for vertex %d — "
                            "delta is inconsistent with base graph\n", d, v);
            exit(1);
        }
        new_nindex[v + 1] = new_nindex[v] + (ull)d;
    }
    const ull new_edges = new_nindex[nodes];

    const std::string tmp_path = out_path + ".tmp." + std::to_string(getpid());
    FILE* out = fopen(tmp_path.c_str(), "wb");
    if (!out) { fprintf(stderr, "ERROR: flash_compactor could not open %s for writing\n", tmp_path.c_str()); exit(1); }

    const int sentinel = -1;
    if (fwrite(&nodes, sizeof(nodes), 1, out) != 1 ||
        fwrite(&sentinel, sizeof(sentinel), 1, out) != 1 ||
        fwrite(&new_edges, sizeof(new_edges), 1, out) != 1) {
        fprintf(stderr, "ERROR: flash_compactor failed to write header to %s\n", tmp_path.c_str()); exit(1);
    }
    if (fwrite(new_nindex.data(), sizeof(ull), (size_t)nodes + 1, out) != (size_t)nodes + 1) {
        fprintf(stderr, "ERROR: flash_compactor failed to write nindex to %s\n", tmp_path.c_str()); exit(1);
    }

    // ---- Pass 2: nlist merge, streamed vertex-by-vertex straight to
    // disk — never materializes the full merged neighbor list in RAM. ----
    std::vector<int> row_buf;
    for (int v = 0; v < nodes; v++) {
        row_buf.clear();
        const int* b = base.neighbors_begin(v);
        const int* e = base.neighbors_end(v);

        auto rit = delta.removed_adj.find(v);
        if (rit == delta.removed_adj.end() || rit->second.empty()) {
            row_buf.insert(row_buf.end(), b, e);
        } else {
            const auto& tomb = rit->second;
            for (const int* p = b; p != e; ++p) {
                if (!tomb.count(*p)) row_buf.push_back(*p);
            }
        }

        auto ait = delta.added_adj.find(v);
        if (ait != delta.added_adj.end()) {
            row_buf.insert(row_buf.end(), ait->second.begin(), ait->second.end());
        }

        const ull expected = new_nindex[v + 1] - new_nindex[v];
        if (expected != row_buf.size()) {
            fprintf(stderr, "ERROR: flash_compactor degree mismatch at vertex %d (expected %llu, got %zu)\n",
                    v, expected, row_buf.size());
            exit(1);
        }

        if (!row_buf.empty() && fwrite(row_buf.data(), sizeof(int), row_buf.size(), out) != row_buf.size()) {
            fprintf(stderr, "ERROR: flash_compactor failed to write nlist row for vertex %d\n", v);
            exit(1);
        }
    }

    if (fflush(out) != 0) { fprintf(stderr, "ERROR: flash_compactor fflush failed on %s\n", tmp_path.c_str()); exit(1); }
    if (fsync(fileno(out)) != 0) { fprintf(stderr, "ERROR: flash_compactor fsync failed on %s\n", tmp_path.c_str()); exit(1); }
    fclose(out);

    // ---- Atomic replace: never expose a half-written file at the
    // canonical path. rename() is atomic within the same filesystem. ----
    if (rename(tmp_path.c_str(), out_path.c_str()) != 0) {
        fprintf(stderr, "ERROR: flash_compactor rename(%s -> %s) failed\n", tmp_path.c_str(), out_path.c_str());
        exit(1);
    }

    // fsync the containing directory too, for durability against a crash
    // immediately after the rename (the rename's atomicity w.r.t.
    // visibility doesn't need this; power-loss durability does).
    std::string dir_path = ".";
    size_t slash = out_path.find_last_of('/');
    if (slash != std::string::npos) dir_path = out_path.substr(0, slash);
    int dir_fd = ::open(dir_path.c_str(), O_RDONLY);
    if (dir_fd >= 0) {
        fsync(dir_fd);
        ::close(dir_fd);
    }
}

#endif // FLASH_COMPACTOR_H

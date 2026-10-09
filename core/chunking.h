// chunking.h: bounded-degree round chunking. Decides how many candidate vertices (from the
// front of a list) fit into one GPU round without the round's output exceeding its buffer.

#pragma once
#include <vector>

// Returns how many of `candidates` (counted from the front) can be admitted
// into one round without letting their cumulative degree exceed `cap` --
// which doubles as both the count cap and the degree-sum cap, matching the
// two call sites this was extracted from exactly (both pass the same
// MAX_ROUND_FRONTIER constant for both checks). A single candidate whose
// own degree exceeds `cap` is excluded entirely (never admitted alone,
// still overflowing) -- a returned 0 on a non-empty input means "smallest
// candidate too large for this round", not "nothing to do"; callers should
// warn rather than silently treat it as an empty round.
//
// GraphT only needs a `long long degree(int) const`-shaped method --
// HybridGraph satisfies this already; templated so this header doesn't need
// to include FlashGraph.h.
template <typename GraphT>
inline int bound_by_cumulative_degree(GraphT& graph, const std::vector<int>& candidates, long long cap) {
    int n = 0;
    long long elem_probe = 0;
    for (int v : candidates) {
        if (n >= cap) break;
        long long d = (long long)graph.degree(v);
        if (elem_probe + d > cap) break;
        elem_probe += d;
        n++;
    }
    return n;
}

// local_repair.h: search-bound and deferral policy for the non-LCL problems (BFS, CC).
// A local repair that exceeds its bound is deferred and settled by one recomputation after
// the stream (SCOUT_DEFER_FALLBACK=0 settles each one immediately instead).

#pragma once
#include <cstdlib>
#include <string>

// Fixed bound (BFS's shape): CASCADE_SAFETY_BOUND is a plain constant, not
// adapted to graph structure. Exists as an overload (rather than callers
// just writing the literal) so both algorithms' bound decisions have one
// documented home, even though this side of it is trivial.
inline long long make_fallback_bound(long long fixed_bound) {
    return fixed_bound;
}

inline long long make_fallback_bound(long long fixed_bound, double low_degree_threshold,
                                      double measured_avg_degree, long long unbounded_value) {
    return (measured_avg_degree < low_degree_threshold) ? unbounded_value : fixed_bound;
}

inline bool read_defer_fallback_env(bool production_default) {
    const char* env = std::getenv("SCOUT_DEFER_FALLBACK");
    if (env == nullptr) return production_default;
    return std::string(env) != "0";
}


// Default search bound and fallback deferral for BFS and CC.

#pragma once
#include <cstdlib>
#include <string>

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

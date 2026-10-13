// The interface between the runtime and an application.
// LCL problems (core/lcl_driver.h) provide K, seed, prefetch_hint, link, launch, collect;
// non-LCL problems (core/nl_driver.h) provide seed, budget_for, resolved, select, expand,
// commit, fallback.

#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

struct Update {
    int u, v;
    bool ins;
};

// Search budget (SCOUT_BUDGET): count of explored vertices <= b (default), unbounded,
// or mem:<bytes>, the bytes of fetched lists <= Q.
struct Budget {
    enum Mode { COUNT, UNBOUNDED, MEM } mode = COUNT;
    long long b = 0;
    long long Q = 0;
    long long vertices = 0, bytes = 0;

    bool exceeded() const {
        switch (mode) {
            case COUNT: return vertices > b;
            case MEM:   return bytes > Q;
            default:    return false;
        }
    }

    static Budget unbounded() { Budget x; x.mode = UNBOUNDED; return x; }

    static Budget from_env(long long default_b) {
        Budget x;
        x.mode = COUNT;
        x.b = default_b;
        const char* m = std::getenv("SCOUT_BUDGET");
        if (!m) return x;
        if (std::strcmp(m, "unbounded") == 0) x.mode = UNBOUNDED;
        else if (std::strncmp(m, "mem:", 4) == 0) { x.mode = MEM; x.Q = std::atoll(m + 4); }
        return x;
    }

    std::string describe() const {
        switch (mode) {
            case COUNT: return "count (b=" + std::to_string(b) + ")";
            case MEM:   return "mem (Q=" + std::to_string(Q) + " bytes)";
            default:    return "unbounded";
        }
    }
};

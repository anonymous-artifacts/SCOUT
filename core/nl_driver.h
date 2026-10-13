// Maintenance for problems without a fixed repair radius: updates are applied in order,
// and each one grows a search (via fetch_frontier) until it is resolved or over budget.

#pragma once

#include <vector>

#include "FlashGraph.h"
#include "fetch.h"
#include "problem.h"

template <class App>
inline void nl_process_batch(App& app, HybridGraph& g, const std::vector<Update>& B) {
    g.memo_clear();
    FetchResult fr;
    for (const Update& e : B) {
        // Apply the update; this invalidates both endpoints' memoized lists.
        if (e.ins) g.insertEdge(e.u, e.v);
        else g.removeEdge(e.u, e.v);
        typename App::Search R;
        Budget bud = app.budget_for(e);
        // An update that changes nothing needs no search.
        if (!app.seed(e, R, bud)) continue;
        bool exceeded = bud.exceeded();
        while (!exceeded && !app.resolved(R)) {
            // Pick the next vertices to expand and fetch their lists.
            std::vector<int> Y = app.select(R, bud);
            if (bud.exceeded()) { exceeded = true; break; }
            fetch_frontier(g, Y, fr);
            for (size_t i = 0; i < fr.size(); i++) bud.bytes += (long long)fr[i].size() * (long long)sizeof(int);
            app.expand(R, Y, fr, bud);
            if (app.resolved(R)) break;
            if (bud.exceeded()) exceeded = true;
        }
        // Over budget: fall back (now or after the stream); otherwise apply the repair.
        if (exceeded) app.fallback(R);
        else app.commit(R);
    }
}

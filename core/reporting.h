// Console output: run header, per-batch line, [STAGE] and [FETCH] lines, final summary.

#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

struct ReportKV {
    std::string key;
    std::string value;
};

inline void print_run_header(const std::string& algo_name, const std::string& graph_path,
                              long long num_nodes, long long num_edges, double avg_degree,
                              const std::string& gpu_name, bool integrated) {
    std::cout << "=== SCOUT " << algo_name << " ===" << std::endl;
    std::cout << "Graph: " << graph_path << std::endl;
    std::cout << "  nodes=" << num_nodes << " edges=" << num_edges
              << " avg_degree=" << std::fixed << std::setprecision(2) << avg_degree << std::endl;
    std::cout << "GPU: " << gpu_name << " | "
              << (integrated ? "integrated (zero-copy path)" : "discrete (explicit-transfer path)")
              << std::endl;
}

inline void print_batch_result(int batch_idx, int num_batches, int edges_in_batch, double batch_ms,
                                const std::vector<ReportKV>& extra = {}) {
    std::cout << "Batch " << batch_idx << "/" << num_batches
              << " | edges=" << edges_in_batch
              << " | time=" << std::fixed << std::setprecision(3) << batch_ms << "ms";
    for (const auto& kv : extra) std::cout << " | " << kv.key << "=" << kv.value;
    std::cout << std::endl;
}

inline void print_stage_times(int batch_idx, double fetch_ms, double seed_pack_ms, double h2d_ms,
                              double repair_ms, double commit_ms) {
    std::cout << "[STAGE] batch=" << batch_idx << std::fixed << std::setprecision(3)
              << " fetch_ms=" << fetch_ms << " seed_pack_ms=" << seed_pack_ms << " h2d_ms=" << h2d_ms
              << " repair_ms=" << repair_ms << " commit_ms=" << commit_ms << std::endl;
}

inline void print_final_summary(double setup_ms, double processing_ms, double e2e_ms,
                                 const std::string& initial_label, long long initial_value,
                                 const std::string& final_label, long long final_value) {
    std::cout << "\n=== FINAL SUMMARY ===" << std::endl;
    std::cout << "Setup time:      " << std::fixed << std::setprecision(3) << setup_ms << " ms" << std::endl;
    std::cout << "Processing time: " << std::fixed << std::setprecision(3) << processing_ms << " ms" << std::endl;
    std::cout << "End-to-end time: " << std::fixed << std::setprecision(3) << e2e_ms << " ms" << std::endl;
    std::cout << "Initial " << initial_label << ": " << initial_value << std::endl;
    std::cout << "Final " << final_label << ":   " << final_value << std::endl;
}

inline void print_fetch_stats(int batch_idx, long long pages, long long faulted, long long lists,
                              long long memo_hits) {
    std::cout << "[FETCH] batch=" << batch_idx << " pages=" << pages << " faulted=" << faulted
              << " lists=" << lists << " memo_hits=" << memo_hits << std::endl;
}

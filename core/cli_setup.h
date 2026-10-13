// Command line, device setup, and batch / initial-state file reading.

#pragma once

#include "gpu_common.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

class HybridGraph;

struct ScoutCliArgs {
    std::string egr_path;
    std::string context_path;
    std::string batch_dir;
    int num_batches;
    int gpu_id;
};

inline ScoutCliArgs parse_scout_cli(int argc, char** argv, const char* usage_algo_name) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <graph.egr> <initial_" << usage_algo_name
                  << "_context.txt> <batch_folder> <num_batches> <gpu_device_id>" << std::endl;
        exit(1);
    }
    ScoutCliArgs a;
    a.egr_path = argv[1];
    a.context_path = argv[2];
    a.batch_dir = argv[3];
    a.num_batches = std::stoi(argv[4]);
    a.gpu_id = std::stoi(argv[5]);
    return a;
}

inline bool gpu_warmup_and_device_setup(int gpu_id) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    CUDA_CHECK(cudaFree(0));
    return detect_integrated_gpu(gpu_id);
}

inline std::vector<std::filesystem::path> list_batch_files_sorted(const std::string& batch_dir) {
    std::vector<std::filesystem::path> batch_paths;
    for (const auto& entry : std::filesystem::directory_iterator(batch_dir)) {
        if (entry.is_regular_file()) batch_paths.push_back(entry.path());
    }
    std::sort(batch_paths.begin(), batch_paths.end(),
               [](const std::filesystem::path& a, const std::filesystem::path& b) {
                   return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
               });
    return batch_paths;
}

struct EdgeUpdate {
    int u, v;
    bool is_insertion;
};

template <typename GraphT>
// A line "u v" (1-indexed) is an insertion if the edge is absent, else a deletion.
inline std::vector<EdgeUpdate> parse_batch_file(const std::filesystem::path& path, GraphT& graph, int num_nodes) {
    std::vector<EdgeUpdate> batch;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int src, dst;
        if (iss >> src >> dst) {
            src--; dst--;
            if (src >= 0 && src < num_nodes && dst >= 0 && dst < num_nodes && src != dst) {
                batch.push_back({src, dst, !graph.isAdjacent(src, dst)});
            }
        }
    }
    return batch;
}

#include <cstdio>
#include <cstdlib>
inline std::vector<long long> read_ints_fast(const std::string& path) {
    std::vector<long long> out;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return out;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf((size_t)sz, '\0');
    size_t got = fread(&buf[0], 1, (size_t)sz, f);
    fclose(f);
    buf.resize(got);
    out.reserve(got / 4);
    const char* p = buf.c_str();
    char* end;
    while (*p) {
        long long v = strtoll(p, &end, 10);
        if (end == p) { ++p; continue; }
        out.push_back(v);
        p = end;
    }
    return out;
}

// Initial proper coloring for graphs too large for eclgc (more than 2^31 edges):
// sequential greedy (smallest color not used by an already-colored neighbor),
// reading the .egr through mmap. Writes <egr>.colors.txt, one color per line.
//
//   g++ -O3 -o greedy_color greedy_color.cpp && ./greedy_color graph.egr
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s graph.egr\n", argv[0]); return 2; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    const char* base = (const char*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }
    madvise((void*)base, st.st_size, MADV_SEQUENTIAL);

    int32_t nodes = *(const int32_t*)base;
    int32_t sentinel = *(const int32_t*)(base + 4);
    bool wide = sentinel == -1;
    size_t off = 8;
    uint64_t edges = wide ? *(const uint64_t*)(base + 8) : (uint64_t)sentinel;
    if (wide) off = 16;
    const int64_t* nindex64 = wide ? (const int64_t*)(base + off) : nullptr;
    const int32_t* nindex32 = wide ? nullptr : (const int32_t*)(base + off);
    const int32_t* nlist = (const int32_t*)(base + off + (size_t)(nodes + 1) * (wide ? 8 : 4));
    auto idx = [&](int64_t v) -> int64_t { return wide ? nindex64[v] : nindex32[v]; };
    fprintf(stderr, "greedy_color: %d nodes, %llu edges\n", nodes, (unsigned long long)edges);

    std::vector<int> color(nodes, -1);
    std::vector<int> mark;            // mark[c] == v  <=>  color c is used by a neighbor of v
    int max_c = -1;
    for (int v = 0; v < nodes; v++) {
        for (int64_t e = idx(v); e < idx(v + 1); e++) {
            int c = color[nlist[e]];
            if (c >= 0) {
                if (c >= (int)mark.size()) mark.resize(c + 1, -1);
                mark[c] = v;
            }
        }
        int c = 0;
        while (c < (int)mark.size() && mark[c] == v) c++;
        color[v] = c;
        if (c > max_c) max_c = c;
    }

    std::string out = std::string(argv[1]) + ".colors.txt";
    std::string tmp = out + ".tmp";      // renamed at the end so no reader ever sees a partial file
    FILE* f = fopen(tmp.c_str(), "w");
    for (int v = 0; v < nodes; v++) fprintf(f, "%d\n", color[v]);
    fclose(f);
    rename(tmp.c_str(), out.c_str());
    fprintf(stderr, "greedy_color: %d colors -> %s\n", max_c + 1, out.c_str());
    return 0;
}

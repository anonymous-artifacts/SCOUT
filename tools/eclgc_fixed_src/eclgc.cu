/*
ECL-GC code: ECL-GC is a graph-coloring algorithm with shortcutting. The CUDA
implementation thereof is quite fast. It operates on graphs stored in binary
CSR format.

Copyright (c) 2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Ghadeer Alabandi, Evan Powers, and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-GC/.

Publication: This work is described in detail in the following paper.
Ghadeer Alabandi, Evan Powers, and Martin Burtscher. Increasing the Parallelism
of Graph Coloring via Shortcutting. Proceedings of the 2020 ACM SIGPLAN
Symposium on Principles and Practice of Parallel Programming, pp. 262-275.
February 2020.
*/


#include <algorithm>
#include <cuda.h>
#include <chrono>
#include <string>
#include <cstdio>
#include "ECLgraph.h"

using namespace std::chrono;

static const int Device = 0;

static const int ThreadsPerBlock = 512;
static const unsigned int Warp = 0xffffffff;
static const int WS = 32;  // warp size and bits per int
static const int MSB = 1 << (WS - 1);
static const int Mask = (1 << (WS / 2)) - 1;

static __device__ int wlsize = 0;

// GPU memory helpers: track program-specific (delta) memory usage
static double g_baseline_used_mb = -1.0; // set before any cudaMalloc
static double g_peak_delta_mb    = 0.0;  // peak program-specific usage
static double g_gpu_total_mb     = 0.0;

static void gpu_memory_set_baseline()
{
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    g_gpu_total_mb     = (double)total_bytes / (1024.0 * 1024.0);
    g_baseline_used_mb = g_gpu_total_mb - (double)free_bytes / (1024.0 * 1024.0);
  }
}

static void print_gpu_memory_usage(const char* label)
{
  size_t free_bytes = 0, total_bytes = 0;
  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess) {
    printf("%s GPU memory: cudaMemGetInfo failed: %s\n",
           label, cudaGetErrorString(err));
    return;
  }
  const double total_mb = (double)total_bytes / (1024.0 * 1024.0);
  const double free_mb  = (double)free_bytes  / (1024.0 * 1024.0);
  const double used_mb  = total_mb - free_mb;
  double delta_mb = used_mb - g_baseline_used_mb;
  if (delta_mb < 0.0) delta_mb = 0.0;
  printf("%s program memory: %.2f MB (delta from baseline %.2f MB)\n",
         label, delta_mb, g_baseline_used_mb);
  // Track peak delta
  if (delta_mb > g_peak_delta_mb) g_peak_delta_mb = delta_mb;
  g_gpu_total_mb = total_mb;
}


// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static __device__ unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

typedef unsigned long long ull;

// __shfl_sync is 32-bit; pack/unpack for 64-bit values
static __device__ __forceinline__ ull shfl_ull(const ull v, const int src_lane)
{
  unsigned int lo = (unsigned int)(v & 0xFFFFFFFFull);
  unsigned int hi = (unsigned int)(v >> 32);
  lo = __shfl_sync(Warp, lo, src_lane);
  hi = __shfl_sync(Warp, hi, src_lane);
  return ((ull)hi << 32) | (ull)lo;
}


static __global__
void init(const int nodes, const ull edges, const ull* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nlist2, int* const __restrict__ posscol, int* const __restrict__ posscol2, int* const __restrict__ color, int* const __restrict__ wl)
{
  const int lane = threadIdx.x % WS;
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;

  int maxrange = -1;
  for (int v = thread; __any_sync(Warp, v < nodes); v += threads) {
    bool cond = false;
    ull beg = 0, end = 0, pos = 0;
    int degv = 0, active = 0;
    if (v < nodes) {
      beg = nidx[v];
      end = nidx[v + 1];
      const ull du = end - beg;
      if (du > (ull)INT_MAX) {printf("degree too large\n"); asm("trap;");}
      degv = (int)du;
      cond = (degv >= WS);
      if (cond) {
        wl[atomicAdd(&wlsize, 1)] = v;
      } else {
        active = 0;
        pos = beg;
        for (ull i = beg; i < end; i++) {
          const int nei = nlist[(size_t)i];
          if (nei == v) continue;  // skip self-loops
          const ull dnu = nidx[nei + 1] - nidx[nei];
          if (dnu > (ull)INT_MAX) {printf("degree too large\n"); asm("trap;");}
          const int degn = (int)dnu;
          if ((degv < degn) || ((degv == degn) && (hash(v) < hash(nei))) || ((degv == degn) && (hash(v) == hash(nei)) && (v < nei))) {
            active |= (unsigned int)MSB >> (int)(i - beg);
            pos++;
          }
        }
      }
    }

    int bal = __ballot_sync(Warp, cond);
    while (bal != 0) {
      const int who = __ffs(bal) - 1;
      bal &= bal - 1;
      const int wv = __shfl_sync(Warp, v, who);
      const ull wbeg = shfl_ull(beg, who);
      const ull wend = shfl_ull(end, who);
      const int wdegv = (int)(wend - wbeg);
      ull wpos = wbeg;
      for (ull i = wbeg + (ull)lane; __any_sync(Warp, i < wend); i += (ull)WS) {
        int wnei;
        bool prio = false;
        if (i < wend) {
          wnei = nlist[(size_t)i];
          if (wnei != wv) {  // skip self-loops
            const ull wdnu = nidx[wnei + 1] - nidx[wnei];
            if (wdnu > (ull)INT_MAX) {printf("degree too large\n"); asm("trap;");}
            const int wdegn = (int)wdnu;
            prio = ((wdegv < wdegn) || ((wdegv == wdegn) && (hash(wv) < hash(wnei))) || ((wdegv == wdegn) && (hash(wv) == hash(wnei)) && (wv < wnei)));
          }
        }
        const int b = __ballot_sync(Warp, prio);
        const int offs = __popc(b & ((1 << lane) - 1));
        if (prio) nlist2[(size_t)(wpos + (ull)offs)] = wnei;
        wpos += __popc(b);
      }
      if (who == lane) pos = wpos;
    }

    if (v < nodes) {
      const int range = (int)(pos - beg);
      maxrange = max(maxrange, range);
      color[v] = (cond || (range == 0)) ? (range << (WS / 2)) : active;
      posscol[v] = (range >= WS) ? -1 : (MSB >> range);
    }
  }
  if (maxrange >= Mask) {printf("too many active neighbors\n"); asm("trap;");}

  for (ull i = (ull)thread; i < edges / (ull)WS + 1ull; i += (ull)threads) posscol2[(size_t)i] = -1;
}


static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void runLarge(const int nodes, const ull* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ posscol, int* const __restrict__ posscol2, volatile int* const __restrict__ color, const int* const __restrict__ wl)
{
  const int stop = wlsize;
  if (stop != 0) {
    const int lane = threadIdx.x % WS;
    const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
    const int threads = gridDim.x * ThreadsPerBlock;
    bool again;
    do {
      again = false;
      for (int w = thread; __any_sync(Warp, w < stop); w += threads) {
        bool shortcut, done, cond = false;
        int v, data, range, pcol;
        ull beg = 0;
        if (w < stop) {
          v = wl[w];
          data = color[v];
          range = data >> (WS / 2);
          if (range > 0) {
            beg = nidx[v];
            pcol = posscol[v];
            cond = true;
          }
        }

        int bal = __ballot_sync(Warp, cond);
        while (bal != 0) {
          const int who = __ffs(bal) - 1;
          bal &= bal - 1;
          const int wv = __shfl_sync(Warp, v, who);
          const int wdata = __shfl_sync(Warp, data, who);
          const int wrange = wdata >> (WS / 2);
          const ull wbeg = shfl_ull(beg, who);
          const int wmincol = wdata & Mask;
          const int wmaxcol = wmincol + wrange;
          const ull wend = wbeg + (ull)wmaxcol;
          const ull woffs = wbeg / (ull)WS;
          int wpcol = __shfl_sync(Warp, pcol, who);

          bool wshortcut = true;
          bool wdone = true;
          for (ull i = wbeg + (ull)lane; __any_sync(Warp, i < wend); i += (ull)WS) {
            int nei, neidata, neirange;
            if (i < wend) {
              nei = nlist[(size_t)i];
              if (nei != wv) {  // skip self-loops
                neidata = color[nei];
                neirange = neidata >> (WS / 2);
                const bool neidone = (neirange == 0);
                wdone &= neidone; //consolidated below
                if (neidone) {
                  const int neicol = neidata;
                  if (neicol < WS) {
                    wpcol &= ~((unsigned int)MSB >> neicol); //consolidated below
                  } else {
                    const ull pi = woffs + (ull)(neicol / WS);
                    if ((wmincol <= neicol) && (neicol < wmaxcol) && ((posscol2[(size_t)pi] << (neicol % WS)) < 0)) {
                      atomicAnd((int*)&posscol2[(size_t)pi], ~((unsigned int)MSB >> (neicol % WS)));
                    }
                  }
                } else {
                  const int neimincol = neidata & Mask;
                  const int neimaxcol = neimincol + neirange;
                  if ((neimincol <= wmincol) && (neimaxcol >= wmincol)) wshortcut = false; //consolidated below
                }
              }
            }
          }
          wshortcut = __all_sync(Warp, wshortcut);
          wdone = __all_sync(Warp, wdone);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 1);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 2);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 4);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 8);
          wpcol &= __shfl_xor_sync(Warp, wpcol, 16);
          if (who == lane) pcol = wpcol;
          if (who == lane) done = wdone;
          if (who == lane) shortcut = wshortcut;
        }

        if (w < stop) {
          if (range > 0) {
            const int mincol = data & Mask;
            int val = pcol, mc = 0;
            if (pcol == 0) {
              const ull offs = beg / (ull)WS;
              mc = max(1, mincol / WS);
              while ((val = posscol2[(size_t)(offs + (ull)mc)]) == 0) mc++;
            }
            int newmincol = mc * WS + __clz(val);
            if (mincol != newmincol) shortcut = false;
            if (shortcut || done) {
              pcol = (newmincol < WS) ? ((unsigned int)MSB >> newmincol) : 0;
            } else {
              const int maxcol = mincol + range;
              const int range = maxcol - newmincol;
              newmincol = (range << (WS / 2)) | newmincol;
              again = true;
            }
            posscol[v] = pcol;
            color[v] = newmincol;
          }
        }
      }
    } while (__any_sync(Warp, again));
  }
}


static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void runSmall(const int nodes, const ull* const __restrict__ nidx, const int* const __restrict__ nlist, volatile int* const __restrict__ posscol, int* const __restrict__ color)
{
  const int thread = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int threads = gridDim.x * ThreadsPerBlock;

  if (thread == 0) wlsize = 0;

  bool again;
  do {
    again = false;
    for (int v = thread; v < nodes; v += threads) {
      __syncthreads();  // optional
      int pcol = posscol[v];
      if (__popc(pcol) > 1) {
        const ull beg = nidx[v];
        int active = color[v];
        int allnei = 0;
        int keep = active;
        do {
          const int old = active;
          active &= active - 1;
          const int curr = old ^ active;
          const ull i = beg + (ull)__clz(curr);
          const int nei = nlist[(size_t)i];
          if (nei == v) {  // skip self-loops
            keep ^= curr;
            continue;
          }
          const int neipcol = posscol[nei];
          allnei |= neipcol;
          if ((pcol & neipcol) == 0) {
            pcol &= pcol - 1;
            keep ^= curr;
          } else if (__popc(neipcol) == 1) {
            pcol ^= neipcol;
            keep ^= curr;
          }
        } while (active != 0);
        if (keep != 0) {
          const int best = (unsigned int)MSB >> __clz(pcol);
          if ((best & ~allnei) != 0) {
            pcol = best;
            keep = 0;
          }
        }
        again |= keep;
        if (keep == 0) keep = __clz(pcol);
        color[v] = keep;
        posscol[v] = pcol;
      }
    }
  } while (again);
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg);  cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg);  cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  float stop() {cudaEventRecord(end, 0);  cudaEventSynchronize(end);  float ms;  cudaEventElapsedTime(&ms, beg, end);  return 0.001f * ms;}
};


int main(int argc, char* argv[])
{
  auto tt_start = std::chrono::high_resolution_clock::now();
  printf("ECL-GC v1.2 (%s)\n", __FILE__);
  printf("Copyright 2020 Texas State University\n\n");

  if (argc != 2) {printf("USAGE: %s input_file_name\n\n", argv[0]);  exit(-1);}
  if (WS != 32) {printf("ERROR: warp size must be 32\n\n");  exit(-1);}
  if (WS != sizeof(int) * 8) {printf("ERROR: bits per word must match warp size\n\n");  exit(-1);}
  if ((ThreadsPerBlock < WS) || ((ThreadsPerBlock % WS) != 0)) {printf("ERROR: threads per block must be a multiple of the warp size\n\n");  exit(-1);}
  if ((ThreadsPerBlock & (ThreadsPerBlock - 1)) != 0) {printf("ERROR: threads per block must be a power of two\n\n");  exit(-1);}

  ECLgraph g = readECLgraph(argv[1]);
  printf("configuration: %d nodes and %llu edges (%s)\n", g.nodes, (unsigned long long)g.edges, argv[1]);
  printf("average degree: %.2f edges per node\n", 1.0 * (double)g.edges / g.nodes);

  int* const color = new int [g.nodes];

  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  printf("gpu: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);

  // Set GPU memory baseline before allocations
  gpu_memory_set_baseline();

  ull *nidx_d;
  int *nlist_d, *nlist2_d, *posscol_d, *posscol2_d, *color_d, *wl_d;
  if (cudaSuccess != cudaMalloc((void **)&nidx_d, (g.nodes + 1) * sizeof(ull))) {printf("ERROR: could not allocate nidx_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&nlist_d, (size_t)g.edges * sizeof(int))) {printf("ERROR: could not allocate nlist_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&nlist2_d, (size_t)g.edges * sizeof(int))) {printf("ERROR: could not allocate nlist2_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&posscol_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate posscol_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&posscol2_d, (size_t)(g.edges / WS + 1) * sizeof(int))) {printf("ERROR: could not allocate posscol2_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&color_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate color_d\n\n");  exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&wl_d, g.nodes * sizeof(int))) {printf("ERROR: could not allocate wl_d\n\n");  exit(-1);}

  if (cudaSuccess != cudaMemcpy(nidx_d, g.nindex, (g.nodes + 1) * sizeof(ull), cudaMemcpyHostToDevice)) {printf("ERROR: copying nidx to device failed\n\n");  exit(-1);}
  if (cudaSuccess != cudaMemcpy(nlist_d, g.nlist, (size_t)g.edges * sizeof(int), cudaMemcpyHostToDevice)) {printf("ERROR: copying nlist to device failed\n\n");  exit(-1);}

  const int blocks = SMs * mTpSM / ThreadsPerBlock;
  cudaFuncSetCacheConfig(init, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(runLarge, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(runSmall, cudaFuncCachePreferL1);

  GPUTimer timer_total, timer_phase1, timer_phase2, timer_phase3;
  double time_phase1 = 0.0, time_phase2 = 0.0, time_phase3 = 0.0;

  timer_total.start();

  // Phase 1: Initialize (compute priorities, filter neighbors, setup worklist)
  timer_phase1.start();
  init<<<blocks, ThreadsPerBlock>>>(g.nodes, g.edges, nidx_d, nlist_d, nlist2_d, posscol_d, posscol2_d, color_d, wl_d);
  cudaDeviceSynchronize();
  time_phase1 = timer_phase1.stop();

  // Phase 2: Process large-degree vertices (shortcutting)
  timer_phase2.start();
  runLarge<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist2_d, posscol_d, posscol2_d, color_d, wl_d);
  cudaDeviceSynchronize();
  time_phase2 = timer_phase2.stop();

  // Phase 3: Process small-degree vertices (bitmap coloring)
  timer_phase3.start();
  runSmall<<<blocks, ThreadsPerBlock>>>(g.nodes, nidx_d, nlist_d, posscol_d, color_d);
  cudaDeviceSynchronize();
  time_phase3 = timer_phase3.stop();

  const float runtime = timer_total.stop();

  // Sample memory after coloring compute (captures peak during execution)
  print_gpu_memory_usage("ECL-GC (after coloring compute)");

  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.6f Mnodes/s\n", g.nodes * 0.000001 / runtime);
  printf("throughput: %.6f Medges/s\n", (double)g.edges * 0.000001 / runtime);
  printf("\n=== Profiling Breakdown ===\n");

  // Normalize phases to sum to 100%
  double total_phase_time = time_phase1 + time_phase2 + time_phase3;
  if (total_phase_time > 0.0) {
    printf("Phase 1 (Initialize - compute priorities, filter neighbors, setup worklist): %.6f s (%.1f%%)\n", time_phase1, (time_phase1/total_phase_time)*100);
    printf("Phase 2 (Process large-degree vertices - shortcutting): %.6f s (%.1f%%)\n", time_phase2, (time_phase2/total_phase_time)*100);
    printf("Phase 3 (Process small-degree vertices - bitmap coloring): %.6f s (%.1f%%)\n", time_phase3, (time_phase3/total_phase_time)*100);
  } else {
    printf("Phase 1 (Initialize - compute priorities, filter neighbors, setup worklist): %.6f s (N/A)\n", time_phase1);
    printf("Phase 2 (Process large-degree vertices - shortcutting): %.6f s (N/A)\n", time_phase2);
    printf("Phase 3 (Process small-degree vertices - bitmap coloring): %.6f s (N/A)\n", time_phase3);
  }

  if (cudaSuccess != cudaMemcpy(color, color_d, g.nodes * sizeof(int), cudaMemcpyDeviceToHost)) {printf("ERROR: copying color from device failed\n\n");  exit(-1);}
  cudaFree(wl_d);  cudaFree(color_d);  cudaFree(posscol2_d);  cudaFree(posscol_d);  cudaFree(nlist2_d);  cudaFree(nlist_d);  cudaFree(nidx_d);

    // Stop total timer BEFORE validation

  // for (int v = 0; v < g.nodes; v++) {
  //   if (color[v] < 0) {printf("ERROR: found unprocessed node in graph (node %d with deg %llu)\n\n", v, (unsigned long long)(g.nindex[v + 1] - g.nindex[v]));  exit(-1);}
  //   for (ull i = g.nindex[v]; i < g.nindex[v + 1]; i++) {
  //     if (g.nlist[(size_t)i] == v) continue;  // skip self-loops
  //     if (color[g.nlist[(size_t)i]] == color[v]) {printf("ERROR: found adjacent nodes with same color %d (%d %d)\n\n", color[v], v, g.nlist[(size_t)i]);  exit(-1);}
  //   }
  // }
  // printf("result verification passed\n");

  const int vals = 16;
  int c[vals];
  for (int i = 0; i < vals; i++) c[i] = 0;
  int cols = -1;
  for (int v = 0; v < g.nodes; v++) {
    cols = std::max(cols, color[v]);
    if (color[v] < vals) c[color[v]]++;
  }
  cols++;
  printf("colors used: %d\n", cols);

  int sum = 0;
  for (int i = 0; i < std::min(vals, cols); i++) {
    sum += c[i];
    printf("col %2d: %10d (%5.1f%%)\n", i, c[i], 100.0 * sum / g.nodes);
  }
  
  auto tt_end = std::chrono::high_resolution_clock::now();

  double tsecs = std::chrono::duration_cast<std::chrono::duration<double>>(tt_end - tt_start).count();
  printf("Total GC time: %.6f s\n", tsecs);

  // Write the color array to disk: one color per line, 0-indexed node
  // order -- matches this project's own coloring seed-file convention
  // (algorithms/coloring/scout_coloring.cu / hybrid_coloring.cu both read this
  // format as-is, no reformatting). Added because this profiling variant
  // otherwise discards the computed coloring after printing stats.
  {
    std::string out_path = std::string(argv[1]) + ".colors.txt";
    FILE* outf = fopen(out_path.c_str(), "w");
    if (outf) {
      for (int v = 0; v < g.nodes; v++) fprintf(outf, "%d\n", color[v]);
      fclose(outf);
      printf("colors written to %s\n", out_path.c_str());
    } else {
      fprintf(stderr, "WARNING: could not open %s for writing\n", out_path.c_str());
    }
  }

  delete [] color;
  freeECLgraph(g);

  // Print peak program-specific GPU memory consumed during entire run
  double peak_pct = (g_gpu_total_mb > 0.0) ? (g_peak_delta_mb / g_gpu_total_mb * 100.0) : 0.0;
  printf("Peak program GPU memory: %.2f MB (%.1f%% of %.2f MB total)\n",
         g_peak_delta_mb, peak_pct, g_gpu_total_mb);
  return 0;
}

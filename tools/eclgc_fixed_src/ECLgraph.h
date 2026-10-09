/*
Copyright (c) 2016-2020, Texas State University. All rights reserved.

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

Author: Martin Burtscher
*/


#ifndef ECL_GRAPH
#define ECL_GRAPH

#include <cstdlib>
#include <cstdio>
#include <climits>

typedef unsigned long long ull;

struct ECLgraph {
  int nodes;
  ull edges;
  ull* nindex;
  int* nlist;
  int* eweight;
};

ECLgraph readECLgraph(const char* const fname)
{
  ECLgraph g;
  int cnt;

  FILE* f = fopen(fname, "rb");  if (f == NULL) {fprintf(stderr, "ERROR: could not open file %s\n\n", fname);  exit(-1);}
  cnt = fread(&g.nodes, sizeof(g.nodes), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to read nodes\n\n");  exit(-1);}
  int edges_i32 = 0;
  cnt = fread(&edges_i32, sizeof(edges_i32), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to read edges\n\n");  exit(-1);}

  // Support both formats:
  // 32-bit: [int32 nodes][int32 edges][int32 nindex[n+1]][int32 nlist[edges]][optional int32 eweight[edges]]
  // 64-bit: [int32 nodes][int32 -1][uint64 edges][uint64 nindex[n+1]][int32 nlist[edges]][optional int32 eweight[edges]]
  if (edges_i32 == -1) {
    ull edges_u64 = 0;
    cnt = fread(&edges_u64, sizeof(edges_u64), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to read edges (64-bit)\n\n");  exit(-1);}
    g.edges = edges_u64;
  } else {
    if ((g.nodes < 1) || (edges_i32 < 0)) {fprintf(stderr, "ERROR: node or edge count too low\n\n");  exit(-1);}
    g.edges = (ull)edges_i32;
  }

  const ull edges = g.edges;
  if (edges > (ull)(SIZE_MAX / sizeof(int))) {fprintf(stderr, "ERROR: edge list too large for host allocation\n\n");  exit(-1);}
  if ((ull)(g.nodes + 1) > (ull)(SIZE_MAX / sizeof(ull))) {fprintf(stderr, "ERROR: nindex too large for host allocation\n\n");  exit(-1);}

  g.nindex = (ull*)malloc((size_t)(g.nodes + 1) * sizeof(g.nindex[0]));
  g.nlist = (int*)malloc((size_t)edges * sizeof(g.nlist[0]));
  g.eweight = (int*)malloc((size_t)edges * sizeof(g.eweight[0]));
  if ((g.nindex == NULL) || (g.nlist == NULL) || (g.eweight == NULL)) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}

  if (edges_i32 == -1) {
    cnt = fread(g.nindex, sizeof(g.nindex[0]), g.nodes + 1, f);  if (cnt != g.nodes + 1) {fprintf(stderr, "ERROR: failed to read neighbor index list (64-bit)\n\n");  exit(-1);}
  } else {
    // read int32 offsets and widen
    int* nindex32 = (int*)malloc((size_t)(g.nodes + 1) * sizeof(int));
    if (nindex32 == NULL) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}
    cnt = fread(nindex32, sizeof(nindex32[0]), g.nodes + 1, f);  if (cnt != g.nodes + 1) {fprintf(stderr, "ERROR: failed to read neighbor index list\n\n");  exit(-1);}
    for (int i = 0; i <= g.nodes; i++) g.nindex[i] = (ull)nindex32[i];
    free(nindex32);
  }

  if (edges > 0) {
    const size_t got = fread(g.nlist, sizeof(g.nlist[0]), (size_t)edges, f);
    if (got != (size_t)edges) {fprintf(stderr, "ERROR: failed to read neighbor list\n\n");  exit(-1);}
  }

  size_t gotw = fread(g.eweight, sizeof(g.eweight[0]), (size_t)edges, f);
  if (gotw == 0) {
    free(g.eweight);
    g.eweight = NULL;
  } else {
    if (gotw != (size_t)edges) {fprintf(stderr, "ERROR: failed to read edge weights\n\n");  exit(-1);}
  }
  fclose(f);

  return g;
}

void writeECLgraph(const ECLgraph g, const char* const fname)
{
  if ((g.nodes < 1)) {fprintf(stderr, "ERROR: node count too low\n\n");  exit(-1);}
  int cnt;
  FILE* f = fopen(fname, "wb");  if (f == NULL) {fprintf(stderr, "ERROR: could not open file %s\n\n", fname);  exit(-1);}
  cnt = fwrite(&g.nodes, sizeof(g.nodes), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to write nodes\n\n");  exit(-1);}

  // Preserve strict legacy 32-bit format when possible, else write 64-bit sentinel format.
  // Legacy 32-bit: nodes(int32), edges(int32), nindex(int32[n+1]), nlist(int32[m]), eweight(int32[m]) optional.
  if (g.edges <= (ull)INT_MAX) {
    const int e32 = (int)g.edges;
    cnt = fwrite(&e32, sizeof(e32), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to write edges\n\n");  exit(-1);}
    int* nindex32 = (int*)malloc((size_t)(g.nodes + 1) * sizeof(int));
    if (nindex32 == NULL) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}
    for (int i = 0; i <= g.nodes; i++) {
      if (g.nindex[i] > (ull)INT_MAX) {fprintf(stderr, "ERROR: nindex too large for legacy format\n\n");  exit(-1);}
      nindex32[i] = (int)g.nindex[i];
    }
    cnt = (int)fwrite(nindex32, sizeof(nindex32[0]), (size_t)(g.nodes + 1), f);
    if (cnt != g.nodes + 1) {fprintf(stderr, "ERROR: failed to write neighbor index list\n\n");  exit(-1);}
    free(nindex32);
    if (g.edges > 0) {
      size_t wrote = fwrite(g.nlist, sizeof(g.nlist[0]), (size_t)g.edges, f);
      if (wrote != (size_t)g.edges) {fprintf(stderr, "ERROR: failed to write neighbor list\n\n");  exit(-1);}
    }
    if (g.eweight != NULL) {
      size_t wrote = fwrite(g.eweight, sizeof(g.eweight[0]), (size_t)g.edges, f);
      if (wrote != (size_t)g.edges) {fprintf(stderr, "ERROR: failed to write edge weights\n\n");  exit(-1);}
    }
    fclose(f);
    return;
  } else {
    int sentinel = -1;
    cnt = fwrite(&sentinel, sizeof(sentinel), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to write edges sentinel\n\n");  exit(-1);}
    cnt = fwrite(&g.edges, sizeof(g.edges), 1, f);  if (cnt != 1) {fprintf(stderr, "ERROR: failed to write edges (64-bit)\n\n");  exit(-1);}
  }

  cnt = fwrite(g.nindex, sizeof(g.nindex[0]), g.nodes + 1, f);  if (cnt != g.nodes + 1) {fprintf(stderr, "ERROR: failed to write neighbor index list\n\n");  exit(-1);}
  if (g.edges > 0) {
    cnt = (int)fwrite(g.nlist, sizeof(g.nlist[0]), (size_t)g.edges, f);
    if ((ull)cnt != g.edges) {fprintf(stderr, "ERROR: failed to write neighbor list\n\n");  exit(-1);}
  }
  if (g.eweight != NULL) {
    cnt = (int)fwrite(g.eweight, sizeof(g.eweight[0]), (size_t)g.edges, f);
    if ((ull)cnt != g.edges) {fprintf(stderr, "ERROR: failed to write edge weights\n\n");  exit(-1);}
  }
  fclose(f);
}

void freeECLgraph(ECLgraph &g)
{
  if (g.nindex != NULL) free(g.nindex);
  if (g.nlist != NULL) free(g.nlist);
  if (g.eweight != NULL) free(g.eweight);
  g.nindex = NULL;
  g.nlist = NULL;
  g.eweight = NULL;
}

#endif

# SCOUT: True-External Memory Dynamic Graph Maintenance on Memory-Constrained GPUs

SCOUT is a GPU framework for **maintaining graph analytics** on a large, continuously changing graph, instead of recomputing them from scratch after every change. Its base graph is kept as a storage-resident (mmap'd) binary CSR file rather than fully materialized in RAM, so the graph itself never has to fit in host or device memory. Incoming edge insertions/deletions are kept in an in-memory scratchpad on top of that base graph, and for each batch SCOUT fetches only the neighborhoods the update affects, repairs them on the GPU, and commits the changed entries of the solution. The same engine runs on discrete GPUs (explicit PCIe copies) and on devices where the CPU and GPU share one physical memory pool, like Jetson Orin (zero-copy access).

This repository maintains four problems with the same storage-resident engine:

| Problem | Source | Repair |
|---|---|---|
| `mis` — maximal independent set | `src/mis/scout_mis.cu` | LCL, fixed 2-hop repair region; CPU fetch of batch *K+1* overlaps the GPU repair of batch *K* |
| `coloring` — graph coloring | `src/coloring/scout_coloring.cu` | LCL, fixed 1-hop repair region |
| `bfs` — breadth-first search levels | `src/bfs/scout_bfs.cu` | non-LCL, adaptive repair of the affected subtrees |
| `cc` — connected components | `src/cc/scout_cc.cu` | non-LCL, adaptive bidirectional search on each deletion |

All four read the same `.egr` binary CSR graph format and the same batch-folder format, and are invoked with the same 5 positional arguments (see [Running](#running)).

## Requirements

- A CUDA-capable GPU and the CUDA toolkit (`nvcc`), tested against CUDA 12.4 (L40S) and CUDA 11.4 (Jetson AGX Orin). `nvcc` needs to support `-arch=native` (CUDA ≥ 11.6) or you'll need to pass your GPU's architecture explicitly (see [Building](#building)).
- A C++17-capable host compiler with OpenMP support (e.g. a reasonably recent `gcc`/`g++`).
- Python 3 with `numpy` for the scripts under `tools/` (graph conversion, batch generation, initial-MIS computation). Not needed to build or run the binaries themselves.

## Cloning

```bash
git clone <this-repository-url> SCOUT
cd SCOUT
```

## Building

`run.sh` builds whichever problem you ask for automatically the first time you run it (see [Running](#running)) — you don't need a separate build step for normal use. To build the binaries directly instead:

```bash
for p in mis coloring bfs cc; do
    nvcc -O3 -std=c++17 -Xcompiler -fopenmp -arch=native -I. src/$p/scout_$p.cu -o bin/scout_$p
done
```

`-arch=native` asks `nvcc` to target whatever GPU is installed on the build machine. On an older CUDA toolkit that doesn't support it, replace it with your GPU's explicit architecture flag instead, e.g. `-arch=sm_89` for an L40S or `-arch=sm_87` for a Jetson AGX Orin. `run.sh` picks this up from the `SCOUT_ARCH_FLAG` environment variable, so you don't need to edit the script to change it.

A binary at `bin/scout_<problem>` is reused as-is by `run.sh` on later runs.

## Running

The quickest way to see SCOUT working end to end, using the small bundled sample dataset under `datasets/sample/`:

```bash
./run.sh --problem=mis --sample
./run.sh --problem=coloring --sample
./run.sh --problem=bfs --sample
./run.sh --problem=cc --sample
```

Against your own graph and batch stream:

```bash
./run.sh --problem=cc \
    --graph=path/to/graph.egr \
    --init=path/to/initial_components.txt \
    --batch=path/to/batch_folder \
    --num-batches=10 \
    --gpu=0
```

- `--problem` —> `mis`, `coloring`, `bfs`, or `cc` (see the table above).
- `--graph` —> the base graph, as a SCOUT `.egr` binary CSR file.
- `--init` —> the initial solution this run seeds from and then maintains incrementally (see [File formats](#file-formats)).
- `--batch` —> a folder of batch files named `1.mtx`, `2.mtx`, ... (see [Batch files](#file-formats) below), applied in that numeric order.
- `--num-batches` —> how many of those batch files to actually replay. Defaults to the number of `*.mtx` files found in `--batch` if omitted.
- `--gpu` —> CUDA device ID to run on. Defaults to `0`.

Run `./run.sh --help` for the full flag reference, including the `NVCC`, `SCOUT_ARCH_FLAG`, and `SCOUT_SKIP_COMPACTION` environment overrides.

You can also invoke any built binary directly, without `run.sh`, using the same 5 positional arguments:

```bash
bin/scout_<problem> <graph.egr> <initial_state.txt> <batch_folder> <num_batches> <gpu_id>
```

Each run prints one line per batch and a final summary with the end-to-end time and the size of the maintained solution.

## Using your own graph

The `tools/` scripts take you from a plain-text edge list to everything `run.sh` needs:

1. **Convert to `.egr`.** Your edge list must be a MatrixMarket-style `.mtx` file listing each undirected edge in **both** directions (`u v` and `v u`), one directed entry per line, 1-indexed. `datasets/sample/sample.mtx` is a working example.

   ```bash
   python3 tools/efficient_converter.py your_graph.mtx your_graph.egr
   ```

2. **Compute the initial solution** to seed the incremental maintenance from:

   ```bash
   python3 tools/compute_initial_mis.py --egr your_graph.egr --out your_graph_mis.txt     # mis

   nvcc -O3 -arch=native tools/eclgc_fixed_src/eclgc.cu -o bin/eclgc
   bin/eclgc your_graph.egr                                      # coloring -> your_graph.egr.colors.txt

   nvcc -O3 -arch=native tools/bfs_ground_truth.cu -o bin/bfs_ground_truth
   bin/bfs_ground_truth your_graph.egr your_graph_bfs.txt 0      # bfs from source vertex 0

   nvcc -O3 -arch=native tools/excc_src/excc_v1.cu -o bin/excc
   bin/excc your_graph.egr                                       # cc -> your_graph.egr.components.txt
   ```

   For graphs with more than 2^31 adjacency entries, `tools/greedy_color.cpp` (`g++ -O3`) produces the initial coloring on the CPU.

3. **Generate batch files** — toggled insert/delete against current adjacency the first time each is replayed:

   ```bash
   python3 tools/generate_mixed_batches.py --egr your_graph.egr \
       --outdir your_batches --batch-size 100 --num-batches 10 --seed 42
   ```

   `generate_test_batches.py`, `generate_low_degree_batches.py`, and `generate_preferential_batches.py` produce deletion-only, low-degree, and high-degree-targeted streams with the same flags.

## File formats

**`.egr` binary CSR** (little-endian):
```
int32   nodes
int32   edge_count_or_sentinel   // -1 if the real count needs 64 bits
[int64  edge_count]              // present only if the above was -1
int32/int64  nindex[nodes + 1]   // int64 entries iff the sentinel was used
int32   nlist[edge_count]
```
`nindex[v] .. nindex[v+1]` indexes into `nlist` for vertex `v`'s neighbors. An undirected edge (u, v) must appear as a directed entry in **both** `nlist` rows.

**Initial-state files** (plain text, no header):

| Problem | Content |
|---|---|
| `mis` | one 1-indexed vertex ID per line, for every vertex in the set |
| `coloring` | one color per line, in vertex order |
| `bfs` | `vertex level` per line, 1-indexed vertex IDs |
| `cc` | one component ID per line, in vertex order |

**Batch files**: plain text, named `1.mtx`, `2.mtx`, ... inside the batch folder (processed in that numeric order). Each line is `u v` (1-indexed); the edge is inserted if absent and deleted if present.

## Project layout

```
SCOUT/
├── run.sh                    # build (if needed) + run any of the four problems
├── src/
│   ├── mis/scout_mis.cu              # maximal independent set
│   ├── coloring/scout_coloring.cu    # graph coloring
│   ├── bfs/scout_bfs.cu              # breadth-first search
│   └── cc/scout_cc.cu                # connected components
├── core/                     # shared headers used by all four binaries
│   ├── FlashGraph.h          # mmap-backed .egr reader + update scratchpad
│   ├── flash_compactor.h     # end-of-run base+scratchpad -> fresh .egr merge
│   ├── gpu_common.h          # CUDA_CHECK, device setup, buffer allocation
│   ├── cli_setup.h           # argument parsing, batch/initial-state file I/O
│   ├── reporting.h           # standardized console output
│   ├── round_buffer.h        # per-round GPU buffers (bfs, coloring)
│   ├── chunking.h            # bounded-degree round chunking
│   └── local_repair.h        # search bound and deferral policy (bfs, cc)
├── tools/                    # graph conversion / batch / initial-state utilities
├── datasets/sample/          # small bundled graph, initial states, and batches
└── bin/                      # build output (created by run.sh / nvcc)
```

`tools/eclgc_fixed_src` and `tools/excc_src` are adapted from the ECL graph codes and keep their original license headers.

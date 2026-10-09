#!/usr/bin/env python3
"""
Generate fixed-size random-edge batches directly from a .egr CSR file, for
any graph regardless of size -- generalizes roadnetca_fixed_batches.py (which
required a .mtx and materialized the full edge set as a Python set, infeasible
for a graph like com-Friendster's 3.43B edges) to work off the .egr directly,
without ever materializing more than the sampled batch itself.

Same synthetic-perturbation approach as roadnetca_fixed_batches.py: no real
temporal event stream exists for most graphs (BASMATI is the one exception,
handled by basmati_fixed_batches.py instead, which regroups its real event
stream rather than sampling), so batches are real edges drawn at random from
the base graph, without replacement across all requested batches, and SCOUT's
own runtime toggle (isAdjacent) turns each one into a deletion.

Memory-safe sampling: rather than materializing every edge, this draws random
FLAT POSITIONS into the .egr's own nlist array (memmapped, O(1) per lookup),
maps each position back to its source vertex via searchsorted on nindex
(fully materialized -- only O(nodes), as in the rest of the tools
does), and canonicalizes (min(u,v), max(u,v)) to dedupe the two directed
entries every undirected edge has. Draws in batches until enough distinct,
non-self-loop edges are collected -- collision/self-loop loss is negligible
against a graph with any real size, so this converges in 1-2 rounds even when
sampling a few thousand edges out of billions.

Output convention matches every other batch file in this pipeline: 1-indexed
vertex IDs (MatrixMarket-style, since the dynamic
SCOUT runtime's own batch loader both do "id - 1" on read), plain "src dst"
per line, no header.

Usage:
    python3 generate_test_batches.py --egr datasets/EGRs/com-Friendster.egr \
        --outdir datasets/Batches/com-Friendster/fixed100 \
        --batch-size 100 --num-batches 3 --seed 42
"""
import argparse
import os
import struct
import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        nodes = struct.unpack("i", f.read(4))[0]
        sentinel = struct.unpack("i", f.read(4))[0]
        if sentinel == -1:
            edges = struct.unpack("Q", f.read(8))[0]
            wide = True
        else:
            edges = sentinel
            wide = False
        header_end = f.tell()
    return nodes, edges, wide, header_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egr", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--num-batches", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nodes, edges, wide, header_end = read_header(args.egr)
    print(f"Graph: {nodes} nodes, {edges} directed CSR entries (wide={wide})")

    stride = 8 if wide else 4
    dtype = np.int64 if wide else np.int32
    nindex_raw = np.memmap(args.egr, dtype=dtype, mode="r", offset=header_end, shape=(nodes + 1,))
    nindex = np.array(nindex_raw, dtype=np.int64)  # O(nodes) -- safe to materialize
    nlist_offset = header_end + (nodes + 1) * stride
    nlist_mmap = np.memmap(args.egr, dtype=np.int32, mode="r", offset=nlist_offset, shape=(edges,))

    total_needed = args.batch_size * args.num_batches
    if total_needed > edges:
        raise ValueError(f"Need {total_needed} edges but graph only has {edges} directed CSR entries")

    rng = np.random.default_rng(args.seed)
    found = {}  # canonical (u, v) -> True, insertion order preserved (dict, py3.7+)
    draw_size = max(total_needed * 2, 1000)
    attempts = 0
    while len(found) < total_needed:
        attempts += 1
        if attempts > 50:
            raise RuntimeError(f"Could not find {total_needed} distinct non-self-loop edges after {attempts} draw rounds")
        positions = rng.integers(0, edges, size=draw_size)
        positions.sort()  # ascending, so searchsorted below can be done in one vectorized call
        # nindex[u] <= p < nindex[u+1]  =>  u = searchsorted(nindex, p, side='right') - 1
        us = np.searchsorted(nindex, positions, side="right") - 1
        vs = nlist_mmap[positions].astype(np.int64)
        for u, v in zip(us.tolist(), vs.tolist()):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key not in found:
                found[key] = True
                if len(found) >= total_needed:
                    break

    edges_list = list(found.keys())[:total_needed]
    idx = rng.permutation(len(edges_list))  # avoid any positional bias from draw order when chunking into batches
    edges_list = [edges_list[i] for i in idx]

    os.makedirs(args.outdir, exist_ok=True)
    for k in range(1, args.num_batches + 1):
        chunk = edges_list[(k - 1) * args.batch_size: k * args.batch_size]
        out_path = os.path.join(args.outdir, f"{k}.mtx")
        with open(out_path, "w") as f:
            for u, v in chunk:
                f.write(f"{u + 1} {v + 1}\n")  # 0-indexed CSR -> 1-indexed batch-file convention
        print(f"batch {k}: {len(chunk)} edges -> {out_path}")

    print(f"Wrote {args.num_batches} batches ({args.batch_size} edges each) to {args.outdir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generates fixed-size batches with a genuine insert/delete MIX, unlike
roadnetca_fixed_batches.py/generate_test_batches.py (which sample only real
existing edges -- every draw is adjacent on first encounter, so every batch
they produce is 100% deletions by construction).

Each batch is half real existing edges (adjacent -> classified as deletes by
every downstream consumer's own insert-if-not-adjacent/delete-if-adjacent
toggle logic) and half genuine non-edges (random vertex pairs confirmed NOT
adjacent -> classified as inserts by that same toggle logic). Written in the
same plain "src dst" (1-indexed) convention every tool in this pipeline
already shares, so flash/oom_async/hybrid_mis all interpret the identical
batch file identically -- a fair, apples-to-apples test across all three.

Usage:
    python3 generate_mixed_batches.py --egr base.egr \
        --outdir datasets/Batches/<name>/fixed<size> \
        --batch-size 100 --num-batches 1 --seed 42
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
    ap.add_argument("--insert-ratio", type=float, default=0.5,
                     help="Fraction of edges that are insertions (0.0-1.0). "
                          "Default 0.5 matches the original 50/50 behavior. "
                          "E.g. 0.2 -> 20%% insert / 80%% delete.")
    args = ap.parse_args()

    if not (0.0 <= args.insert_ratio <= 1.0):
        raise ValueError(f"--insert-ratio must be in [0.0, 1.0], got {args.insert_ratio}")

    nodes, edges, wide, header_end = read_header(args.egr)
    print(f"Base graph: {nodes} nodes, {edges} edges (wide={wide})")

    stride = 8 if wide else 4
    dtype = np.int64 if wide else np.int32
    nindex_raw = np.memmap(args.egr, dtype=dtype, mode="r", offset=header_end, shape=(nodes + 1,))
    nindex = np.array(nindex_raw, dtype=np.int64)
    nlist_offset = header_end + (nodes + 1) * stride
    nlist_mmap = np.memmap(args.egr, dtype=np.int32, mode="r", offset=nlist_offset, shape=(edges,))

    def get_neighbors(v):
        return set(int(x) for x in nlist_mmap[nindex[v]:nindex[v + 1]])

    rng = np.random.default_rng(args.seed)
    total_needed = args.batch_size * args.num_batches
    insert_target = round(total_needed * args.insert_ratio)
    half = total_needed - insert_target  # deletes; inserts = total_needed - half
    print(f"Target mix: {half} deletes + {insert_target} inserts "
          f"({args.insert_ratio*100:.0f}% insert / {(1-args.insert_ratio)*100:.0f}% delete)")

    # 1. Real existing edges, for deletions -- same sampling approach as
    #    generate_test_batches.py (random flat positions into nlist, mapped
    #    back to (u,v) via searchsorted).
    delete_edges = {}
    draw_size = max(half * 3, 1000) if half > 0 else 0
    attempts = 0
    while len(delete_edges) < half:
        attempts += 1
        if attempts > 50:
            raise RuntimeError(f"Could not find {half} distinct delete edges after {attempts} rounds")
        positions = rng.integers(0, edges, size=draw_size)
        positions.sort()
        us = np.searchsorted(nindex, positions, side="right") - 1
        vs = nlist_mmap[positions].astype(np.int64)
        for u, v in zip(us.tolist(), vs.tolist()):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key not in delete_edges:
                delete_edges[key] = True
                if len(delete_edges) >= half:
                    break
    print(f"Found {len(delete_edges)} real edges for deletion")

    # 2. Genuine non-edges, for insertions -- random vertex pairs, rejected
    #    if adjacent in the base graph OR already used as a delete edge.
    #    Neighbor sets are fetched lazily/cached (only for vertices actually
    #    drawn), memory-safe at any graph size same as the delete sampling.
    insert_count = total_needed - half
    insert_edges = {}
    neighbor_cache = {}
    attempts = 0
    while len(insert_edges) < insert_count:
        attempts += 1
        if attempts > 200:
            raise RuntimeError(f"Could not find {insert_count} distinct non-edges after {attempts} rounds")
        batch_size_draw = max((insert_count - len(insert_edges)) * 3, 1000)
        us = rng.integers(0, nodes, size=batch_size_draw)
        vs = rng.integers(0, nodes, size=batch_size_draw)
        for u, v in zip(us.tolist(), vs.tolist()):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key in delete_edges or key in insert_edges:
                continue
            if u not in neighbor_cache:
                neighbor_cache[u] = get_neighbors(u)
            if v in neighbor_cache[u]:
                continue  # actually adjacent, not a real non-edge
            insert_edges[key] = True
            if len(insert_edges) >= insert_count:
                break
    print(f"Found {len(insert_edges)} genuine non-edges for insertion")

    all_edges = list(delete_edges.keys()) + list(insert_edges.keys())
    idx = rng.permutation(len(all_edges))
    all_edges = [all_edges[i] for i in idx]

    os.makedirs(args.outdir, exist_ok=True)
    for k in range(1, args.num_batches + 1):
        chunk = all_edges[(k - 1) * args.batch_size: k * args.batch_size]
        out_path = os.path.join(args.outdir, f"{k}.mtx")
        with open(out_path, "w") as f:
            for u, v in chunk:
                f.write(f"{u + 1} {v + 1}\n")  # 0-indexed CSR -> 1-indexed batch convention
        print(f"batch {k}: {len(chunk)} edges -> {out_path}")

    print(f"Wrote {args.num_batches} mixed batch(es) ({args.batch_size} edges each: "
          f"~{half} deletes + ~{insert_count} inserts before shuffling) to {args.outdir}")


if __name__ == "__main__":
    main()

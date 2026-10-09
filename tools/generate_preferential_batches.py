"""
Generate fixed-size edge batches from a .egr CSR file, biased toward
high-degree vertices via preferential attachment: each sampled edge's
first endpoint is drawn with probability proportional to its own degree,
so hubs are affected far more often than low-degree vertices, producing
highly skewed changes concentrated around them. Works off the .egr
directly via memmap, without ever materializing the full edge set in RAM.

Companion to generate_test_batches.py (uniform-over-edges) and
generate_low_degree_batches.py (1/degree, low-degree-biased).

Usage:
    python3 generate_preferential_batches.py --egr datasets/EGRs/com-Friendster.egr \
        --outdir datasets/Batches/com-Friendster/hub100 \
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
    nindex = np.array(nindex_raw, dtype=np.int64)
    nlist_offset = header_end + (nodes + 1) * stride
    nlist_mmap = np.memmap(args.egr, dtype=np.int32, mode="r", offset=nlist_offset, shape=(edges,))

    degree = nindex[1:] - nindex[:-1]
    # Preferential attachment: weight proportional to degree itself, so
    # hubs dominate the sample. Degree-0 vertices are naturally excluded
    # (weight 0).
    weight = degree.astype(np.float64)
    total_weight = weight.sum()
    if total_weight <= 0:
        raise ValueError("Graph has no vertex with degree > 0 -- nothing to sample")
    p = weight / total_weight

    total_needed = args.batch_size * args.num_batches
    if total_needed > edges:
        raise ValueError(f"Need {total_needed} edges but graph only has {edges} directed CSR entries")

    rng = np.random.default_rng(args.seed)
    found = {}
    # Sampling is concentrated on a small set of hub vertices, so repeated
    # draws of the same hub collide with an already-found edge more often
    # than in the uniform sampler -- draw more per round and allow more
    # rounds to compensate.
    draw_size = max(total_needed * 4, 2000)
    attempts = 0
    while len(found) < total_needed:
        attempts += 1
        if attempts > 100:
            raise RuntimeError(f"Could not find {total_needed} distinct non-self-loop edges after {attempts} draw rounds")
        us = rng.choice(nodes, size=draw_size, p=p)
        deg_u = degree[us]
        offsets = (rng.random(size=draw_size) * deg_u).astype(np.int64)
        flat_pos = nindex[us] + offsets
        vs = nlist_mmap[flat_pos].astype(np.int64)
        for u, v in zip(us.tolist(), vs.tolist()):
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key not in found:
                found[key] = True
                if len(found) >= total_needed:
                    break

    edges_list = list(found.keys())[:total_needed]
    idx = rng.permutation(len(edges_list))
    edges_list = [edges_list[i] for i in idx]

    os.makedirs(args.outdir, exist_ok=True)
    for k in range(1, args.num_batches + 1):
        chunk = edges_list[(k - 1) * args.batch_size: k * args.batch_size]
        out_path = os.path.join(args.outdir, f"{k}.mtx")
        with open(out_path, "w") as f:
            for u, v in chunk:
                f.write(f"{u + 1} {v + 1}\n")
        print(f"batch {k}: {len(chunk)} edges -> {out_path}")

    print(f"Wrote {args.num_batches} preferential-attachment (hub-biased) batches ({args.batch_size} edges each) to {args.outdir}")


if __name__ == "__main__":
    main()

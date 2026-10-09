"""
Computes a greedy maximal independent set (MIS) directly from a .egr CSR
file, to use as the initial seed MIS that SCOUT's MIS binary maintains
incrementally as batches are replayed.

Output format matches what both binaries read for <initial_mis_context.txt>:
one 1-indexed node ID per line, no header.

Usage:
    python3 compute_initial_mis.py --egr datasets/sample/sample.egr \
        --out datasets/sample/sample_mis.txt
"""
import argparse
import struct
import numpy as np


def read_egr(path):
    with open(path, "rb") as f:
        nodes = struct.unpack("i", f.read(4))[0]
        sentinel = struct.unpack("i", f.read(4))[0]
        if sentinel == -1:
            edges = struct.unpack("q", f.read(8))[0]
            nindex = np.frombuffer(f.read((nodes + 1) * 8), dtype=np.int64)
        else:
            edges = sentinel
            nindex = np.frombuffer(f.read((nodes + 1) * 4), dtype=np.int32)
        nlist = np.frombuffer(f.read(edges * 4), dtype=np.int32)
    return nodes, nindex, nlist


def greedy_mis(nodes, nindex, nlist):
    in_mis = np.zeros(nodes, dtype=bool)
    blocked = np.zeros(nodes, dtype=bool)
    
    for v in range(nodes):
        if blocked[v]:
            continue
        in_mis[v] = True
        start, end = int(nindex[v]), int(nindex[v + 1])
        for u in nlist[start:end]:
            blocked[u] = True
    return in_mis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egr", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    nodes, nindex, nlist = read_egr(args.egr)
    print(f"{args.egr}: {nodes} nodes, {len(nlist)} directed edge entries")

    in_mis = greedy_mis(nodes, nindex, nlist)
    count = int(in_mis.sum())
    print(f"greedy MIS: {count} members ({100 * count / nodes:.1f}% of nodes)")

    with open(args.out, "w") as f:
        for v in range(nodes):
            if in_mis[v]:
                f.write(f"{v + 1}\n")  # 1-indexed to match the binaries' own convention
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

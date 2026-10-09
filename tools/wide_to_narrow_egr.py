#!/usr/bin/env python3
"""Converts this project's wide-format .egr (int32 nodes, -1 sentinel,
int64 edge count, int64 nindex[nodes+1], int32 nlist[edges]) to the
narrow/plain ECLgraph.h format ECL-GC's own reader expects (int32 nodes,
int32 edges, int32 nindex[nodes+1], int32 nlist[edges], no sentinel, no
eweight block written -- ECL-GC treats a short read of eweight as "no
weights" and proceeds unweighted, matching graph coloring's own
unweighted problem definition).

Only viable when edges fits in a signed int32 (~2.147B) -- flagged loudly
otherwise rather than silently truncating.

Usage: wide_to_narrow_egr.py <in_wide.egr> <out_narrow.egr>
"""
import struct, sys

def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path, "rb") as f:
        nodes = struct.unpack("i", f.read(4))[0]
        sentinel = struct.unpack("i", f.read(4))[0]
        if sentinel == -1:
            edges = struct.unpack("q", f.read(8))[0]
            nindex_raw = f.read(8 * (nodes + 1))
            nindex = list(struct.unpack(f"{nodes+1}q", nindex_raw))
        else:
            edges = sentinel
            nindex_raw = f.read(4 * (nodes + 1))
            nindex = list(struct.unpack(f"{nodes+1}i", nindex_raw))
        nlist_raw = f.read(4 * edges)

    if edges > 2**31 - 1:
        print(f"ERROR: {edges} edges overflows int32 -- ECL-GC's narrow "
              f"format cannot represent this graph. Not writing output.",
              file=sys.stderr)
        sys.exit(1)

    with open(out_path, "wb") as f:
        f.write(struct.pack("i", nodes))
        f.write(struct.pack("i", edges))
        f.write(struct.pack(f"{nodes+1}i", *nindex))
        f.write(nlist_raw)

    print(f"Wrote {out_path}: {nodes} nodes, {edges} edges (narrow format)")

if __name__ == "__main__":
    main()

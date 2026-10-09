import sys
import struct
import numpy as np

def convert_mtx_to_egr(input_file, output_file):
    print(f"Reading header of {input_file}...")
    num_nodes = 0
    num_edges = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                num_nodes = max(int(parts[0]), int(parts[1]))
                num_edges = int(parts[2]) if len(parts) >= 3 else 0
                break
                
    if num_edges == 0:
        print("Detecting edge count automatically...")
        with open(input_file, 'r') as f:
            for line in f:
                if not line.startswith('%') and len(line.split()) >= 2:
                    num_edges += 1
        num_edges -= 1 # subtract dim line
        
    print(f"Graph stats: {num_nodes} nodes, {num_edges} edges.")
    
    # Needs to be a valid 64-bit egr
    # 64-bit format: [int32 nodes][int32 -1][int64 edges][int64[] nindex][int32[] nlist]
    
    # Pass 1: Count degrees to build nindex
    print("Pass 1: Computing node degrees (Out-of-core)...")
    degrees = np.zeros(num_nodes + 1, dtype=np.uint32)
    
    lines_read = 0
    with open(input_file, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.startswith('%'): continue
                in_header = False
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                u = int(parts[0]) - 1 # 1-indexed to 0-indexed
                if u < num_nodes:
                    degrees[u] += 1
                    
            lines_read += 1
            if lines_read % 100000000 == 0:
                print(f"  Processed {lines_read} edges...")
                
    print("Building nindex array...")
    # Cumulative sum to get starting indices
    nindex = np.zeros(num_nodes + 2, dtype=np.int64)
    np.cumsum(degrees, out=nindex[1:])
    # Cleanup degrees array to save ram
    del degrees
    
    num_edges = nindex[-2]  # Actual edges processed
    print(f"Final valid edges: {num_edges}")
    
    # We write nodes, -1, and edges directly
    pass2_positions = nindex[:-1].copy()
    
    print("Pass 2: Populating edge lists directly to disk...")
    # Because nlist is huge, we will memory map the final output file
    
    with open(output_file, 'wb') as f:
        # Write header
        f.write(struct.pack('i', num_nodes))
        f.write(struct.pack('i', -1))
        f.write(struct.pack('q', num_edges))
        
        # Write nindex
        f.write(nindex[:-1].tobytes())
        
        # Determine offset for nlist
        nlist_offset = f.tell()
        
        # Pre-allocate nlist area with zeros
        print("  Pre-allocating binary file...")
        f.truncate(nlist_offset + (num_edges * 4))
        
    print("  Mapping binary payload to memory...")
    nlist_mmap = np.memmap(output_file, dtype=np.int32, mode='r+', offset=nlist_offset, shape=(num_edges,))
    
    lines_read = 0
    with open(input_file, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.startswith('%'): continue
                in_header = False
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                u = int(parts[0]) - 1
                v = int(parts[1]) - 1
                
                if u < num_nodes:
                    idx = pass2_positions[u]
                    nlist_mmap[idx] = v
                    pass2_positions[u] += 1
                    
            lines_read += 1
            if lines_read % 100000000 == 0:
                print(f"  Mapped {lines_read} edges...")
                
    nlist_mmap.flush()
    print("Conversion complete successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python efficient_converter.py <input.mtx> <output.egr>")
        sys.exit(1)
    convert_mtx_to_egr(sys.argv[1], sys.argv[2])

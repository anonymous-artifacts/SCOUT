#!/bin/bash
# run.sh -- build (if needed) and run one of SCOUT's four problems.
#
# Usage:
#   ./run.sh --problem=mis|coloring|bfs|cc --graph=<graph.egr> \
#       --init=<initial_state.txt> --batch=<batch_folder> \
#       [--num-batches=N] [--gpu=ID]
#
#   ./run.sh --problem=mis|coloring|bfs|cc --sample
#       (shortcut: fills in graph/init/batch/num-batches from the small
#        bundled sample dataset under datasets/sample/)
#
# --num-batches defaults to the number of *.mtx files found in the batch
# folder if not given. --gpu defaults to 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<EOF
Usage: $0 --problem=mis|coloring|bfs|cc --graph=<graph.egr> --init=<initial_state.txt> --batch=<batch_folder> [--num-batches=N] [--gpu=ID]
       $0 --problem=mis|coloring|bfs|cc --sample

  --problem       mis, coloring, bfs, or cc
  --graph         path to the base graph, as a SCOUT .egr binary CSR file
  --init          path to the initial solution this run seeds from (format depends on --problem)
  --batch         path to a folder of batch files named 1.mtx, 2.mtx, ... (see datasets/sample/batches)
  --num-batches   how many batch files to replay (default: count of *.mtx files in --batch)
  --gpu           CUDA device id to run on (default: 0)
  --sample        use the small bundled sample dataset instead of --graph/--init/--batch

Environment overrides:
  NVCC              nvcc binary to build with (default: nvcc on PATH)
  SCOUT_ARCH_FLAG   nvcc architecture flag (default: -arch=native, needs CUDA >= 11.6;
                    on an older toolkit set this explicitly, e.g. SCOUT_ARCH_FLAG=-arch=sm_87)
  SCOUT_SKIP_COMPACTION=1   mis: skip the end-of-run .egr compaction step
EOF
}

PROBLEM=""
GRAPH=""
INIT=""
BATCH=""
NUM_BATCHES=""
GPU="0"
USE_SAMPLE=0

for arg in "$@"; do
    case "$arg" in
        --problem=*)     PROBLEM="${arg#--problem=}" ;;
        --graph=*)       GRAPH="${arg#--graph=}" ;;
        --init=*)        INIT="${arg#--init=}" ;;
        --batch=*)       BATCH="${arg#--batch=}" ;;
        --num-batches=*) NUM_BATCHES="${arg#--num-batches=}" ;;
        --gpu=*)         GPU="${arg#--gpu=}" ;;
        --sample)        USE_SAMPLE=1 ;;
        -h|--help)       usage; exit 0 ;;
        *)
            echo "ERROR: unrecognized argument: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

case "$PROBLEM" in
    mis)      SAMPLE_INIT="sample_mis.txt" ;;
    coloring) SAMPLE_INIT="sample_colors.txt" ;;
    bfs)      SAMPLE_INIT="sample_bfs.txt" ;;
    cc)       SAMPLE_INIT="sample_components.txt" ;;
    "")       echo "ERROR: --problem=mis|coloring|bfs|cc is required" >&2; usage; exit 1 ;;
    *)        echo "ERROR: --problem must be mis, coloring, bfs, or cc, got '$PROBLEM'" >&2; exit 1 ;;
esac

if [ "$USE_SAMPLE" = "1" ]; then
    GRAPH="${GRAPH:-$SCRIPT_DIR/datasets/sample/sample.egr}"
    INIT="${INIT:-$SCRIPT_DIR/datasets/sample/$SAMPLE_INIT}"
    BATCH="${BATCH:-$SCRIPT_DIR/datasets/sample/batches}"
fi

for v in GRAPH INIT BATCH; do
    if [ -z "${!v}" ]; then
        echo "ERROR: --$(echo $v | tr 'A-Z' 'a-z') is required (or use --sample)" >&2
        usage
        exit 1
    fi
done
[ -f "$GRAPH" ] || { echo "ERROR: graph file not found: $GRAPH" >&2; exit 1; }
[ -f "$INIT" ]  || { echo "ERROR: initial-state file not found: $INIT" >&2; exit 1; }
[ -d "$BATCH" ] || { echo "ERROR: batch folder not found: $BATCH" >&2; exit 1; }

if [ -z "$NUM_BATCHES" ]; then
    NUM_BATCHES=$(find "$BATCH" -maxdepth 1 -name '*.mtx' | wc -l | tr -d ' ')
fi

SRC="$SCRIPT_DIR/src/$PROBLEM/scout_$PROBLEM.cu"
BIN="$SCRIPT_DIR/bin/scout_$PROBLEM"
if [ ! -x "$BIN" ]; then
    NVCC="${NVCC:-nvcc}"
    ARCH="${SCOUT_ARCH_FLAG:--arch=native}"
    echo "Building $BIN ..."
    mkdir -p "$SCRIPT_DIR/bin"
    "$NVCC" -O3 -std=c++17 -Xcompiler -fopenmp $ARCH -I"$SCRIPT_DIR" "$SRC" -o "$BIN"
fi

exec "$BIN" "$GRAPH" "$INIT" "$BATCH" "$NUM_BATCHES" "$GPU"

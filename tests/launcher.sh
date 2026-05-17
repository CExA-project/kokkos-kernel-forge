#!/bin/bash

set -o pipefail

if [ $# -lt 3 ]; then
  echo "The script expects at least 3 arguments"
  exit 1
fi

LIBKKF_PATH=$1
EXECUTABLE=$2
REPLAY_EXECUTABLE=$3
COMPARE_OUTPUT=$([ -z "$4" ]; echo $?)

OUTPUT_DIR=$(mktemp -d)
REFERENCE_OUTPUT="$OUTPUT_DIR/reference.txt"
REPLAY_OUTPUT="$OUTPUT_DIR/replay.txt"

clean_exit() {
  rm -rf "$OUTPUT_DIR"
  exit $1
}

rm -f *.h5

KOKKOS_TOOLS_LIBS="$LIBKKF_PATH" KOKKOS_TOOLS_ARGS="--kkf-dump-kernel-label=test_kernel" $EXECUTABLE | tee "$REFERENCE_OUTPUT"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  clean_exit $EXIT_CODE
fi

$REPLAY_EXECUTABLE --kernel-replayer-dump="$(ls *_in.h5)" --kernel-replayer-out-dump="$(ls *_out.h5)" | tee "$REPLAY_OUTPUT"
if [ $EXIT_CODE -ne 0 ]; then
  clean_exit $EXIT_CODE
fi

if [ $COMPARE_OUTPUT -eq 0 ]; then
  clean_exit 0
fi

CLEAN_REFERENCE_OUTPUT="$OUTPUT_DIR/clean_reference.txt"
grep -vF "[kokkos-hooks]" "$REFERENCE_OUTPUT" > "$CLEAN_REFERENCE_OUTPUT"
if ! diff "$CLEAN_REFERENCE_OUTPUT" "$REPLAY_OUTPUT"; then
  clean_exit 1
fi


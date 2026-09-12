#!/usr/bin/env bash
# Same pipeline as mirror_sdcard.sh but only the first 200 MB, and without
# touching the services - to prove the plumbing before the real run.
set -uo pipefail
dd if=/dev/mmcblk0 bs=4M count=50 iflag=direct status=none \
  | tee >(sha256sum | awk '{print "[selftest] raw sha256: "$1}' >&2) \
  | zstd -3 -T4 -c
echo "[selftest] done" >&2

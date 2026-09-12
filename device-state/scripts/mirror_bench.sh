#!/usr/bin/env bash
# Measure how fast the card can be read, to size the mirror job honestly.
set -u
sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
echo "read speed (1 GB, direct):" >&2
dd if=/dev/mmcblk0 of=/dev/null bs=4M count=250 iflag=direct 2>&1 | tail -1 >&2
echo "compression of the first 2 GB at zstd -3:" >&2
raw=$((512*1048576))
comp=$(dd if=/dev/mmcblk0 bs=4M count=128 iflag=direct 2>/dev/null | zstd -3 -T4 -c 2>/dev/null | wc -c)
awk -v r="$raw" -v c="$comp" 'BEGIN{printf "  512 MB -> %.0f MB  (%.2fx)\n", c/1048576, r/c}' >&2

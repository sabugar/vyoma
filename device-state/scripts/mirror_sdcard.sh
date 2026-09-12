#!/usr/bin/env bash
# Stream a compressed image of the whole boot device to stdout.
#
# The whole device, not just the root partition: this card carries fifteen
# partitions and the Jetson boot chain needs all of them, so an image of
# /dev/mmcblk0p1 alone would restore to a card that does not boot.
#
# The application is stopped first so the filesystem is quiet while it is
# read. The image is still only crash-consistent - ext4 replays its journal
# on first mount, which is what would happen after a power cut - so this is
# a restore-grade backup, not a database snapshot.
#
# Progress and the checksum of the raw device go to stderr; stdout carries
# image bytes only, so the caller can redirect it straight into a file.
set -uo pipefail

restore_services() {
    systemctl start pocketinfer.service 2>/dev/null
    systemctl start ollama.service 2>/dev/null
    echo "[mirror] services restarted" >&2
}
trap restore_services EXIT INT TERM

echo "[mirror] stopping services so the card is quiet" >&2
systemctl stop pocketinfer.service 2>/dev/null
systemctl stop ollama.service 2>/dev/null
sync; sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

SIZE=$(blockdev --getsize64 /dev/mmcblk0)
echo "[mirror] device /dev/mmcblk0 = $((SIZE/1073741824)) GB, reading at ~87 MB/s" >&2
echo "[mirror] expect roughly 25 minutes; stdout is zstd-compressed" >&2

dd if=/dev/mmcblk0 bs=4M iflag=direct status=progress 2>/tmp/mirror_progress \
  | tee >(sha256sum | awk '{print "[mirror] sha256 of the raw device: "$1}' >&2) \
  | zstd -3 -T4 -c
rc=$?
tail -1 /tmp/mirror_progress >&2
echo "[mirror] dd exit status $rc" >&2
exit $rc

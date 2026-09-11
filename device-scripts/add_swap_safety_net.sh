#!/bin/bash
set -e
if [ ! -f /swapfile_safety ]; then
  fallocate -l 3G /swapfile_safety
  chmod 600 /swapfile_safety
  mkswap /swapfile_safety
fi
swapon -p -3 /swapfile_safety 2>&1 || echo "(already on)"
grep -q swapfile_safety /etc/fstab || echo '/swapfile_safety none swap sw,pri=-3 0 0' >> /etc/fstab
swapon --show

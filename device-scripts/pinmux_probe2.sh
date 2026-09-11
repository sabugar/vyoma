#!/bin/bash
for d in /sys/kernel/debug/pinctrl/2430000.pinmux /sys/kernel/debug/pinctrl/c300000.pinmux; do
  echo "=== $d ==="
  ls "$d" 2>/dev/null
  for f in "$d"/pinconf-pins "$d"/pins; do
    [ -f "$f" ] || continue
    echo "--- $(basename $f): PAC entries ---"
    grep -i "pac" "$f" 2>/dev/null | head -12
  done
done

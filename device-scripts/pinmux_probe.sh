#!/bin/bash
echo "=== pinctrl maps for PAC.06 / pin 7 ==="
for f in /sys/kernel/debug/pinctrl/*/pinconf-pins /sys/kernel/debug/pinctrl/*/pins; do
  [ -f "$f" ] || continue
  hit=$(grep -in "pac.06\|pac_06" "$f" 2>/dev/null | head -3)
  [ -n "$hit" ] && echo "--- $f ---" && echo "$hit"
done
echo "=== available pinctrl dirs ==="
ls -d /sys/kernel/debug/pinctrl/*/ 2>/dev/null | head
echo "=== gpio debug ==="
head -20 /sys/kernel/debug/gpio 2>/dev/null | grep -iE "gpiochip0|144|PAC" | head -5

#!/bin/bash
set -e

echo "[1/3] Disabling slow disk swapfile (eMMC) - keeping only fast zram..."
swapoff /swapfile 2>&1 || echo "(already off)"
sed -i '\|^/swapfile|d' /etc/fstab
grep -i swap /etc/fstab || echo "(fstab clean - no disk swap will re-enable on reboot)"

echo ""
echo "[2/3] Restarting bhashini_models.service (loads NMT lazy-indic-indic fix)..."
systemctl restart bhashini_models.service
echo "waiting for health..."
for i in $(seq 1 40); do
  resp=$(curl -s -m 2 http://localhost:11400/health 2>/dev/null || true)
  if [ "$resp" = '{"result":"success"}' ]; then
    echo "healthy after $((i*3))s"
    break
  fi
  sleep 3
done

echo ""
echo "[3/3] Restarting pocketinfer.service..."
systemctl restart pocketinfer.service
sleep 5

echo ""
echo "=== DONE. Final memory state: ==="
free -h
echo ""
swapon --show

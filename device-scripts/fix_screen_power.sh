#!/bin/bash
# ASHA Sahayak stability fixes - 14 Aug 2026
# 1. Screen wapas on (POCKETINFER_UI=1)
# 2. displayio CPU-spin fix (sleep(0.0) -> sleep(0.05): screen chalegi, core free)
# 3. Power mode 25W -> 15W (brownout reboots rokne ke liye)
# 4. pocketinfer restart
set -e

echo "[1/4] displayio CPU-spin patch..."
cp -n /usr/local/lib/python3.10/dist-packages/displayio/__init__.py /usr/local/lib/python3.10/dist-packages/displayio/__init__.py.bak
sed -i 's/time\.sleep(0\.0)/time.sleep(0.05)/' /usr/local/lib/python3.10/dist-packages/displayio/__init__.py
grep -n "time.sleep" /usr/local/lib/python3.10/dist-packages/displayio/__init__.py

echo "[2/4] Screen re-enable (POCKETINFER_UI=1) in pocketinfer.service..."
if ! grep -q "POCKETINFER_UI" /etc/systemd/system/pocketinfer.service; then
  sed -i '/^\[Service\]/a Environment=POCKETINFER_UI=1' /etc/systemd/system/pocketinfer.service
fi
grep "Environment" /etc/systemd/system/pocketinfer.service

echo "[3/4] Power mode 25W -> 15W (reboot rokne ke liye)..."
nvpmodel -m 0
nvpmodel -q | head -2

echo "[4/4] pocketinfer.service restart..."
systemctl daemon-reload
systemctl restart pocketinfer.service
sleep 3
systemctl status pocketinfer.service --no-pager | head -4

echo ""
echo "=== SAB HO GAYA. Screen 1-2 minute mein wapas aa jaayegi. ==="

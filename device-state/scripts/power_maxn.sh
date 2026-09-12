#!/usr/bin/env bash
# Set Jetson power mode 2 (maxn). No arguments, so the scoped
# sudoers rule (/usr/bin/bash /home/ubuntu/*.sh) matches exactly.
set -u
/usr/sbin/nvpmodel -m 2 -f /etc/nvpmodel.conf < /dev/null
sleep 2
echo "pmode: $(cat /var/lib/nvpmodel/status 2>/dev/null)"
echo "cpu max: $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)/1000 )) MHz"

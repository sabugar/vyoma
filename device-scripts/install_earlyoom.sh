#!/bin/bash
set -e
apt-get update -qq
apt-get install -y earlyoom
# Prefer killing our heavy services (bhashini_models has systemd Restart=always,
# so a clean kill just costs one ~60s reload instead of a full board watchdog
# reset + reboot). Never kill pocketinfer/ollama/bhashini's parent shells by
# accident - earlyoom picks the largest oom_score process by default, which on
# this box is reliably bhashini_models' python process.
# -m 10: act while 10% RAM is still free (well before real thrashing)
# -s 10: act while 10% swap is still free
# -r 0: continuous monitoring (poll every 100ms is default, fine)
cat > /etc/default/earlyoom << 'EOF'
EARLYOOM_ARGS="-m 10 -s 10 -r 0 --prefer '(bhashini_models|python.*main.py)' --avoid '(systemd|sshd|Xorg|ollama$)'"
EOF
systemctl enable earlyoom
systemctl restart earlyoom
sleep 2
systemctl status earlyoom --no-pager | head -8

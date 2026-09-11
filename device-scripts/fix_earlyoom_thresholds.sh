#!/bin/bash
set -e
cat > /etc/default/earlyoom << 'EOF'
EARLYOOM_ARGS="-m 1 -s 80 -r 0 --prefer ^python$ --avoid (ollama|llama-server|pocketinfer-ser|systemd|sshd|Xorg)"
EOF
systemctl restart earlyoom
sleep 1
journalctl -u earlyoom --since "5 seconds ago" --no-pager | tail -8

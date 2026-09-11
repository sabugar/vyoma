#!/bin/bash
set -e
sed -i 's/--log-level debug/--log-level info/' /etc/systemd/system/pocketinfer.service
systemctl daemon-reload
systemctl restart pocketinfer.service
grep ExecStart /etc/systemd/system/pocketinfer.service

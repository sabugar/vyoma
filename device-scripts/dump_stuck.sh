#!/bin/bash
PID=$(pgrep -f "pocketinfer-service --app")
/home/ubuntu/.local/bin/py-spy dump --pid $PID

#!/bin/bash
/home/ubuntu/.local/bin/py-spy dump --pid $(pgrep -f "pocketinfer-service --app")

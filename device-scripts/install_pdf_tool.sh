#!/bin/bash
set -e
apt-get install -y --no-install-recommends chromium-browser 2>&1 | tail -2 || \
apt-get install -y --no-install-recommends chromium 2>&1 | tail -2

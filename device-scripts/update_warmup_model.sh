#!/bin/bash
set -e
sed -i 's/"model":"[^"]*"/"model":"qwen3:1.7b"/' /usr/local/bin/ollama-warmup.sh
cat /usr/local/bin/ollama-warmup.sh

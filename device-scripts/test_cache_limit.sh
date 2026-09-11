#!/bin/bash
set -e
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/cache-limit.conf << 'EOF'
[Service]
# llama.cpp's server keeps a prompt cache (KV state per unique prompt). Its
# default limit is 8192 MiB - far more than this 8GB board has. Every RAG
# query here has a unique prompt (different retrieved context), so nothing is
# ever reused: the cache is pure growth. Measured: 996 MiB after 50 queries,
# still climbing. Cap it low - we lose nothing because cache hits never happen.
Environment="LLAMA_ARG_CACHE_RAM=128"
EOF
systemctl daemon-reload
systemctl restart ollama.service
sleep 6
systemctl restart ollama-warmup.service
sleep 10
echo "--- warmup done, ab ek query bhejte hai ---"

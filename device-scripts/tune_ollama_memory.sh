#!/bin/bash
set -e
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/cache-limit.conf << 'EOF'
[Service]
# llama.cpp's server keeps a prompt cache (KV state per unique prompt). Its
# default limit is 8192 MiB - far more than this 8GB board has. Every RAG
# query here has a unique prompt (different retrieved context), so a cache hit
# never happens: the cache is pure growth. Measured 996 MiB after 50 queries
# and still climbing, which is what actually pushed the board into swap
# thrashing. Capped low; we lose nothing because it never hits.
Environment="LLAMA_ARG_CACHE_RAM=64"
# Quantise the K/V cache to 8-bit. At num_ctx=2048 the f16 KV buffer is
# 224 MiB; q8_0 roughly halves that. Quality impact is negligible for the
# short, single-turn extractive prompts this app sends, and every 100MB
# matters when the vision-capable LLM and the ASR/NMT/TTS service have to
# coexist in 7.6GB. Requires flash attention, which this build enables.
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
systemctl daemon-reload
echo "config written:"
cat /etc/systemd/system/ollama.service.d/cache-limit.conf

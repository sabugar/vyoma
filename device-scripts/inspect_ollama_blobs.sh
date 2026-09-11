#!/bin/bash
ls -la /usr/share/ollama/.ollama/models/blobs/ | sort -k5 -n | tail -8
echo "=== manifest for qwen3-vl ==="
find /usr/share/ollama/.ollama/models/manifests -type f | grep -i qwen3-vl
echo "=== manifest content ==="
find /usr/share/ollama/.ollama/models/manifests -type f | grep -i qwen3-vl | head -1 | xargs cat | python3 -m json.tool 2>/dev/null | head -40

#!/bin/bash
# Expert memory budget: incremental measurement of each service's real
# contribution to system memory, by stopping everything and bringing
# services up one at a time, measuring `free` after each stable stage.
set -e

snap() {
  local label="$1"
  local used=$(free -m | awk '/^Mem:/{print $3}')
  local avail=$(free -m | awk '/^Mem:/{print $7}')
  local swap=$(free -m | awk '/^Swap:/{print $3}')
  printf "%-32s used=%5dMB  available=%5dMB  swap=%5dMB\n" "$label" "$used" "$avail" "$swap"
  echo "$label,$used,$avail,$swap" >> /tmp/memory_budget.csv
}

echo "label,used_mb,available_mb,swap_mb" > /tmp/memory_budget.csv

echo "=== Stopping everything ==="
systemctl stop pocketinfer.service bhashini_models.service ollama.service ollama-warmup.service 2>&1 | grep -v "^$" || true
sleep 4
snap "0_bare_os_floor"

echo ""
echo "=== Starting ollama.service (daemon only, no model loaded) ==="
systemctl start ollama.service
sleep 5
snap "1_ollama_daemon_idle"

echo ""
echo "=== Loading LLM (qwen3-vl:2b) via warmup ==="
systemctl start ollama-warmup.service
sleep 15
snap "2_llm_loaded"

echo ""
echo "=== Starting bhashini_models (ASR+NMT+TTS) ==="
systemctl start bhashini_models.service
echo "waiting for health..."
for i in $(seq 1 40); do
  resp=$(curl -s -m 2 http://localhost:11400/health 2>/dev/null)
  [ "$resp" = '{"result":"success"}' ] && break
  sleep 3
done
sleep 2
snap "3_bhashini_loaded"

echo ""
echo "=== Starting pocketinfer.service (app + RAG) ==="
systemctl start pocketinfer.service
sleep 8
snap "4_pocketinfer_loaded"

echo ""
echo "=== FULL REPORT ==="
column -t -s, /tmp/memory_budget.csv
echo ""
echo "=== Incremental cost per stage ==="
awk -F, 'NR==1{next} NR==2{prev=$2; print "bare OS floor: "$2"MB"; next} {print $1": +"($2-prev)"MB (delta)"; prev=$2}' /tmp/memory_budget.csv

#!/bin/bash
# Restart the whole inference stack in dependency order.
# Order matters: ollama must load the LLM while the board still has free
# memory, otherwise llama.cpp fits the model to whatever is left and silently
# splits it across CPU/GPU (seen as e.g. "46%/54% CPU/GPU" in `ollama ps`),
# which is many times slower. So: stop everything, load the LLM first, then
# bring the heavy ASR/NMT/TTS service up around it.
set -e
systemctl stop pocketinfer.service bhashini_models.service
sleep 3
systemctl restart ollama.service
sleep 6
systemctl restart ollama-warmup.service
sleep 12
echo "--- LLM placement ---"
ollama ps
systemctl start bhashini_models.service
for i in $(seq 1 60); do
  r=$(curl -s -m 2 http://localhost:11400/health 2>/dev/null || true)
  [ "$r" = '{"result":"success"}' ] && echo "bhashini healthy after $((i*3))s" && break
  sleep 3
done
systemctl start pocketinfer.service
sleep 10
echo "--- final ---"
ollama ps
free -m | awk '/^Mem:/{print "used="$3"MB avail="$7"MB"}; /^Swap:/{print "swap="$3"MB"}'

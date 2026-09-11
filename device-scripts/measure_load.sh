#!/bin/bash
timeout 14 tegrastats --interval 500 > /tmp/power_load.txt 2>/dev/null &
sleep 1
python3 - << 'PY'
import requests
q="बच्चों का वजन कितनी बार लेना चाहिए"
t=requests.post("http://localhost:11400/tts",json={"text":q,"language":"hi"},timeout=180).json()
a=requests.post("http://localhost:11400/asr",json={"audio_base64":t["audio_base64"],"language":"hi"},timeout=180).json()
n=requests.post("http://localhost:11400/nmt",json={"text":a["text"],"src_lang":"hi","tgt_lang":"EN"},timeout=180).json()
r=requests.post("http://127.0.0.1:11434/api/generate",json={"model":"qwen3:1.7b",
  "prompt":"Answer briefly: "+n["translated_text"]+"\n<think>\n\n</think>\n\n","raw":True,"stream":False,
  "keep_alive":-1,"options":{"num_predict":90,"num_ctx":2048}},timeout=240).json()
n2=requests.post("http://localhost:11400/nmt",json={"text":r.get("response","ok")[:150],"src_lang":"EN","tgt_lang":"hi"},timeout=180).json()
requests.post("http://localhost:11400/tts",json={"text":n2["translated_text"],"language":"hi"},timeout=180)
PY
wait
grep -oE "VDD_IN [0-9]+mW" /tmp/power_load.txt | awk '{if($2>m)m=$2; s+=$2; n++} END {printf "  avg %.0f mW | peak %.0f mW  (%d samples)\n", s/n, m, n}'

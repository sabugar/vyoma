"""Score the whole Hindi pipeline against known-correct answers.

Written because fixes were being judged one question at a time, which cannot
tell an improvement from a trade: the retrieval and prompt changes that helped
scenario questions had already been seen to break factual ones. Each question
carries the fact from the manual its answer must contain, so any change can be
re-scored in a minute.

Runs text-only by default (skips ASR) to isolate retrieval/LLM/NMT quality;
pass --audio to include TTS->ASR so ASR errors are measured too.
"""
import argparse
import base64
import json
import sys
import time

import requests

sys.path.insert(0, "/home/ubuntu/pocket-infer-sw/python")
from pocketinfer.applications.hear_the_world import HearTheWorld  # noqa: E402

BH = "http://localhost:11400"
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:1.7b"
# What questions.json's off-topic case asserts against, and what the device
# actually says when it refuses.
REFUSAL = "I do not have this information"


# The prompt and the token budget live on the application class. This file used
# to keep its own copy of both; they drifted, and the eval went on reporting a
# score for a prompt the device had stopped using.
from pocketinfer.models.ollama import Ollama  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true",
                    help="round-trip through TTS+ASR to include ASR errors")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    app = HearTheWorld.__new__(HearTheWorld)
    app.knowledge_chunks = app._load_knowledge_chunks()
    questions = json.load(open("/home/ubuntu/asha_eval/questions.json"))

    passed = 0
    for q in questions:
        hindi = q["hi"]
        if args.audio:
            tts = requests.post(f"{BH}/tts", json={"text": hindi, "language": "hi"},
                                timeout=240).json()
            asr = requests.post(f"{BH}/asr",
                                json={"audio_base64": tts["audio_base64"],
                                      "language": "hi"}, timeout=240).json()
            hindi = asr["text"]

        english = requests.post(f"{BH}/nmt",
                                json={"text": hindi, "src_lang": "hi",
                                      "tgt_lang": "EN"}, timeout=240
                                ).json()["translated_text"]
        chunks = app._retrieve_context(english, top_k=args.top_k)
        if chunks:
            prompt = HearTheWorld.build_prompt(chr(10).join(chunks), english)
            t0 = time.time()
            resp = requests.post(OLLAMA, json={
                "model": MODEL, "prompt": prompt + "\n<think>\n\n</think>\n\n",
                "raw": True, "stream": False, "keep_alive": -1,
                "options": {"num_predict": Ollama.NUM_PREDICT,
                            "num_ctx": Ollama.NUM_CTX,
                            "temperature": Ollama.TEMPERATURE}}, timeout=300).json()
            answer = resp.get("response", "").strip()
            if HearTheWorld.NO_ANSWER_SENTINEL in answer.upper():
                # The device speaks its fixed out-of-scope reply here.
                answer = REFUSAL
            llm_s = time.time() - t0
        else:
            answer = REFUSAL
            llm_s = 0.0

        low = answer.lower()
        hit = any(m.lower() in low for m in q["must"])
        passed += hit
        print(f"[{'PASS' if hit else 'FAIL'}] {q['id']:11} ({llm_s:4.1f}s)")
        if not hit:
            print(f"         chahiye : {q['note']}")
            print(f"         EN q    : {english[:100]}")
            print(f"         mila    : {answer[:130]}")

    print(f"\n  SCORE: {passed}/{len(questions)}")


if __name__ == "__main__":
    main()

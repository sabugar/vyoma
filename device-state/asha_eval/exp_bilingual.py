"""Does showing the model the original Hindi alongside the English recover
what the translation drops?

ASR emits no punctuation, so NMT receives a run-on and silently drops trailing
clauses - "...आशा को क्या करना चाहिए" survives ASR but vanishes from the English,
leaving the model a statement with no question in it. qwen3 is multilingual, so
the Hindi is included as a second view of the same question rather than trying
to repair the translation.
"""
import base64, json, sys, requests
sys.path.insert(0, "/home/ubuntu/pocket-infer-sw/python")
from pocketinfer.applications.hear_the_world import HearTheWorld

BH="http://localhost:11400"; OLLAMA="http://127.0.0.1:11434/api/generate"; MODEL="qwen3:1.7b"
HEAD=('Answer the question using ONLY the CONTEXT below. '
 'Output ONLY the final answer, in 1-2 complete sentences (max 35 words), '
 'including the specific relevant detail(s) from the context (numbers, steps, signs, timing) '
 'so the answer sounds informed, not vague. '
 'Do not explain your reasoning, do not describe the context, do not say '
 '"let me check" or "the text says" or similar - just state the answer directly, '
 'as if you already knew it. '
 'If the context does not contain the answer, output exactly: '
 '"I do not have this information in my training materials."\n\n')

app = HearTheWorld.__new__(HearTheWorld)
app.knowledge_chunks = app._load_knowledge_chunks()
qs = json.load(open("questions.json"))

def ask(prompt):
    r=requests.post(OLLAMA, json={"model":MODEL,"prompt":prompt+"\n<think>\n\n</think>\n\n",
      "raw":True,"stream":False,"keep_alive":-1,
      "options":{"num_predict":110,"num_ctx":2048,"temperature":0.2}}, timeout=300).json()
    return r.get("response","").strip()

for mode in ("english only", "english + hindi"):
    score=0
    for q in qs:
        tts=requests.post(f"{BH}/tts",json={"text":q["hi"],"language":"hi"},timeout=240).json()
        asr=requests.post(f"{BH}/asr",json={"audio_base64":tts["audio_base64"],"language":"hi"},timeout=240).json()
        hi=asr["text"]
        en=requests.post(f"{BH}/nmt",json={"text":hi,"src_lang":"hi","tgt_lang":"EN"},timeout=240).json()["translated_text"]
        ctx=app._retrieve_context(en, top_k=3)
        if not ctx:
            ans="I do not have this information in my training materials."
        else:
            body=f'CONTEXT:\n{chr(10).join(ctx)}\n\n'
            if mode=="english + hindi":
                body+=f'QUESTION (Hindi, as spoken): {hi}\nQUESTION (English translation, may be incomplete): {en}\n\nANSWER:'
            else:
                body+=f'QUESTION: {en}\n\nANSWER:'
            ans=ask(HEAD+body)
        hit=any(m.lower() in ans.lower() for m in q["must"])
        score+=hit
        if not hit: print(f"   [{mode}] FAIL {q['id']}: {ans[:110]}")
    print(f"=> {mode}: {score}/{len(qs)}\n")

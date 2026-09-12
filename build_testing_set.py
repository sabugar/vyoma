#!/usr/bin/env python3
"""Run every question of the page-by-page set through the live device.

The testing sheet prints what the device actually said, in Hindi, so the two
of us are reading the same thing the jury will hear. Nothing here is typed
from memory: re-run this and the sheet is rebuilt from the run.
"""
import json, os, sys, time, urllib.request, logging
sys.path.insert(0, '/home/ubuntu/pocket-infer-sw/python')
logging.disable(logging.CRITICAL)
from pocketinfer.applications.hear_the_world import HearTheWorld as H
from pocketinfer.models.ollama import Ollama as O

B = "http://127.0.0.1:11400"
def post(u, p, t=300):
    r = urllib.request.Request(u, data=json.dumps(p).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.load(x)

class Bare(H):
    def __init__(self): self.settings = {'output_language': 'hi'}
app = Bare()
app.knowledge_chunks = H._load_knowledge_chunks(app)

# The loader does not keep the page each passage came from; rebuild it the
# same way the loader walks the directory, with the same form pages skipped.
D = '/home/ubuntu/asha_knowledge/chunks'
pages = []
for f in sorted(os.listdir(D)):
    if not f.endswith('.txt'):
        continue
    t = H._repair_extraction(open(os.path.join(D, f), encoding='utf-8').read())
    if H._is_form(t):
        continue
    pages.append(f[5:8].lstrip('0'))
page_of = {c: p for c, p in zip(app.knowledge_chunks, pages)}

Q = json.load(open('/home/ubuntu/asha_submission/test_questions.json',
                   encoding='utf-8'))
out = []
for n, q in enumerate(Q, 1):
    rec = dict(q)
    t0 = time.time()
    en = post(B + "/nmt", {"text": H._prepare_hindi_for_translation(q['hi']),
                           "src_lang": "hi", "tgt_lang": "EN"})["translated_text"]
    ctx = app._retrieve_context(en, top_k=3)
    rec['en'] = en
    rec['pages'] = [page_of.get(c, '?') for c in ctx]
    if not ctx:
        rec.update(answered=False, ans_en='', ans_hi='',
                   secs=round(time.time() - t0, 1))
    else:
        r = post("http://127.0.0.1:11434/api/generate", {
            "model": "qwen3:1.7b",
            "prompt": H.build_prompt("\n".join(ctx), en) + "\n<think>\n\n</think>\n\n",
            "raw": True, "stream": False,
            "options": {"num_predict": O.NUM_PREDICT, "num_ctx": O.NUM_CTX,
                        "temperature": O.TEMPERATURE}}).get("response", "").strip()
        if H._is_no_answer(r):
            rec.update(answered=False, ans_en='', ans_hi='',
                       secs=round(time.time() - t0, 1))
        else:
            a = H._strip_restatement(H._strip_lead_in(r), en)
            hi = H._name_in_hindi(post(B + "/nmt", {
                "text": H._disambiguate_for_translation(a),
                "src_lang": "EN", "tgt_lang": "hi"})["translated_text"])
            rec.update(answered=True, ans_en=a, ans_hi=hi,
                       secs=round(time.time() - t0, 1))
    out.append(rec)
    print("  %2d/%d p%-3s %s" % (n, len(Q), rec['page'],
                                 "refused" if not rec['answered'] else rec['ans_hi'][:46]),
          flush=True)
json.dump(out, open('/home/ubuntu/asha_submission/testing.json', 'w'),
          ensure_ascii=False, indent=1)

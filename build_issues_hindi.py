#!/usr/bin/env python3
"""Translate issues.html into Hindi with the Gemini API, one block at a time.

The English document is the source of record; this produces issues-hi.html from
it so the two cannot drift. Blocks are translated individually and each result
is accepted only if its HTML tag sequence is identical to the original, which is
what stops a translation model from quietly restructuring the page. Results are
cached on disk, so a re-run after a rate limit resumes rather than restarts.

Free-tier quota is per model per day, and it is not uniform: most Gemini flash
models allow 20 requests a day, while gemini-3.5-flash-lite and
gemini-3.1-flash-lite allow 500. This document is 34 blocks, so it only fits in
the larger allowance.

The API key is read from the environment and is never written to this file or
to the repository:

    GEMINI_API_KEY=... python3 build_issues_hindi.py
"""
import hashlib, json, os, re, sys, time, urllib.request, pathlib

KEY = os.environ.get("GEMINI_API_KEY")
if not KEY:
    sys.exit("set GEMINI_API_KEY in the environment")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       + MODEL + ":generateContent")
HERE = pathlib.Path(__file__).parent
CACHE = HERE / ".issues-hi-cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

SYSTEM = """You are translating a technical engineering report from English into Hindi.
It was written by an Indian engineering team for a Government of India jury (BHASHINI /
Ministry of Electronics and IT). It describes bugs found while building an offline Hindi
voice assistant, and it is deliberately plain and unmarketed.

RULES

1. Return ONLY the translated HTML fragment. No commentary, no code fences.
2. Preserve every HTML tag, attribute, class and entity EXACTLY, in the same order.
   Translate only the human-readable text between tags. Never add or remove a tag.
3. Do not translate and do not alter: numbers, units, percentages; model, product and
   organisation names (qwen3:1.7b, Piper, MMS VITS, IndicF5, IndicTrans2, IndicConformer,
   AI4Bharat, BHASHINI, Jetson Orin Nano, JetPack, ONNX, CTC, CUDA, HarfBuzz, WeasyPrint);
   licence names; any Devanagari already present, which is example data; and English words
   quoted as evidence of what a translator produced.
4. Natural technical Hindi, the register of a government technical report - not word-by-word,
   not Sanskritised. Keep English terms Indian engineers normally use in English.
5. Keep the honest, understated tone. Do not add adjectives and do not soften a failure.
6. Headings: short Hindi noun phrases."""

def call(fragment, tries=6):
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": fragment}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 16384},
    }).encode()
    delay = 20
    for n in range(tries):
        try:
            req = urllib.request.Request(URL, data=body, headers={
                "Content-Type": "application/json", "x-goog-api-key": KEY})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            parts = [q for q in d["candidates"][0]["content"]["parts"]
                     if "text" in q and not q.get("thought")]
            out = parts[-1]["text"].strip()
            return re.sub(r'^```(?:html)?\s*|\s*```$', '', out).strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and n < tries - 1:
                print("   %d, waiting %ds" % (e.code, delay))
                time.sleep(delay); delay = min(delay * 2, 180)
            else:
                raise

def tags(h):
    return [t.lower() for t in re.findall(r'<\s*/?\s*([a-z0-9]+)', h, re.I)]

src = (HERE / "issues.html").read_text(encoding="utf-8")
i = src.index('<div class="doc">')
head, body = src[:i], src[i:]
pat = (r'(<h2>.*?</h2>|<div class="i.*?</div>|<table>.*?</table>|'
       r'<div class="note">.*?</div>|<p class="lede">.*?</p>|'
       r'<header class="masthead">.*?</header>)')
blocks = re.findall(pat, body, re.S)

out, english = [], []
for n, b in enumerate(blocks, 1):
    key = hashlib.sha256(b.encode()).hexdigest()
    if key in cache:
        hi = cache[key]
    else:
        hi = call(b)
        if tags(b) != tags(hi):
            hi = call("The previous output changed the tag structure. Translate again, "
                      "preserving every tag exactly:\n\n" + b)
        if tags(b) == tags(hi):
            cache[key] = hi
            CACHE.write_text(json.dumps(cache, ensure_ascii=False))
        time.sleep(5)
    if tags(b) != tags(hi):
        english.append(n); hi = b
    out.append(hi)
    print("  %2d/%d %s" % (n, len(blocks), "ok" if n not in english else "LEFT IN ENGLISH"))

result = head + body
for en, hi in zip(blocks, out):
    result = result.replace(en, hi, 1)
# Fixed sets, made consistent here rather than left to vary block by block.
for en, hi in (("Problem", "समस्या"), ("Cause", "कारण"),
               ("Fix", "समाधान"), ("Result", "परिणाम")):
    result = re.sub(r'(<b class="k[^"]*">)\s*%s\s*(</b>)' % en, r'\1%s\2' % hi, result)
for en, hi in (("four changes", "चार बदलाव"), ("five issues", "पाँच समस्याएँ"),
               ("four issues", "चार समस्याएँ"), ("three issues", "तीन समस्याएँ")):
    result = re.sub(r'(<span class="cnt">)\s*%s\s*(</span>)' % en, r'\1%s\2' % hi, result)
result = result.replace("<title>Engineering Issues and Fixes</title>",
                        "<title>इंजीनियरिंग समस्याएँ और समाधान</title>")
(HERE / "issues-hi.html").write_text(result, encoding="utf-8")
print("\n  wrote issues-hi.html ; blocks left in English: %s" % (english or "none"))

#!/usr/bin/env python3
"""Rebuild the ten-question demonstration sheet from demo10.json.

demo10.json is written by re-running the questions through the live device, so
the answers printed here are always the ones it actually gave, never ones typed
from memory.
"""
import html
import json
import pathlib

HERE = pathlib.Path(__file__).parent
data = json.load(open(HERE / "demo10.json", encoding="utf-8"))
css = open(HERE / "_demo_css.html", encoding="utf-8").read()

HEAD = '''<div class="doc">
  <header class="masthead">
    <div><h1>Ten Critical Questions</h1>
    <p class="sub">ASHA Sahayak &mdash; verified demonstration set</p></div>
    <div class="spec"><b>Every answer below was produced by the device</b><br>
    Jetson Orin Nano 8&nbsp;GB &middot; MAXN_SUPER &middot; 11 Sep 2026<br>
    <b>no network at any point</b></div>
  </header>
  <p class="lede">
    These ten were not chosen for how well they read. They were chosen by running twenty-six
    candidate questions through the device and keeping only those whose answers are correct and
    traceable to a page of the ASHA manual. <b>Sixteen were discarded</b> &mdash; among them a question
    about bleeding after delivery, which the device answered from the contraception page because
    Module 7 does not cover postpartum haemorrhage at all. What follows is what the device can
    actually do, not what it ought to.
  </p>'''

TAIL = '''  <div class="note">
    <p class="hd">Two things to know before demonstrating</p>
    <p><b>The translation has two rough edges in this set.</b> "Chest indrawing" comes out as
    <span style="font-family:var(--deva)">&#2331;&#2366;&#2340;&#2368; &#2350;&#2375;&#2306; &#2342;&#2352;&#2381;&#2342;</span>, which means chest <em>pain</em>, and
    "limbs limp" as <span style="font-family:var(--deva)">&#2309;&#2306;&#2327; &#2354;&#2306;&#2327;&#2396;&#2375;</span>, which suggests limping
    rather than floppy. The English is right and the retrieval is right; the Hindi wording is not,
    and it is a translation-model limit, not a retrieval one.</p>
    <p><b>Speak the question as written.</b> Speech recognition is the weakest link in the chain:
    one mis-heard syllable once turned a question about an eye infection into one about a child
    drinking, and the device answered the question it was given. One word decides the meaning.</p>
    <p><b>Recording is a toggle</b> &mdash; one press starts, the next ends it, and 3.5&nbsp;s of
    silence ends it anyway. Timings above are machine time from the stop, and exclude how long the
    question itself takes to ask.</p>
  </div>
</div>'''

parts = [css, HEAD]
for i, r in enumerate(data, 1):
    parts.append(f'''  <div class="q">
    <div class="qhead"><span class="num">{i:02d}</span><span class="title">{html.escape(r["title"])}</span></div>
    <p class="ask">{html.escape(r["hi"])}</p>
    <p class="why"><b>Why it matters</b>{html.escape(r["why"])}</p>
    <div class="ans">
      <div class="lbl">What the device says</div>
      <p class="hi">{html.escape(r["ans_hi"])}</p>
      <p class="en">{html.escape(r["ans_en"])}</p>
    </div>
    <p class="meta">manual pages {" + ".join(r["pages"])} &nbsp;&middot;&nbsp; {r["secs"]} s to answer</p>
  </div>''')
parts.append(TAIL)

out = HERE / "demo-questions.html"
out.write_text("\n".join(parts), encoding="utf-8")

from weasyprint import HTML  # noqa: E402
doc = HTML(filename=str(out)).render()
HTML(filename=str(out)).write_pdf(str(HERE / "ASHA-Sahayak-Demo-Questions.pdf"))
print(f"{len(data)} questions, {len(doc.pages)} pages")

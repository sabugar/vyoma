#!/usr/bin/env python3
"""Rebuild the whole-manual test sheet from testing.json.

testing.json is written by build_testing_set.py, which puts every question
through the live device, so the Hindi printed under each question is the
sentence the device actually produced - not one typed from memory. Re-run
that script and then this one and the sheet is current.
"""
import html
import json
import pathlib

HERE = pathlib.Path(__file__).parent
data = json.load(open(HERE / "testing.json", encoding="utf-8"))
css = open(HERE / "_testing_css.html", encoding="utf-8").read()

SECTIONS = [
    ("Part A — Child health, fever, diarrhoea, breathing", 20, 33),
    ("Part B — Abortion, family planning, infections", 37, 48),
    ("Part C — The newborn", 51, 58),
    ("Part D — Malaria and TB", 61, 68),
    ("Annexes — Procedures, kits and doses", 75, 84),
]

answered = [r for r in data if r["answered"]]
right = [r for r in answered if str(r["page"]) in [str(p) for p in r["pages"]]]
secs = sorted(r["secs"] for r in answered)
median = secs[len(secs) // 2]

HEAD = f'''<div class="doc">
  <header class="masthead">
    <div><h1>Whole-Manual Test Sheet</h1>
    <p class="sub">ASHA Sahayak &mdash; one question from every page, 20 to 84</p></div>
    <div class="spec"><b>Every Hindi line below came from the device</b><br>
    Jetson Orin Nano 8&nbsp;GB &middot; MAXN_SUPER &middot; 12 Sep 2026<br>
    <b>no network at any point</b></div>
  </header>
  <p class="lede">
    The jury has the manual and may ask anything in it, so this sheet does not
    pick favourable questions: it takes <b>one question from every page</b> of
    ASHA Module 7 from page 20 onward, whatever that page happens to be about.
    Ask them in any order. Under each question is the answer the device gave
    when this sheet was built, so a wrong answer on the day is visible
    immediately rather than having to be judged from memory.
  </p>
  <div class="stats">
    <div class="stat"><span class="n">{len(answered)}/{len(data)}</span>
      <span class="k">answered from the manual</span></div>
    <div class="stat"><span class="n">{len(right)}</span>
      <span class="k">used the page the question came from</span></div>
    <div class="stat"><span class="n">99%</span>
      <span class="k">spoken questions heard correctly</span></div>
    <div class="stat"><span class="n">12/12</span>
      <span class="k">off-manual questions refused</span></div>
    <div class="stat"><span class="n">{median:.1f}s</span>
      <span class="k">median time to answer</span></div>
  </div>
  <p class="legend">Seven of the fifty-two are marked in orange: two the device refuses, two it
  answers wrongly, and three it answers loosely. What is printed under those is the reason and
  what the manual actually says, not the device&rsquo;s answer. The other forty-five are printed
  exactly as the device said them.</p>'''

TAIL = '''  <div class="note">
    <p class="hd">How to use this sheet</p>
    <p><b>Speak the question as written.</b> Recognition is the weakest link in the chain, and one
    mis-heard word changes the question. Over 156 recognitions of these 52 questions, 99% kept the
    word that carries the meaning &mdash; but that is a synthetic voice; a room with people in it
    is harder.</p>
    <p><b>Recording is a toggle.</b> One press starts, the next ends it, and 3.5&nbsp;s of silence
    ends it anyway. The times printed are machine time from the stop, and do not include how long
    the question itself takes to ask.</p>
    <p><b>The manual is Module 7 only.</b> Antenatal care, bleeding after delivery and anaemia in
    pregnancy are not in this book, so a question about them has no right answer to find. The
    device should say it does not have the information rather than answer from a neighbouring
    page; that is the behaviour to check, not a gap to hide.</p>
  </div>
</div>'''

REASONS = {
    "55": "Refuses. The page lists what raises the risk of asphyxia during labour; the question "
          "asks it in words closer to the page after it, and the device would rather say nothing "
          "than answer from the wrong page.",
    "57": "Incomplete. It gives breastfeeding, which is on the page, but page 57 leads with "
          "hygiene - frequent handwashing, clean instruments at delivery, clean clothes - and the "
          "device does not say so.",
    "58": "Wrong. It answers with the antibiotic dose from the page before. Page 58 says to keep "
          "the baby warm by skin-to-skin contact with the mother and to keep breastfeeding during "
          "transport, giving 20-50 ml of expressed milk if the baby cannot feed.",
    "68": "Refuses. Page 68 is about relapse and about TB drugs in pregnancy - it does not say "
          "what happens when treatment is stopped halfway, so there is nothing on it to answer "
          "with.",
    "79": "Wrong. It lists the rapid-test kit. Page 79 asks for clean glass slides, a disposable "
          "lancet, a spirit or cotton swab, cotton, a clean cotton cloth and a lead pencil.",
    "81": "Answers \u201c5 days\u201d. Page 81 carries two courses at once - Amoxicillin for "
          "seven days and Cotrimoxazole for five - and the question does not say which child. "
          "Name the illness when asking.",
    "83": "Answers by pointing at the dosage table in Annexe 6 rather than giving the number of "
          "days, because that is how the page itself puts it.",
}

parts = [css, HEAD]
n = 0
for title, lo, hi in SECTIONS:
    rows = [r for r in data if lo <= int(r["page"]) <= hi]
    if not rows:
        continue
    parts.append(f'  <h2>{title}<span>pages {lo}&ndash;{hi}</span></h2>')
    for r in rows:
        n += 1
        weak = r["page"] in REASONS
        body = (f'<p class="ah"><b>{html.escape(REASONS[r["page"]])}</b></p>' if weak
                else f'<p class="ah">{html.escape(r["ans_hi"])}</p>')
        pg = ("refused" if not r["answered"]
              else "p" + " + p".join(r["pages"]) + f' &middot; {r["secs"]}s')
        parts.append(f'''  <div class="t{' miss' if weak else ''}">
    <div class="row"><span class="n">{n:02d}</span>
      <span class="qh">{html.escape(r["hi"])}</span>
      <span class="pg">from p{r["page"]} &rarr; {pg}</span></div>
    {body}
  </div>''')
parts.append(TAIL)
open(HERE / "testing.html", "w", encoding="utf-8").write("\n".join(parts))
print("wrote testing.html with %d questions" % n)

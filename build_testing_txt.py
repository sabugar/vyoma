#!/usr/bin/env python3
"""Write the test sheet as plain text as well as PDF.

The PDF renders Devanagari correctly but its text layer does not survive
copying: WeasyPrint loses the mapping back to the original characters
wherever a vowel sign is reordered, so चाहिए copies as "चा हए" and यदि as
"द". Every Devanagari font on this device behaves the same way, so this is
the renderer, not the font. The sheet is therefore also written as UTF-8
text, which is what to copy from.
"""
import json, pathlib

HERE = pathlib.Path(__file__).parent
data = json.load(open(HERE / "testing.json", encoding="utf-8"))
SECTIONS = [("Part A - Child health, fever, diarrhoea, breathing", 20, 33),
            ("Part B - Abortion, family planning, infections", 37, 48),
            ("Part C - The newborn", 51, 58),
            ("Part D - Malaria and TB", 61, 68),
            ("Annexes - Procedures, kits and doses", 75, 84)]
out = ["ASHA Sahayak - whole-manual test sheet",
       "One question from every page of ASHA Module 7, page 20 onward.",
       "The Hindi below is what the device actually said when this was built.",
       ""]
n = 0
for title, lo, hi in SECTIONS:
    rows = [r for r in data if lo <= int(r["page"]) <= hi]
    if not rows:
        continue
    out += ["", "=" * 72, "%s  (pages %d-%d)" % (title, lo, hi), "=" * 72]
    for r in rows:
        n += 1
        out.append("")
        out.append("%02d. [p%s] %s" % (n, r["page"], r["hi"]))
        if r["answered"]:
            out.append("    -> %s" % r["ans_hi"])
            out.append("    (%s | %s s)" % (" + p".join(["p" + p for p in r["pages"]])
                                            .replace("pp", "p"), r["secs"]))
        else:
            out.append("    -> REFUSED (device says it does not have this information)")
open(HERE / "testing.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("wrote testing.txt with %d questions" % n)

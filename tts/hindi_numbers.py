import re

# VITS is a character-level TTS model - it has no notion of "twenty", it can
# only spell out digit-by-digit (and several digits like 5/6/7/9 aren't even
# in this model's vocab, so those get silently dropped by the tokenizer).
# Numbers in NMT-translated Hindi text must be spelled out as words before
# synthesis, or numeric details (dosages, weeks, ages - exactly the kind of
# specific detail the RAG prompt is designed to surface) come out garbled or
# missing entirely.

_ONES = [
    "शून्य", "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
    "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस",
    "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास",
    "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर",
    "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी",
    "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अठासी", "नवासी", "नब्बे",
    "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]


def _under_1000(n: int) -> str:
    if n < 100:
        return _ONES[n]
    hundreds, rem = divmod(n, 100)
    word = f"{_ONES[hundreds]} सौ"
    if rem:
        word += f" {_ONES[rem]}"
    return word


def number_to_hindi_words(n: int) -> str:
    if n == 0:
        return _ONES[0]
    if n < 0:
        return f"माइनस {number_to_hindi_words(-n)}"
    if n < 1000:
        return _under_1000(n)
    if n < 100000:
        thousands, rem = divmod(n, 1000)
        word = f"{_under_1000(thousands)} हज़ार"
        if rem:
            word += f" {_under_1000(rem)}"
        return word
    # Beyond lakhs is not expected in this content (dosages/weeks/ages) -
    # fall back to digit-by-digit rather than guessing lakh/crore grouping.
    return " ".join(_ONES[int(d)] for d in str(n))


_NUMBER_RE = re.compile(r"\d+")


def spell_out_numbers(text: str) -> str:
    return _NUMBER_RE.sub(lambda m: number_to_hindi_words(int(m.group())), text)

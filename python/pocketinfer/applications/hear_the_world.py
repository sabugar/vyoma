import base64
from pocketinfer.applications.base import BaseApplication
from pocketinfer.applications.registry import RegisterApplication

from pocketinfer.models.ollama import Ollama
from pocketinfer.models.piper import Piper
from pocketinfer.models.vosk import Vosk
from pocketinfer.models.asr import Asr
from pocketinfer.models.nmt import Nmt
from pocketinfer.models.tts import Tts

from pocketinfer.audio import AudioPlayer

from io import BytesIO
from subprocess import check_output

import logging
import time
import wave
import os
import json
import queue
import threading
import sys
import threading
import re
import gc
import ctypes

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


# Register this class as an application that can run on the Pocket Infer Device
# The argument here is a dictionary of metadata about the application
# Metadata will be used to instantiate the application and ensure dependencies are met
@RegisterApplication({
    "name": "Hear The World",
    "description": "An application that allows the user to ask questions about their surroundings.",
    "author": "PocketInfer",
    "version": "0.1.0",
    "models": {
        # qwen3:1.7b - the text-only sibling of qwen3-vl:2b. Same model family
        # and reasoning behaviour (it is likewise a thinking model, so it still
        # needs the raw+<think>-prefill handling in models/ollama.py), but with
        # no CLIP tower: this app dropped the camera, and the vision half was
        # costing a 322MB reserved compute buffer plus the projector weights
        # for a code path that could never run.
        #
        # Model choice here is a safety decision, not just a memory one.
        # Measured with the correct rule ("Do not bathe a baby until its weight
        # is 2000 gm") present verbatim in the retrieved context:
        #   qwen2.5:1.5b  817MB -> "the baby should be bathed daily until its
        #                           weight reaches 2000 grams"  NEGATION REVERSED
        #   qwen2.5:3b   1684MB -> "I do not have this information"  (safe, wrong)
        #   qwen3-vl:2b  3084MB -> correct, but rambles and does not fit
        #   qwen3:1.7b    921MB -> correct, concise            <-- chosen
        "ollama": {"model_name": "qwen3:1.7b"},
        # "ollama": {"model_name": "qwen3-vl:2b"},   # vision-capable, +2.2GB
        # "ollama": {"model_name": "qwen2.5:1.5b"},  # unsafe, see above
        # "ollama": {"model_name": "moondream:1.8B"},
        # "ollama": {"model_name": "ministral-3:3B"},
        "piper": {"voice_name": "en_US-lessac-medium"},
        "vosk": {"model_name": "vosk-model-small-en-us-0.15"},
        "asr": {},
        "nmt": {},
        "tts": {},
    },
    "default_settings": {
        "input_language": "en",
        "output_language": "en",
    },
    "service_dependencies": ["ollama", "bashini_models"],
})
class HearTheWorld(BaseApplication):
    def start(self):
        # Load any models or resources needed for the application
        # Piper (TTS) and Vosk (ASR) are only used on the English path. In hi/hi
        # mode Bhashini handles both, but loading them anyway costs ~500MB of
        # onnxruntime/CUDA context - enough to push qwen3-vl off the GPU
        # (17%/83% -> 56%/44% CPU/GPU) and drop it from 28 tok/s to 0.45 tok/s.
        self.piper = None
        self.vosk = None
        if self.settings["input_language"] == "en" or self.settings["output_language"] == "en":
            self.piper = Piper(voice_name=self.METADATA["models"]["piper"]["voice_name"],
                               audio_device=self.board.alsa_playback_device)
            self.vosk = Vosk(model_name=self.METADATA["models"]["vosk"]["model_name"])
        else:
            self.logger.info("Non-English mode: skipping Piper/Vosk load to free memory for the LLM")
        self.ollama = Ollama(model_name=self.METADATA["models"]["ollama"]["model_name"])
        self.asr = Asr()
        self.nmt = Nmt()
        self.tts = Tts()
        self.board.subscribe_to_ui(self.ui_cb)
        # Proceed with running the application in it's own thread
        if not os.path.exists("/tmp/hear_the_world_en_logs"):
            os.makedirs("/tmp/hear_the_world_en_logs")

        # Load ASHA knowledge base for RAG
        self.knowledge_chunks = self._load_knowledge_chunks()
        self.logger.info(f"Loaded {len(self.knowledge_chunks)} knowledge chunks for RAG")

        super().start()

    # English stopwords - excluded from retrieval scoring so queries like
    # "what should I do" don't match every chunk on filler words.
    STOPWORDS = set((
        'a an the is are was were am be been being i you he she it we they me him her us them '
        'my your his its our their mine yours hers ours theirs this that these those what which '
        'who whom when where why how do does did doing done have has had having can could should '
        'would may might must shall will of at by for with about into through during before after '
        'above below to from up down in out on off over under again further then once here there '
        'all any both each few more most other some such no nor not only own same so than too '
        'very just and but if or because as until while also'
    ).split())

    # British spellings used in the ASHA module -> American, so NMT output
    # (usually American) matches book text (usually British).
    SPELLING_MAP = {
        'diarrhoea': 'diarrhea', 'anaemia': 'anemia', 'oedema': 'edema',
        'haemoglobin': 'hemoglobin', 'labour': 'labor', 'behaviour': 'behavior',
        'centre': 'center', 'litre': 'liter', 'paediatric': 'pediatric',
        'foetus': 'fetus', 'programme': 'program', 'counselling': 'counseling',
    }

    # Suffixes stripped before matching, longest first so "-ing" is not left
    # as a stray "-g". Deliberately small - this is inflection only, not a
    # full stemmer, because aggressive stemming collapses distinct clinical
    # terms.
    SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "ers", "er", "ed", "es", "s")

    @staticmethod
    def _terms(text):
        '''Normalize text into stemmed content-word terms for retrieval.'''
        terms = []
        for w in re.findall(r'[a-z]+', text.lower()):
            w = HearTheWorld.SPELLING_MAP.get(w, w)
            if w in HearTheWorld.STOPWORDS or len(w) < 3:
                continue
            # Strip one inflectional ending, then cap the length. The previous
            # version only truncated to 6 characters, which silently failed
            # whenever a 5-letter root gained a suffix: the question's
            # "weighed" became "weighe" while the manual's "weigh" stayed
            # "weigh", so "how often should children be weighed" could never
            # match "weigh every child monthly" - the one page that answers it.
            for suffix in HearTheWorld.SUFFIXES:
                if len(w) - len(suffix) >= 4 and w.endswith(suffix):
                    w = w[:-len(suffix)]
                    break
            # A silent trailing "e" is the other half of the same problem:
            # "bathed" strips to "bath" but "bathe" would stay "bathe".
            if len(w) > 4 and w.endswith("e"):
                w = w[:-1]
            # 5, not 6: "children" truncates to "child" and meets the manual's
            # "child", which a 6-character cut ("childr") never would.
            terms.append(w[:5] if len(w) > 5 else w)
        return terms

    # The PDF the knowledge base was extracted from renders some headings and
    # emphasised runs twice, one character apart, so the text came out as
    # "ddiiaarrrrhhooeeaa" and "WWaarrnniinngg ssiiggnnss". 79 words across 8 of
    # the 76 pages. They are invisible to search - a question about diarrhoea
    # never matched that page - and meaningless to the model when they do reach
    # it as context, so they are repaired on load rather than left in place.
    @staticmethod
    def _undouble(word):
        """Collapse "ddiiaarrrrhhooeeaa" to "diarrhoea"; leave anything else."""
        if (len(word) >= 6 and len(word) % 2 == 0
                and all(word[i].lower() == word[i + 1].lower()
                        for i in range(0, len(word), 2))):
            return "".join(word[i] for i in range(0, len(word), 2))
        return word

    # Two more artefacts from the same extraction, both audited across all 76
    # pages. The running head "ASHA Module 7" is set vertically in the margin
    # and comes out reversed, as "7 eludoM AHSA", on 78 occurrences; and the
    # bullet glyph in the danger-sign lists comes out as a bare letter u, so
    # the manual's "u Any child who is underweight" reaches the model looking
    # like a typo. Neither reaches the index - the stemmer drops one-letter
    # tokens and no question produces "eludom" - but both are noise in the
    # context the model reads, and the second one sits in exactly the danger
    # sign lists that matter most.
    _REVERSED_HEAD = re.compile(r'\d*\s*eludoM\s+AHSA\s*', re.I)
    _U_BULLET = re.compile(r'(?<=[\s])u(?=\s+[A-Z])')

    # Four pages came out reversed end to end - the home-visit checklist forms,
    # which are set rotated in the PDF, so the extractor walked them backwards:
    # "fo noitanimaxE( mrof tisiv emoH" for "Home visit form (Examination of".
    # 794 words of newborn follow-up guidance that no question could reach and
    # the model could not read. Reversing the characters restores both the
    # spelling and the word order, because both were reversed together.
    _COMMON = frozenset("the and for with that this from have been are was not "
                        "you your will can has".split())

    @classmethod
    def _looks_reversed(cls, text):
        words = re.findall(r'[a-z]{2,}', text.lower())
        if len(words) < 25:
            return False
        fwd = sum(1 for w in words if w in cls._COMMON)
        rev = sum(1 for w in re.findall(r'[a-z]{2,}', text[::-1].lower())
                  if w in cls._COMMON)
        # Needs to be both better backwards and convincingly so, or a page of
        # tables and numbers could flip itself on a one-word margin.
        return rev > fwd and rev >= 5

    # The last four pages of the module are the supervisor's home-visit
    # checklists - forms to tick, not guidance to read. Once un-reversed they
    # started competing in retrieval, and because their text is a flattened
    # table ("All limbs limp ... Yes/No Yes/No Yes/No") they crowded the clean
    # sign list on page 57 out of the context: the sepsis answer lost two of
    # its five signs and the eval went 9/10 to 8/10. They are the only pages in
    # the manual carrying repeated Yes/No boxes, which is what identifies them.
    @staticmethod
    def _is_form(text):
        return len(re.findall(r'Yes\s*/\s*No', text, re.I)) >= 5

    # Transcription slips, not vocabulary corrections. The recogniser sometimes
    # emits a character twice - नाभभी for नाभी, कौनसीी for कौनसी - or stacks two
    # vowel signs on one consonant, नाभीि, which is not writable Hindi at all.
    # The cost is real: नाभभी translated to "the navel is peeling off" and the
    # pus vanished from the question entirely, so the answer came back about
    # something else. Collapsed to one character, the same sentence translates
    # to "the navel has turned red and pus is coming out of it".
    #
    # This deliberately holds no word list and cannot turn one word into a
    # different word. It only removes a repeat the writing system does not
    # allow, which is the same repair the manual's own text needed.
    _DOUBLED_CHAR = re.compile(r'([\u0915-\u0939\u093e-\u094c\u0902\u0903])\1')
    _STACKED_MATRA = re.compile(r'([\u093e-\u094c])[\u093e-\u094c]+')

    # Measured weakness of this recogniser, not of the speaker. Putting one
    # question per topic through TTS and back through ASR, 23 of 30 survived
    # intact; every one of the seven that did not failed the same way - the
    # model does not separate consonants that are articulated in the same
    # place:
    #
    #     दवा -> दबा ("press the baby")      पसली -> फसली ("crop")
    #     सेप्सिस -> सेब्सिस                  नाभि -> ना भी ("nose")
    #     पीप -> पी पा ("drinking")          कॉपर टी -> प्रॉपर्टी ("property")
    #
    # So the terms below are matched phonetically rather than literally: each
    # consonant is folded to its place of articulation before comparing, which
    # is exactly the distinction the recogniser is losing. The list holds only
    # words this manual actually uses, and a term is replaced solely when the
    # transcript word is not already Hindi the list knows.
    _ASR_TERMS = (
        'नाभि', 'पीप', 'मवाद', 'दवा', 'पसली', 'सेप्सिस', 'खसरा', 'खसरे',
        'टीका', 'टीके', 'दस्त', 'बुखार', 'वजन', 'कुपोषित', 'मलेरिया',
        'गर्भपात', 'नसबंदी', 'दौरे', 'प्रसव', 'रक्तस्राव', 'पीलिया',
        'स्तनपान', 'नवजात', 'निमोनिया', 'टीबी', 'ऐंठन', 'सुस्ती',
        # A second pass of the probe turned up these: कॉपर came back as
        # "कोपरती", गोली as "बोली", मरीज as "वरीज".
        'कॉपर', 'गोली', 'मरीज', 'गर्दन', 'निर्जलीकरण',
    )

    # Only the confusions actually observed, and only between sounds made in
    # the same place. Folding more than this (velars together, or र with ल)
    # made unrelated words collide: दूध became दस्त, "milk" became "diarrhoea",
    # and खतरे became खसरे, "danger" became "measles". A correction that can do
    # that is worse than the fault it fixes.
    _CONFUSABLE = (
        set('पफबभवम'),     # दवा/दबा, पसली/फसली, सेप्सिस/सेब्सिस, मरीज/वरीज
        set('टठतथ'),        # टीका/तीका
        set('डढदध'),
        set('सशष'),
        # Short and long forms of the same vowel. नाभि comes back as नाभी even
        # when the consonants survive, and the two spellings are the same word.
        set('िी'),
        set('ुू'),
        # द with भ is not a shared place of articulation - one is dental, the
        # other labial - so this pair is here on evidence rather than on
        # phonetics. नाभी came back as नादी, and checked against every word in
        # the twenty-eight sentences known to transcribe correctly, plus नदी,
        # दादी, नाड़ी, भाभी, दूध and दाई, it replaces none of them.
        set('दभ'),
    )

    @classmethod
    def _sub_cost(cls, a, b):
        if a == b:
            return 0.0
        for group in cls._CONFUSABLE:
            if a in group and b in group:
                return 0.4
        return 1.0

    @classmethod
    def _weighted_distance(cls, a, b):
        """Edit distance where a confusable substitution barely counts."""
        if abs(len(a) - len(b)) > 1:
            return 99.0
        prev = [float(j) for j in range(len(b) + 1)]
        for i, ca in enumerate(a, 1):
            cur = [float(i)]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1.0, cur[j - 1] + 1.0,
                               prev[j - 1] + cls._sub_cost(ca, cb)))
            prev = cur
        return prev[-1]

    # The recogniser writes the same sound two ways - सफ़ेद for सफेद, जाँच for
    # जांच, सांस for साँस. These are spellings, not mishearings, but they break
    # a literal comparison, so they are levelled before anything is matched.
    _SPELLING = str.maketrans({'\u093c': None, '\u0901': '\u0902'})

    @classmethod
    def _level_spelling(cls, text):
        return (text or '').translate(cls._SPELLING)

    @classmethod
    def _strip_marks(cls, word):
        return re.sub(r'[\u093c\u094d]', '', word)

    @classmethod
    def _match_term(cls, token, limit=0.8):
        """The clinical term this token is a mangling of, or None."""
        tok = cls._strip_marks(token)
        if len(tok) < 3:
            return None
        best, best_d = None, 99.0
        for term in cls._ASR_TERMS:
            cand = cls._strip_marks(term)
            # The opening sound anchors the match; without it, short words
            # wander into each other.
            if cls._sub_cost(tok[0], cand[0]) > 0.4:
                continue
            d = cls._weighted_distance(tok, cand)
            if d < best_d:
                best, best_d = term, d
        # At most two confusable slips, and nothing else.
        return best if best_d <= limit else None

    # Both words mean pus, but only one of them survives translation. Asked
    # about पीप at the navel, the translator returned "the navel is peeling"
    # three times out of four and "peeking" once; the same sentences with मवाद
    # returned "pus" every time. The question is not changed - it is written in
    # the synonym the translator actually knows.
    _HI_FOR_TRANSLATION = (('पीप', 'मवाद'),)

    @classmethod
    def _prepare_hindi_for_translation(cls, text):
        for weak, strong in cls._HI_FOR_TRANSLATION:
            text = (text or '').replace(weak, strong)
        return text

    @classmethod
    def _correct_terms(cls, text):
        """Repair clinical words the recogniser mangled, leaving the rest alone."""
        known = set(cls._ASR_TERMS)
        tokens = (text or '').split()
        out, i = [], 0
        while i < len(tokens):
            if tokens[i] in known:
                out.append(tokens[i]); i += 1; continue
            # A word split in two - नाभि heard as "ना भी" - but only when
            # neither half is a word the list already knows.
            if (i + 1 < len(tokens) and tokens[i + 1] not in known
                    and len(cls._strip_marks(tokens[i])) <= 4):
                # A rejoined pair may carry one extra letter - पीप comes back
                # as "पी पा" - so it gets a little more room than a single
                # token, which it can afford: both halves being unknown words
                # and the first being very short is already a strong filter.
                # Checked against innocent joins (केबाद, कीकमी, कोदूध, जन्मके)
                # and none of them match at any threshold.
                hit = cls._match_term(tokens[i] + tokens[i + 1], limit=1.0)
                if hit:
                    out.append(hit); i += 2; continue
            out.append(cls._match_term(tokens[i]) or tokens[i])
            i += 1
        return ' '.join(out)

    @classmethod
    def _normalise_transcript(cls, text):
        text = text or ''
        previous = None
        while previous != text:
            previous = text
            text = cls._DOUBLED_CHAR.sub(r'\1', text)
        return cls._correct_terms(
            cls._level_spelling(cls._STACKED_MATRA.sub(r'\1', text)))

    @classmethod
    def _repair_extraction(cls, text):
        if cls._looks_reversed(text):
            text = text[::-1]
        text = re.sub(r'[A-Za-z]{6,}', lambda m: cls._undouble(m.group(0)), text)
        text = cls._REVERSED_HEAD.sub('', text)
        text = cls._U_BULLET.sub('\u2022', text)
        return text

    def _load_knowledge_chunks(self, chunks_dir='/home/ubuntu/asha_knowledge/chunks'):
        '''Load pre-chunked ASHA Module-7 text and build a TF-IDF index.'''
        self._chunk_tf = []   # per-chunk: {stem: count}
        self._chunk_len = []
        df = {}               # stem -> number of chunks containing it
        chunks = []
        if os.path.exists(chunks_dir):
            for fname in sorted(os.listdir(chunks_dir)):
                if fname.endswith('.txt'):
                    with open(os.path.join(chunks_dir, fname), 'r', encoding='utf-8') as f:
                        text = self._repair_extraction(f.read())
                    if self._is_form(text):
                        continue
                    terms = self._terms(text)
                    tf = {}
                    for t in terms:
                        tf[t] = tf.get(t, 0) + 1
                    for t in tf:
                        df[t] = df.get(t, 0) + 1
                    chunks.append(text)
                    self._chunk_tf.append(tf)
                    self._chunk_len.append(max(len(terms), 1))
        n = max(len(chunks), 1)
        import math
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        return chunks

    # A page only counts as relevant if the question's own words carry it
    # there. Accepting any score above zero meant one shared common word was
    # enough: "I want to buy a new saree" pulled in three manual pages and got
    # a confident answer about newborn care, and an ASR-mangled question
    # ("seizures" heard as "twins") did the same. Dividing the best score by
    # the number of query terms separates the two populations cleanly -
    # measured over 14 realistic questions plus the 10-question eval set,
    # in-scope scored 1.74-5.41 and out-of-scope 0.00-1.49. 1.6 sits in that
    # gap with margin on both sides; at 1.8 a real question about a cord
    # infection (1.74) starts being refused.
    # Recalibrated when the scope test moved from "one page carries the
    # question" to "the manual covers the subject" - the numbers are on a
    # different scale. Measured over fifteen real questions and twelve
    # off-topic ones: in-scope 3.65-6.09 with a single mangled outlier at 2.34,
    # off-topic 0.00-2.56. 3.0 sits in that band; anywhere from 2.6 to 3.4
    # gives the same answer on this set, so the exact figure is not delicate.
    MIN_RELEVANCE = 3.0

    # What the model is told to emit when the retrieved pages do not answer the
    # question. Caught before anything is spoken, so the user hears the proper
    # refusal rather than this marker.
    NO_ANSWER_SENTINEL = "NO ANSWER IN CONTEXT"

    # The prompt asks for the answer directly, and mostly gets it, but the
    # model still sometimes opens with a stock lead-in. It survives translation
    # as "इसका उत्तर हैः", which is the first thing the ASHA hears and says
    # nothing. Stripped here rather than fought for in the prompt, because it
    # is cheap and certain.
    _LEAD_IN = re.compile(
        r'^\s*(the\s+answer\s+is|answer|according to the (context|guidance|text)|'
        r'based on the (context|text))\s*[:,]?\s*', re.I)

    # Telling the model not to repeat the question back does not stop it - it
    # still opens with a restatement in about one answer in four. That is not
    # merely untidy: spoken Hindi runs about 0.08s per character through this
    # voice, so an 85-character echo of the question is seven seconds of audio
    # that says nothing, before the ASHA hears a word of the answer. Dropped
    # here when the opening sentence is mostly the question's own words and
    # there is a real answer behind it. See _is_restatement below.

    # English "drink" with no object is ambiguous, and the translator resolves
    # it to alcohol often enough to matter: "sunken eyes, poor drinking
    # ability, drinking poorly" came out as "खराब शराब पीना" - drinking
    # alcohol badly - as a sign of dehydration in a child. Giving the verb an
    # object fixes it ("खराब तरल पदार्थ पीना"). Probed the rest of the domain's
    # ambiguous words the same way - stool, labour, discharge, delivery,
    # passing, water breaking - and they all resolve correctly in context, so
    # this stays a single narrow substitution rather than a glossary.
    # Narrow on purpose: only where the verb is qualified by an adverb and so
    # stands without an object, which is the construction that goes wrong.
    # Substituting on every bare "drink" also works but makes ordinary lines
    # clumsy - "give the child ORS to drink" becomes "ORS to drink fluids".
    _BARE_DRINK = re.compile(
        r'\b(drink|drinks|drinking)\b(?=\s+(?:poorly|badly|well|less|eagerly))',
        re.I)

    @classmethod
    def _disambiguate_for_translation(cls, text):
        return cls._BARE_DRINK.sub(lambda m: m.group(1) + ' fluids', text or '')

    @classmethod
    def _is_restatement(cls, sentence, query):
        """True when this sentence just echoes the question back.

        A sentence carrying a figure is never an echo. The model frequently
        restates the situation and gives the rule in one breath - "the baby is
        very small in weight, so do not bathe it until its weight is 2000 gm" -
        and treating that as an echo deletes the only number in the answer.
        """
        if re.search(r'\d', sentence or ''):
            return False
        first = set(cls._terms(sentence))
        q_terms = set(cls._terms(query))
        if not first or not q_terms:
            return False
        return len(first & q_terms) / len(first) >= 0.7

    @classmethod
    def _strip_restatement(cls, answer, query):
        parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', answer or '') if p.strip()]
        if len(parts) < 2:
            return answer
        return " ".join(parts[1:]) if cls._is_restatement(parts[0], query) else answer

    @classmethod
    def _strip_lead_in(cls, text):
        cleaned = cls._LEAD_IN.sub('', text or '', count=1).lstrip()
        # Only accept the strip if something is actually left to say.
        if len(cleaned) <= 20:
            return text or ''
        # Removing "According to the context," leaves a lower-case opening.
        return cleaned[0].upper() + cleaned[1:]

    @classmethod
    def build_prompt(cls, context, query):
        """The one place the RAG prompt is written.

        asha_eval/run_eval.py used to keep its own copy, which drifted: it was
        still scoring the old wording, and at a different num_predict, so the
        eval was reporting 9/10 for a prompt the device no longer used. Both
        call this now, so a prompt change is measured by the next eval run
        rather than going unnoticed.

        The wording asking for 1-2 sentences is load-bearing and the obvious
        addition - telling the model not to repeat the question - is
        deliberately absent. Measured over the nine answerable eval questions
        with the full context: this wording scores 8/9; adding "do not repeat
        the question back" drops it to 7/9, and so does widening the budget to
        1-3 sentences. Both make the model reach for a general action instead
        of the manual's own number, so the bathing answer stops saying 2000 gm.
        The echo is removed after generation instead, by _strip_restatement.

        The refusal rule comes first on purpose. It used to sit at the end,
        after an instruction to make the answer "sound informed, not vague" -
        two orders pulling opposite ways, and the model followed the one that
        told it to answer.
        """
        return (
            f'First decide: does the CONTEXT below actually contain the answer '
            f'to the QUESTION?\n'
            f'If it does NOT, output exactly: {cls.NO_ANSWER_SENTINEL}\n'
            f'Only if it DOES, answer in 1-2 complete sentences (max 35 words) '
            f'using the specific details from the context - numbers, steps, '
            f'signs, timing. State the answer directly, as if you already knew '
            f'it; do not explain your reasoning, do not describe the context, '
            f'and do not say "let me check" or "the text says".\n\n'
            f'CONTEXT:\n{context}\n\n'
            f'QUESTION: {query}\n\n'
            f'ANSWER:'
        )

    # Fixed replies are written in the output language, not translated. They
    # are the same every time, so paying NMT for them only adds latency and a
    # chance of mangling; a refusal in particular has to come out right every
    # time. Each one tells the ASHA what to do next - "I don't know" on its own
    # leaves her no better off than before she asked.
    FIXED_REPLIES = {
        # Has to fit anything that falls outside the manual, not just a
        # health question the manual happens to miss. Sending someone to a
        # health centre because they asked why it has not rained in Ahmedabad
        # reads as nonsense, so the referral is stated as a condition rather
        # than as advice, and the first sentence explains the boundary so the
        # answer makes sense whatever was asked.
        'out_of_scope': {
            # Kept short on purpose. Spoken Hindi runs about 0.08s per
            # character here, so the fuller wording took 11.5s to say - longer
            # than most real answers, for a reply that carries no information.
            # This one says the same thing in 6.3s.
            'en': ("This is not in the ASHA manual. If it is a health "
                   "question, ask your health centre."),
            'hi': ("यह सवाल आशा की किताब में नहीं है। सेहत से जुड़ा हो तो "
                   "स्वास्थ्य केंद्र से पूछें।"),
        },
        'not_understood': {
            'en': "Sorry, I did not catch that. Please ask again.",
            'hi': "माफ़ कीजिए, समझ नहीं आया। कृपया दोबारा पूछें।",
        },
    }

    def _fixed_reply(self, key):
        """Return (english_for_the_log, text_in_the_output_language)."""
        entry = self.FIXED_REPLIES[key]
        out_lang = self.settings['output_language']
        return entry['en'], entry.get(out_lang, entry['en'])

    # The ASHA's word and the manual's word are not always the same, and a
    # bag-of-words retriever has no way to know they mean one thing. Asked in
    # Hindi about the नाभी, the translator says "navel"; the manual only ever
    # writes "umbilicus", "umbilical" or "cord", so a question about pus at the
    # navel scored 1.13 and 1.36 against a 1.60 floor and was refused - while
    # page 75 says plainly what to do about it. Phrased as "umbilical cord" the
    # same question scored 1.65 and was answered, which is the whole bug in two
    # numbers.
    #
    # Built by taking fifteen realistic questions, translating them the way the
    # device does, and listing every word the manual never uses. Four came
    # back; these two are the ones with a real equivalent in the text. Kept
    # this short on purpose - a general thesaurus would blur the scope test
    # that keeps cricket scores out.
    _SYNONYMS = {
        'navel': ('umbil', 'cord'),
        'swoll': ('disten', 'swell'),
        # "When is the measles vaccine administered?" scored 2.34 against a 3.0
        # floor purely because the manual says given, never administered, and
        # in a three-word question one unmatched word is a third of the score.
        'admin': ('given', 'dose'),
    }

    @classmethod
    def _variants(cls, term):
        return (term,) + cls._SYNONYMS.get(term, ())

    def _retrieve_context(self, query, top_k=3):
        '''TF-IDF retrieval: score chunks by content-word overlap weighted by
        term rarity, so "abortion"/"diarrhea" beat stopword noise. Returns []
        when nothing clears MIN_RELEVANCE, which is what makes the device
        refuse instead of inventing.'''
        if not self.knowledge_chunks:
            return []
        q_terms = set(self._terms(query))
        if not q_terms:
            return []
        import math
        # Weight of everything the question is asking about, used below to ask
        # how much of it a page actually accounts for.
        # Each query term is matched through its variants, so the word an
        # ASHA uses finds the word the manual prints. The weight of a term
        # comes from whichever variant the manual actually knows.
        variants = {t: self._variants(t) for t in q_terms}
        term_idf = {t: max([self._idf.get(v, 0.0) for v in vs] or [0.0]) or 1.0
                    for t, vs in variants.items()}
        idf_total = sum(term_idf.values()) or 1.0
        # Two different questions get two different numbers. "Is this subject
        # in the manual at all" is answered by the raw overlap, which separates
        # health questions from cricket scores cleanly. "Which page answers it"
        # is answered by the coverage-weighted score below. Using the weighted
        # score for both was tried and fails the first job: it crushes short
        # queries that match one rare word, putting a real question about a
        # copper T (0.21) below a question about recharging a phone (0.83).
        scores = []
        raw_best = 0.0
        term_best = {}
        for i, tf in enumerate(self._chunk_tf):
            matched, score = [], 0.0
            # log(1+tf) * idf: a chunk mentioning "diarrhoea" 20x beats a short
            # TOC page mentioning it once, without being length-normalized away.
            for t in q_terms:
                best = max((math.log(1 + tf[v]) * self._idf.get(v, 1.0)
                            for v in variants[t] if v in tf), default=0.0)
                if best > 0.0:
                    matched.append(t)
                    score += best
                # Scope is judged across the whole manual, not one page:
                # see term_best below.
                if best > term_best.get(t, 0.0):
                    term_best[t] = best

            # ...but summing term contributions on their own let one common
            # word carry a page past the page that actually answers. Asked how
            # to stay clean during menstruation, a sepsis page won on "clean"
            # and "keep" while scoring nothing on "menstrual"; asked when a
            # copper T can be fitted, the ORS page won on "tea" alone. In both
            # cases the right page was sitting at rank 2. Coverage is the share
            # of the question's own idf weight that this page accounts for, and
            # squaring it makes a page that ignores the distinctive word lose
            # to one that does not. Measured over six labelled questions, this
            # ranks the correct page first in all six; raw scoring managed four.
            raw_best = max(raw_best, score)
            coverage = (sum(term_idf[t] for t in matched) / idf_total)
            # Applied once, not
            # squared. Squaring ranked one more labelled case first, but it
            # also pulled a low-birth-weight weighing schedule ("Days 7, 14,
            # 21, 28") into the context for "how often should children be
            # weighed", and the model answered from that instead of the
            # monthly guidance - 9/10 on the eval, every run. Applied once,
            # that page stays out and the eval holds at 10/10.
            scores.append((score * coverage, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        # Whether the manual covers the subject is a different question from
        # which page answers it, and asking it of a single page gets it wrong.
        # "The baby's navel has turned red and pus is coming out" scored 1.45
        # because the highest-scoring page carried only two of its six words,
        # and was refused - while page 75 says what to do about pus on the
        # umbilicus. Letting every word find its own best page instead
        # separates the two populations properly: measured over ten real
        # questions and nine off-topic ones, in-scope scores 3.55-6.97 and
        # out-of-scope 0.00-2.53. The old one-page measure overlapped - its
        # worst in-scope question scored below its best off-topic one.
        best_per_term = sum(term_best.values()) / len(q_terms)
        # asha_eval/run_eval.py calls this directly on a bare instance that
        # has no application logger, so do not assume one exists.
        log = getattr(self, "logger", None) or logging.getLogger(__name__)
        if best_per_term < self.MIN_RELEVANCE:
            log.info("Best chunk scored %.2f per term, below %.2f - out of scope",
                     best_per_term, self.MIN_RELEVANCE)
            return []
        log.info("Retrieval relevance %.2f per term", best_per_term)
        return [self.knowledge_chunks[i] for score, i in scores[:top_k] if score > 0]

    def ui_cb(self, msg):
        if msg == 'Reset':
            self.logger.info('Reset!')
            check_output('systemctl restart pocketinfer', shell=True)
        elif msg == 'Reboot':
            self.logger.info('REbooting!')
            # check_output('reboot', shell=True)
        elif msg == 'Shutdown':
            self.logger.info('Shutdown!')
            # check_output('halt', shell=True)
        elif msg.startswith('ASR'):
            self.settings['input_language'] = msg[4:].lower()
        elif msg.startswith('TTS'):
            self.settings['output_language'] = msg[4:].lower()

    def delayed_write_toptext(self, text, delay=1.0):
        def delayed_write(text, delay):
            time.sleep(delay)
            self.board.top_text(text)
        th = threading.Thread(target=delayed_write, args=(text, delay), daemon=True)
        th.start()

    def delayed_write_bottext(self, text, delay=1.0):
        def delayed_write(text, delay):
            time.sleep(delay)
            self.board.bottom_text(text)
        th = threading.Thread(target=delayed_write, args=(text, delay), daemon=True)
        th.start()

    def delayed_write_led_anim(self, val, delay=1.0):
        def delayed_write(val, delay):
            time.sleep(delay)
            self.board.led_animation(val)
        th = threading.Thread(target=delayed_write, args=(val, delay), daemon=True)
        th.start()


    # Speak each sentence as the model finishes writing it, rather than
    # waiting for the whole answer, then the whole translation, then the whole
    # synthesis. Measured on this board: generation ends at 1.65-2.57s, and the
    # old path then spent a further 0.82s translating and 1.66s synthesising
    # before any sound came out. Per sentence those cost about 0.5s and 0.7s
    # and they overlap the model still writing the next one, so the first audio
    # starts roughly 2.5s earlier. Set POCKETINFER_STREAM=0 to fall back to the
    # single-shot path.
    STREAM_ANSWER = os.environ.get('POCKETINFER_STREAM', '1') == '1'
    # Three, not two. The model opens by restating the question in about one
    # answer in four, and at two sentences that left only one for the answer
    # itself: asked what to do when a newborn will not breathe, it gave suction
    # of the throat and then the nose and stopped, losing "ventilate with bag
    # and mask" - the step that matters most. The prompt now forbids restating
    # as well, but the budget should not depend on the model obeying it.
    MAX_ANSWER_SENTENCES = 3

    @staticmethod
    def _join_wavs(chunks):
        """Concatenate WAV clips into one, for the interaction log."""
        if not chunks:
            return b""
        if len(chunks) == 1:
            return chunks[0]
        first = wave.open(BytesIO(chunks[0]), 'rb')
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
        for c in chunks[1:]:
            w = wave.open(BytesIO(c), 'rb')
            frames.append(w.readframes(w.getnframes()))
        out = BytesIO()
        with wave.open(out, 'wb') as w:
            w.setparams(params)
            w.writeframes(b"".join(frames))
        return out.getvalue()

    def _pipeline_answer(self, llm_prompt, query):
        """Stream, translate, synthesise and play the answer sentence by sentence.

        Returns (english, translated, audio_bytes, first_audio_at). Playback
        happens inside, overlapped with generation: a background thread turns
        each finished sentence into audio while the foreground plays whatever
        is already queued, so the two never wait on each other.
        """
        out_lang = self.settings['output_language']
        audio_q = queue.Queue()
        english, translated = [], []

        def produce():
            try:
                for sent in self.ollama.generate_sentences(
                        llm_prompt, max_sentences=self.MAX_ANSWER_SENTENCES):
                    if not english:
                        sent = self._strip_lead_in(sent)
                        # _strip_restatement needs a whole answer; here the
                        # sentences arrive one at a time, so test this one and
                        # skip it rather than letting it be spoken.
                        if self._is_restatement(sent, query):
                            continue
                    if self.NO_ANSWER_SENTINEL in sent.upper():
                        # Caught before synthesis, so nothing of it is spoken.
                        # Leaving english empty makes the caller fall back to
                        # the fixed refusal.
                        english[:] = []
                        break
                    english.append(sent)
                    hi = sent if out_lang == 'en' else self.nmt.infer(
                        self._disambiguate_for_translation(sent),
                        "EN", out_lang)['translated_text']
                    translated.append(hi)
                    audio_q.put((hi, base64.b64decode(
                        self.tts.infer(hi, out_lang)['audio_base64'])))
            except Exception:
                # Never let a streaming failure hang the foreground. It must
                # also never be reported to the ASHA as "this is not in the
                # manual": a NameError in this thread once did exactly that,
                # turning every streamed answer into a refusal while retrieval
                # was still scoring 4.75. The flag tells the caller to fall
                # back to the single-shot path instead of refusing.
                self.logger.exception("Streaming answer failed")
                failed.append(True)
            finally:
                audio_q.put(None)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()

        chunks, first_audio_at = [], None
        failed = []
        while True:
            item = audio_q.get()
            if item is None:
                break
            _, wav = item
            if first_audio_at is None:
                first_audio_at = time.time()
                self.board.statusbar("Running: Playback")
                self.delayed_write_led_anim(0)
            chunks.append(wav)
            self.board.bottom_text(" ".join(translated))
            clip = wave.open(BytesIO(wav), 'rb')
            with AudioPlayer(clip.getframerate(),
                             self.board.alsa_playback_device) as player:
                player.play(clip.readframes(clip.getnframes()))
        worker.join(timeout=2.0)
        if failed:
            return None
        return (" ".join(english), " ".join(translated),
                self._join_wavs(chunks), first_audio_at)

    def run(self):
        self.logger.debug('Starting with settings: %s', self.settings)
        while self.running:
            try:
                self.board.statusbar("Ready - Press to speak")
                # self.board.UI.force_refresh()
                self.board.wait_for_trigger_button_down()
                self.board.statusbar("Sun raha hoon - boliye")
                self.board.top_text("")
                self.board.bottom_text("")
                audio_start = time.time()
                # When user presses button, start recording audio.
                # Camera/image capture is intentionally not used for this deployment -
                # the app answers spoken questions directly rather than describing a
                # photographed scene. Re-add camera_frame_jpg() + pass images=[img] to
                # ollama.generate() below if visual grounding is needed again later.
                if self.piper is not None:
                    self.piper.stop_playback()  # If previous TTS is still playing, stop it
                self.board.audio.start()
                self.board.wait_for_trigger_button_up()
                audio_stop = time.time()
                # When user releases button, stop recording
                self.board.audio.stop()
                self.board.statusbar("Running: ASR")
                self.board.led_animation(1)
                asr_start = time.time()
                # Perform ASR on the recorded audio, convert it to text
                if self.settings["input_language"] != 'en':
                    # Bhashini ASR expects 16kHz; the mic negotiates ~44.1kHz (audio.py).
                    # Without convert_rate, get_wav_data() sends the raw captured rate
                    # unresampled, which corrupts conjunct consonants on longer utterances.
                    # (The English/Vosk path below already does this correctly.)
                    wav_bytes = self.board.audio.to_audio_data().get_wav_data(convert_rate=16000)
                    asr_result = self.asr.infer(wav_bytes, self.settings["input_language"])
                else:
                    asr_result = self.vosk.recognize(self.board.audio.to_audio_data())
                raw_query = self._normalise_transcript(asr_result['text'])
                asr_stop = time.time()
                self.logger.info("Detected query is '{}'".format(raw_query))
                self.board.top_text(raw_query)
                if not raw_query.strip():
                    # ASR occasionally returns 200 with an empty transcription (e.g. a
                    # very short/quiet press). NMT rejects empty text with a 400, which
                    # would otherwise crash this loop iteration.
                    self.logger.warning("ASR returned empty text, skipping this interaction")
                    # Shown to a Hindi-speaking user, so write it in Hindi -
                    # the rest of the screen is Hindi-only by this point.
                    self.board.bottom_text(
                        "सुनाई नहीं दिया। कृपया दोबारा बोलें।"
                        if self.settings["output_language"] != "en"
                        else "Sorry, I didn't catch that. Please try again.")
                    continue
                # Perform NMT on the recognized text, convert it to the target language
                if self.settings['input_language'] != 'en':
                    self.board.statusbar(f"Running: NMT {self.settings['input_language']} -> en")
                    query = self.nmt.infer(
                        self._prepare_hindi_for_translation(raw_query),
                        self.settings["input_language"], "EN")['translated_text']
                    self.logger.info("Translated query is '{}'".format(query))
                    # The English translation is an internal pipeline step, not
                    # something to put in front of the user: showing it here
                    # replaced the Hindi question mid-interaction, so the screen
                    # flipped Hindi -> English -> Hindi and read as a glitch.
                    # It stays in the log for debugging.
                else:
                    query = raw_query
                nmt_a_stop = time.time()
                # Perform LLM inference on the recognized text (no image - text-only Q&A)
                self.board.statusbar("Running: LLM")
                llm_start = time.time()
                preset_output = None

                # Retrieve relevant pages from ASHA Module-7 knowledge base
                # top_k=3, not 2: retrieval ranks by term overlap, so a page
                # that merely repeats a common word can outrank the right one.
                # Measured case - "water comes out during delivery" put the
                # ORS/diarrhoea page (which says "water" repeatedly while
                # explaining how to mix ORS) and the sepsis page above the
                # actual labour/amniotic-fluid page, which landed at rank 3
                # and so was cut off entirely at top_k=2, producing a wrong
                # "not in my materials" answer for content the manual does
                # cover. 8/8 vs 7/8 on the regression set; prompt grows to
                # ~1.1k tokens, which measurably costs nothing (3.5s LLM).
                context_chunks = self._retrieve_context(query, top_k=3)
                if context_chunks:
                    context = '\n\n'.join(context_chunks)
                    self.logger.info(f"Retrieved {len(context_chunks)} context chunks for RAG")
                else:
                    context = ""
                    self.logger.warning("No relevant context found in knowledge base")

                # Powerful RAG prompt: restrict LLM to ONLY use provided context
                if context:
                    llm_prompt = self.build_prompt(context, query)
                    if self.STREAM_ANSWER:
                        # Speaks as it goes; nothing left to do below but log.
                        resp = None
                        streamed = self._pipeline_answer(llm_prompt, query)
                        if streamed is None:
                            # Streaming broke; answer the ordinary way rather
                            # than telling her the manual does not cover it.
                            self.logger.warning(
                                "Falling back to single-shot generation")
                            resp = self.ollama.generate(images=[], prompt=llm_prompt)
                            result = self._strip_restatement(
                                self._strip_lead_in(resp.response.strip()), query)
                            if self.NO_ANSWER_SENTINEL in result.upper():
                                result = ""
                        else:
                            result = streamed[0].strip()
                        if streamed is not None and not result:
                            # Streaming produced nothing usable - fall through
                            # to the ordinary path's empty-answer handling.
                            streamed = None
                    else:
                        streamed = None
                        resp = self.ollama.generate(images=[], prompt=llm_prompt)
                        result = self._strip_restatement(
                            self._strip_lead_in(resp.response.strip()), query)
                        if self.NO_ANSWER_SENTINEL in result.upper():
                            result = ""
                else:
                    # Nothing in the manual matched, so there is nothing to
                    # ground an answer in. Do NOT fall back to asking the LLM
                    # unaided: this device answers as an ASHA health manual,
                    # and an ungrounded model happily answers anything from
                    # its own knowledge (measured: "What is the capital of
                    # France?" -> "Paris", "How do I fix my motorcycle
                    # engine?" -> engine repair advice). Worse, on a health
                    # question it would invent plausible-sounding medical
                    # guidance. Refusing here is also faster - no LLM call.
                    resp = None
                    streamed = None
                    result, preset_output = self._fixed_reply('out_of_scope')
                llm_end = time.time()
                # Fixed replies are already the length they should be; the
                # two-sentence trim below is for model output only, and was
                # clipping the referral half off the logged refusal.
                if result and streamed is None and preset_output is None:
                    # Safety net: on complex/sensitive topics the model sometimes
                    # ignores the length instruction and rambles for the whole
                    # token budget, producing minutes of TTS audio. Cut to at
                    # most the first 2 sentences (prompt now asks for 1-2) so
                    # playback stays reasonable regardless.
                    cut_at = None
                    search_from = 0
                    for _ in range(2):
                        next_idx = None
                        for end in ('. ', '.\n', '! ', '? '):
                            idx = result.find(end, search_from)
                            if idx != -1 and (next_idx is None or idx < next_idx):
                                next_idx = idx
                        if next_idx is None:
                            break
                        cut_at = next_idx
                        search_from = next_idx + 1
                    if cut_at is not None:
                        result = result[:cut_at + 1]
                    elif len(result) > 320:
                        result = result[:320].rsplit(' ', 1)[0] + '.'
                if not result:
                    # The model occasionally burns its whole token budget on hidden
                    # "thinking" despite /no_think and returns nothing (see
                    # models/ollama.py's done_reason=="length" warning). Fall back
                    # to a fixed message instead of crashing NMT with empty text.
                    # Two ways to get here: the model emitted the no-answer
                    # sentinel because the retrieved pages did not cover the
                    # question, or it returned nothing at all. The first is far
                    # more common now that the prompt asks for that decision up
                    # front, and saying so is more useful than claiming not to
                    # have heard.
                    result, preset_output = self._fixed_reply(
                        'out_of_scope' if context_chunks else 'not_understood')
                self.logger.info("Result is '{}'".format(result))
                # Likewise the English answer: the user sees only the translated
                # Hindi one, written below once NMT has produced it.
                # Perform NMT on the LLM response, convert it back to the original language
                first_audio_at = None
                if preset_output is not None:
                    nmt_result = preset_output
                elif streamed is not None:
                    # Already translated and spoken, sentence by sentence.
                    _, nmt_result, tts_result_bytes, first_audio_at = streamed
                    self.logger.info("Translated result is '{}'".format(nmt_result))
                elif self.settings['output_language'] != 'en':
                    self.board.statusbar(f"Running: NMT en -> {self.settings['output_language']}")
                    nmt_result = self.nmt.infer(
                        self._disambiguate_for_translation(result),
                        "EN", self.settings["output_language"])['translated_text']
                    self.logger.info("Translated result is '{}'".format(nmt_result))
                else:
                    nmt_result = result
                nmt_b_stop = time.time()
                # Question stays as the user asked it (already on screen since
                # ASR); only the answer needs writing, and immediately - the
                # delayed variants existed to undo the English flashes above.
                if streamed is None:
                    self.board.bottom_text(nmt_result)
                    # Perform TTS on the LLM response, convert it to audio and play it back
                    self.board.statusbar("Running: Playback")
                    self.delayed_write_led_anim(0)
                    tts_result = self.tts.infer(nmt_result, self.settings["output_language"])
                    tts_result_bytes = base64.b64decode(tts_result['audio_base64'])
                    app_end = time.time()
                    wave_obj = wave.open(BytesIO(tts_result_bytes), 'rb')
                    with AudioPlayer(wave_obj.getframerate(), self.board.alsa_playback_device) as player:
                        player.play(wave_obj.readframes(wave_obj.getnframes()))
                else:
                    app_end = time.time()
                self.logger.debug(f"Total Run time {app_end-audio_start}s, audio {audio_stop-audio_start}s, ASR {asr_stop-asr_start}, NMT A {nmt_a_stop-asr_stop}, LLM {llm_end-llm_start}, NMT B {nmt_b_stop-llm_end}, TTS {app_end-nmt_b_stop}")
                # Log
                log_id = int(audio_start*1000)
                log_data = {
                    'id': log_id,
                    "query": asr_result,
                    # resp is None when retrieval found nothing and the LLM was
                    # deliberately skipped (see the no-context branch above).
                    "response": resp.model_dump() if resp is not None else None,
                    "timestamps": {
                        "audio_start": audio_start,
                        "audio_stop": audio_stop,
                        "asr_start": asr_start,
                        "asr_stop": asr_stop,
                        "nmt_a_stop": nmt_a_stop,
                        "llm_start": llm_start,
                        "llm_end": llm_end,
                        "nmt_b_stop": nmt_b_stop,
                        "app_end": app_end,
                        # The number the user actually feels: press-stop to
                        # first audible word. None on the non-streaming path.
                        "first_audio_at": first_audio_at
                    },
                    "answer_en": result,
                    "answer_out": nmt_result,
                    "streamed": streamed is not None
                }
                with open("/tmp/hear_the_world_en_logs/log.jsonl", "a") as f:
                    f.write(json.dumps(log_data)+"\n")
                with open("/tmp/hear_the_world_en_logs/audio_{}.wav".format(log_id), "wb") as f:
                    f.write(tts_result_bytes)
                # This board has no RAM to spare for "free later" - CPython's
                # allocator and glibc's malloc both tend to hold freed heap
                # pages rather than returning them to the OS, so RSS creeps
                # up across many interactions (audio buffers, RPC call
                # objects, HTTP response bodies) even with no real leak.
                # Force a collect + malloc_trim after every interaction.
                gc.collect()
                if _libc is not None:
                    _libc.malloc_trim(0)
                # Loop back around and prepare for the next interactionw
            except KeyboardInterrupt:
                self.logger.info("Exit")
                self.board.clear_screen()
                self.running = False
            except Exception as e:
                self.logger.exception("Error in main application loop: %s", e)
                self.board.statusbar("Error: {}".format(str(e)))
                time.sleep(1)

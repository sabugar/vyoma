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
                        text = f.read()
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
    MIN_RELEVANCE = 1.6

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
        scores = []
        for i, tf in enumerate(self._chunk_tf):
            # log(1+tf) * idf: a chunk mentioning "diarrhoea" 20x beats a short
            # TOC page mentioning it once, without being length-normalized away.
            score = 0.0
            for t in q_terms:
                if t in tf:
                    score += math.log(1 + tf[t]) * self._idf.get(t, 1.0)
            scores.append((score, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        best_per_term = scores[0][0] / len(q_terms) if scores else 0.0
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
    MAX_ANSWER_SENTENCES = 2

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

    def _pipeline_answer(self, llm_prompt):
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
                    english.append(sent)
                    hi = sent if out_lang == 'en' else \
                        self.nmt.infer(sent, "EN", out_lang)['translated_text']
                    translated.append(hi)
                    audio_q.put((hi, base64.b64decode(
                        self.tts.infer(hi, out_lang)['audio_base64'])))
            except Exception:
                # Never let a streaming failure hang the foreground; the caller
                # checks for an empty answer and uses the usual fallback text.
                self.logger.exception("Streaming answer failed")
            finally:
                audio_q.put(None)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()

        chunks, first_audio_at = [], None
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
                raw_query = asr_result['text']
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
                    query = self.nmt.infer(raw_query, self.settings["input_language"], "EN")['translated_text']
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
                    llm_prompt = (
                        f'Answer the question using ONLY the CONTEXT below. '
                        f'Output ONLY the final answer, in 1-2 complete sentences (max 35 words), '
                        f'including the specific relevant detail(s) from the context (numbers, steps, '
                        f'signs, timing) so the answer sounds informed, not vague. '
                        f'Do not explain your reasoning, do not describe the context, do not say '
                        f'"let me check" or "the text says" or similar - just state the answer directly, '
                        f'as if you already knew it. '
                        f'If the context does not contain the answer, output exactly: '
                        f'"I do not have this information in my training materials."\n\n'
                        f'CONTEXT:\n{context}\n\n'
                        f'QUESTION: {query}\n\n'
                        f'ANSWER:'
                    )
                    if self.STREAM_ANSWER:
                        # Speaks as it goes; nothing left to do below but log.
                        resp = None
                        streamed = self._pipeline_answer(llm_prompt)
                        result = streamed[0].strip()
                        if not result:
                            # Streaming produced nothing usable - fall through
                            # to the ordinary path's empty-answer handling.
                            streamed = None
                    else:
                        streamed = None
                        resp = self.ollama.generate(images=[], prompt=llm_prompt)
                        result = resp.response.strip().rstrip()
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
                if result and streamed is None:
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
                    result, preset_output = self._fixed_reply('not_understood')
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
                    nmt_result = self.nmt.infer(result, "EN", self.settings["output_language"])['translated_text']
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

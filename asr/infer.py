import base64
import io
import json
import os
import time
import wave
import librosa
import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 16000


class ASRInference:

    # -----------------------------
    # Init → Load all models once
    # -----------------------------
    def __init__(self, checkpoint_dir="checkpoints"):

        # torch preprocessor + Tamil model are lazy-loaded on first actual
        # 'ta' request. This deployment is Hindi-only, and importing torch
        # costs ~1GB of RAM the 8GB board does not have spare - the Hindi v2
        # path below is pure librosa+numpy+onnxruntime.
        self.preprocessor = None
        self._checkpoint_dir = checkpoint_dir
        self.sessions = {}

        # Hindi uses a separate, larger (600M param, int8-quantized) model -
        # the original hi-conformer.onnx's hardcoded 89-token vocab was missing
        # 9 Devanagari consonants (झ छ घ ठ ढ ण ष ङ ञ) and its blank-index
        # computation (blank = len(vocab)) silently dropped ~40 more valid
        # model outputs, corrupting any word containing those characters
        # (e.g. मुझे -> मुे, कुछ -> कु). This replacement model's own vocab is
        # verified complete (257 tokens/language) and produces exact-match
        # transcriptions on words that previously failed.
        v2_dir = os.path.join(os.path.dirname(checkpoint_dir), "checkpoints_v2")
        print("Loading Hindi ASR v2 ONNX sessions...")
        self.hi_encoder_sess = ort.InferenceSession(
            f"{v2_dir}/onnx/encoder_quantized_int8.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.hi_ctc_sess = ort.InferenceSession(
            f"{v2_dir}/onnx/ctc_decoder_quantized_int8.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        with open(f"{v2_dir}/config/vocab.json", encoding="utf-8") as f:
            self.hi_vocab = json.load(f)["hi"]
        with open(f"{v2_dir}/config/language_masks.json", encoding="utf-8") as f:
            self.hi_mask = np.array(json.load(f)["hi"], dtype=bool)

        # onnxruntime pays a one-time cost (CUDA graph/kernel selection) on the
        # first real inference call - pay it now during service startup instead
        # of on the first user request (measured ~19s cold vs <2s once warm).
        print("Warming up Hindi ASR v2 (full inference path)...")
        # Warm through the SAME entry point real requests use, not just the
        # ONNX sessions: the first call also pays librosa/numba JIT compilation
        # and CUDA kernel selection. Warming only the sessions left a ~6-8s
        # penalty on the very first real question (vs ~0.7s once warm) - the
        # worst possible moment for it. One bucket length is enough to trigger
        # the one-time costs; the per-shape work is small by comparison.
        silence = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((silence * 32767).astype(np.int16).tobytes())
        warm_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        for _ in range(2):
            self._infer_hi_v2(warm_b64)

        print("ASR Models loaded successfully.")

        # -------------------------
        # Vocabs
        # -------------------------
        self.VOCAB_TA = [
            '<unk>', 'ா', 'ி', 'ு', 'வ', 'க', '▁ப', 'ை', 'ன', 'ர',
            'ன்', '்', '▁க', 'ம்', 'த', 'ே', 'ய', 'ல்', '▁அ', 'ர்',
            'க்க', '▁வ', 'ல', '▁ம', 'து', 'ட', 'ப்ப', 'ம', '▁த',
            'ப', '▁', 'ச'
        ]

    # -----------------------------
    # CTC Decoder
    # -----------------------------
    def decode_ctc(self, logits, vocab):

        blank = len(vocab)

        token_ids = np.argmax(logits, axis=-1)[0]

        tokens = []
        prev = blank

        for t in token_ids:
            t = int(t)

            if t != blank and t != prev and t < len(vocab):
                tokens.append(vocab[t])

            prev = t

        text = "".join(tokens).replace("▁", " ").strip()

        return text

    # -----------------------------
    # Base64 → Waveform
    # -----------------------------
    def load_audio(self, audio_base64):

        import torch  # lazy: Tamil path only - see __init__ note

        audio_bytes = base64.b64decode(audio_base64)

        signal_np, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True
        )

        if signal_np.size == 0:
            raise ValueError("Empty audio")

        signal = torch.tensor(signal_np).unsqueeze(0)
        length = torch.tensor([signal.shape[1]])

        return signal, length

    # -----------------------------
    # Hindi v2: mel-spectrogram preprocessing (matches the model's own
    # training config: 16kHz, n_fft=512, win=400, hop=160, 80 mel bins)
    # -----------------------------
    # Safety bound only - a stuck/held button shouldn't be able to feed an
    # unbounded clip into the model. Deliberately generous: real questions
    # here are scenario-style and run 20s+ ("एक परिवार बच्चे को जन्म के एक घंटे
    # बाद नहलाने की तैयारी कर रहा है..."), and a 16s cap silently truncated
    # them mid-sentence, which then produced a confidently wrong answer to
    # half a question. Variable input shapes are fine: the first-question
    # latency spike turned out to be librosa/numba JIT (fixed by warming the
    # real inference path in __init__), not per-shape CUDA tuning.
    MAX_SECONDS = 45

    def _preprocess_hi_v2(self, audio_base64):
        audio_bytes = base64.b64decode(audio_base64)
        wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
        if wav.size == 0:
            raise ValueError("Empty audio")
        max_samples = self.MAX_SECONDS * SAMPLE_RATE
        if wav.size > max_samples:
            wav = wav[:max_samples]
        mel = librosa.feature.melspectrogram(
            y=wav, sr=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
            fmin=0.0, fmax=8000.0, n_mels=80, window="hann", power=2.0,
        )
        feats = np.log(mel + 1e-9)
        mean = feats.mean(axis=1, keepdims=True)
        std = feats.std(axis=1, keepdims=True) + 1e-5
        feats = (feats - mean) / std
        return feats.astype(np.float32)

    def _infer_hi_v2(self, audio_base64):
        feats = self._preprocess_hi_v2(audio_base64)
        length = np.array([feats.shape[1]], dtype=np.int64)
        feats = np.expand_dims(feats, axis=0)

        enc_inputs = self.hi_encoder_sess.get_inputs()
        enc_dict = {enc_inputs[0].name: feats}
        if len(enc_inputs) > 1:
            enc_dict[enc_inputs[1].name] = length
        enc_out = self.hi_encoder_sess.run(None, enc_dict)[0]

        ctc_inputs = self.hi_ctc_sess.get_inputs()
        ctc_dict = {ctc_inputs[0].name: enc_out}
        if len(ctc_inputs) > 1:
            ctc_dict[ctc_inputs[1].name] = length
        logits = self.hi_ctc_sess.run(None, ctc_dict)[0]

        logits_sliced = logits[:, :, self.hi_mask]
        pred_ids = np.argmax(logits_sliced, axis=-1)[0]

        tokens = []
        prev = None
        for idx in pred_ids:
            idx = int(idx)
            if idx != prev and idx != 256 and idx < len(self.hi_vocab):
                tokens.append(self.hi_vocab[idx])
            prev = idx
        return "".join(tokens).replace("▁", " ").strip()

    # -----------------------------
    # Main Inference
    # -----------------------------
    def infer(self, audio_base64: str, language: str):

        start_time = time.time()

        if language == "hi":
            text = self._infer_hi_v2(audio_base64)
            return {
                "text": text,
                "language": language,
                "processing_time_sec": round(time.time() - start_time, 3)
            }

        if language != "ta":
            raise ValueError(f"Unsupported language: {language}")

        if "ta" not in self.sessions:
            import torch  # lazy: Tamil path only
            print("Lazy-loading Tamil ASR ONNX session + preprocessor...")
            self.sessions["ta"] = ort.InferenceSession(
                f"{self._checkpoint_dir}/ta-conformer.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.preprocessor = torch.jit.load(
                f"{self._checkpoint_dir}/hi-conformer_preprocess.pt",
                map_location="cpu"
            ).eval()

        vocab = self.VOCAB_TA

        # Audio
        signal, length = self.load_audio(audio_base64)

        # Preprocess (torch was already imported in the lazy-load block above
        # or inside load_audio - re-import is a no-op and keeps it local)
        import torch
        with torch.no_grad():
            feats, feat_len = self.preprocessor(signal, length)

        # ONNX inference
        session = self.sessions[language]

        logits = session.run(
            [session.get_outputs()[0].name],
            {
                session.get_inputs()[0].name: feats.numpy(),
                session.get_inputs()[1].name: feat_len.numpy(),
            }
        )[0]

        text = self.decode_ctc(logits, vocab)

        return {
            "text": text,
            "language": language,
            "processing_time_sec": round(
                time.time() - start_time, 3
            )
        }

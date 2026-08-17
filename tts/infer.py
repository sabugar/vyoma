import base64
import io
import json
import subprocess
import os
import tempfile
import time
import unicodedata

import numpy as np
import onnxruntime as ort

from tts.hindi_numbers import spell_out_numbers


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FLITE_BIN = os.path.join(BASE_DIR, "flite", "bin", "flite")
VOICES_DIR = os.path.join(BASE_DIR, "flite", "voices")

# Hindi previously used Flite's cmu_indic_hin_ab voice - old diphone-concatenation
# synthesis, sounds dated/robotic. Replaced with a neural VITS model (fine-tuned
# female voice), exported to ONNX (model_sr0p8.onnx, speaking_rate=0.8 baked in
# during export for clearer, slower playback).
#
# Why ONNX instead of PyTorch: torch+transformers added ~1.5GB to this service's
# memory footprint. On the 8GB Jetson (LLM + 600M ASR already resident) that
# caused swap thrash, OOM kills of this service, and device hangs. onnxruntime
# is already loaded here for ASR, so the TTS model now costs only its own
# ~114MB. Measured: 3s synthesis (torch CPU took 8-17s).
HI_VITS_DIR = os.path.join(BASE_DIR, "hi_female_vits")
# speaking_rate=0.8 and noise_scale=0.30 are baked in at export time. VITS's
# default noise_scale of 0.667 makes the model sample more freely, which
# slurred consonants enough that a listener had to concentrate to follow a
# sentence - not acceptable when the content is health guidance. 0.30 is
# noticeably crisper at the cost of some expressiveness.
HI_VITS_ONNX = os.path.join(HI_VITS_DIR, "model_sr0p8_ns0p3.onnx")
HI_SAMPLE_RATE = 16000  # from hi_female_vits/config.json


class TTSInference:

    def __init__(self):

        # Validate paths once
        if not os.path.exists(FLITE_BIN):
            raise RuntimeError(
                f"Flite binary not found: {FLITE_BIN}"
            )

        if not os.path.exists(VOICES_DIR):
            raise RuntimeError(
                f"Voices directory not found: {VOICES_DIR}"
            )

        print("Loading Hindi neural TTS (VITS ONNX) model...")
        with open(os.path.join(HI_VITS_DIR, "vocab.json"), encoding="utf-8") as f:
            self.hi_vocab = json.load(f)
        # CUDA first: TTS was the single biggest stage in the pipeline at
        # 4.3s on CPU (vs ~1.5s for everything else combined), and the ASR
        # sessions in this same process already pay for the process's CUDA
        # context - so putting this ~114MB model on GPU too is nearly free
        # memory-wise. Falls back to CPU automatically if CUDA is unavailable.
        self.hi_sess = ort.InferenceSession(
            HI_VITS_ONNX,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # Warm up: first-call graph optimizations are paid once at startup,
        # not on the first user request.
        self._synthesize_hi_vits("नमस्ते")

        # -------------------------------------------------
        # LANGUAGE → VOICE MAP (Flite languages other than Hindi)
        # -------------------------------------------------
        self.LANG_VOICE_MAP = {
            "bn": ["cmu_indic_ben_rm.flitevox"],
            "gu": [
                "cmu_indic_guj_ad.flitevox",
                "cmu_indic_guj_dp.flitevox",
                "cmu_indic_guj_kt.flitevox",
            ],
            "ka": ["cmu_indic_kan_plv.flitevox"],
            "mr": [
                "cmu_indic_mar_aup.flitevox",
                "cmu_indic_mar_slp.flitevox",
            ],
            "pa": ["cmu_indic_pan_amp.flitevox"],
            "ta": ["cmu_indic_tam_sdr.flitevox"],
            "te": [
                "cmu_indic_tel_kpn.flitevox",
                "cmu_indic_tel_sk.flitevox",
                "cmu_indic_tel_ss.flitevox",
            ],
            "en": [
                "cmu_us_aew.flitevox",
                "cmu_us_ahw.flitevox",
                "cmu_us_awb.flitevox",
                "cmu_us_axb.flitevox",
                "cmu_us_bdl.flitevox",
                "cmu_us_clb.flitevox",
            ],
        }

        print("TTS Engine initialized successfully.")

    # -----------------------------------------------------
    # Hindi neural (VITS ONNX) synthesis
    # -----------------------------------------------------
    def _tokenize_hi(self, text: str):
        '''Pure-Python equivalent of transformers.VitsTokenizer for this model
        (verified byte-identical input_ids): NFC-normalize, map each char via
        vocab.json, drop chars not in vocab (e.g. '।', '.', '?', ','),
        and insert blank token 0 before/after every token (add_blank=True).'''
        text = unicodedata.normalize('NFC', text)
        ids = [0]
        for ch in text:
            tok = self.hi_vocab.get(ch)
            if tok is not None:
                ids.append(tok)
                ids.append(0)
        return ids

    def _synthesize_hi_vits(self, text: str) -> bytes:
        text = spell_out_numbers(text)
        ids = self._tokenize_hi(text)
        if len(ids) <= 1:
            raise ValueError("No speakable characters after tokenization")
        x = np.array([ids], dtype=np.int64)
        mask = np.ones_like(x)
        waveform = self.hi_sess.run(
            None, {'input_ids': x, 'attention_mask': mask}
        )[0].reshape(-1).astype(np.float32)
        # Peak-normalise before quantising. The model's raw output peaks around
        # 0.62-0.66, so a straight conversion threw away a third of the dynamic
        # range and came out quiet on the device's small speaker - which reads
        # as "hard to make out" just as much as poor articulation does.
        # 0.95 rather than 1.0 leaves headroom so nothing clips.
        peak = float(np.abs(waveform).max())
        if peak > 0:
            waveform = waveform / peak * 0.95
        pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(HI_SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    # -----------------------------------------------------
    # Core Synthesis
    # -----------------------------------------------------
    def synthesize_to_bytes(
        self,
        text: str,
        lang: str,
        voice_name: str | None = None,
        duration_stretch: float = 1.0,
        f0_mean: int = 110,
    ) -> bytes:

        if lang == "hi":
            return self._synthesize_hi_vits(text)

        if lang not in self.LANG_VOICE_MAP:
            raise ValueError(
                f"Language '{lang}' not supported"
            )

        voices = self.LANG_VOICE_MAP[lang]

        # Default voice
        if voice_name is None:
            voice_name = voices[0]

        if voice_name not in voices:
            raise ValueError(
                f"Voice '{voice_name}' not valid. "
                f"Available: {voices}"
            )

        voice_path = os.path.join(
            VOICES_DIR,
            voice_name
        )

        if not os.path.exists(voice_path):
            raise RuntimeError(
                f"Voice file not found: {voice_path}"
            )

        # Temp wav file
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp_file:

            tmp_wav_path = tmp_file.name

        # Flite command
        cmd = [
            FLITE_BIN,
            "-voice", voice_path,
            "--setf", f"duration_stretch={duration_stretch}",
            "--setf", f"int_f0_target_mean={f0_mean}",
            "-t", text,
            tmp_wav_path,
        ]

        subprocess.run(cmd, check=True)

        # Read audio
        with open(tmp_wav_path, "rb") as f:
            audio_bytes = f.read()

        os.remove(tmp_wav_path)

        return audio_bytes

    # -----------------------------------------------------
    # Public Inference
    # -----------------------------------------------------
    def infer(
        self,
        text: str,
        language: str,
        voice_name: str | None = None,
        duration_stretch: float = 1.0,
        f0_mean: int = 110,
        return_base64: bool = True,
    ):

        start_time = time.time()

        if not text.strip():
            raise ValueError("Text cannot be empty")

        audio_bytes = self.synthesize_to_bytes(
            text=text,
            lang=language,
            voice_name=voice_name,
            duration_stretch=duration_stretch,
            f0_mean=f0_mean,
        )

        result = {
            "language": language,
            "processing_time_sec": round(
                time.time() - start_time, 3
            )
        }

        if return_base64:
            result["audio_base64"] = (
                base64.b64encode(audio_bytes).decode()
            )
        else:
            result["audio_bytes"] = audio_bytes

        return result

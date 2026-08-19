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

# Hindi TTS history on this device, for anyone wondering why there are three
# generations of model here:
#   1. Flite (cmu_indic_hin_ab) - diphone concatenation, robotic.
#   2. A fine-tuned Meta MMS VITS, exported to ONNX. Understandable but
#      listeners had to concentrate; 16kHz; CC-BY-NC licensed.
#   3. Piper (hi_IN-priyamvada-medium) - current. Chosen on measurements
#      rather than preference:
#        synthesis  0.48s   (VITS ~1.0s, IndicF5 5.8-10.4s)
#        weights    64MB    (VITS 250MB, IndicF5 1.4GB)
#        rate       22kHz   (VITS 16kHz)
#      IndicF5 (ai4bharat, MIT) sounded best of the three and is the natural
#      upgrade if this ever runs on a larger board, but its flow-matching
#      sampler needs 5.8s even for a six-word answer - fp16 diverges to NaN,
#      and nfe below 8 degrades audibly - so it cannot meet the 5-8s
#      end-to-end budget this device is held to.
# The Piper voice is trained on AI4Bharat's indicnlp corpus (see MODEL_CARD)
# and is CC-BY-NC-SA: fine for this deployment, but it would have to change
# before any commercial use.
HI_PIPER_DIR = os.path.join(BASE_DIR, "piper_hi")
HI_PIPER_ONNX = os.path.join(HI_PIPER_DIR, "hi_IN-priyamvada-medium.onnx")

# Kept so the previous voice can be restored without re-downloading.
HI_VITS_DIR = os.path.join(BASE_DIR, "hi_female_vits")
HI_VITS_ONNX = os.path.join(HI_VITS_DIR, "model_sr0p8_ns0p3.onnx")


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

        print("Loading Hindi TTS (Piper) model...")
        from piper import PiperVoice
        self.hi_voice = PiperVoice.load(
            HI_PIPER_ONNX, config_path=HI_PIPER_ONNX + ".json")
        self.hi_sample_rate = self.hi_voice.config.sample_rate
        # Pay onnxruntime's first-call graph optimisation now rather than on
        # the user's first question.
        self._synthesize_hi("नमस्ते")

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
    # Hindi synthesis (Piper)
    # -----------------------------------------------------
    def _synthesize_hi(self, text: str) -> bytes:
        # Numbers still have to be spelled out: Piper reads Devanagari text,
        # and a bare "20" is not reliably voiced as "बीस". Dosages, weeks and
        # weights are exactly the details these answers hinge on.
        text = spell_out_numbers(text)
        chunks = []
        for chunk in self.hi_voice.synthesize(text):
            # Piper's API has moved between returning raw bytes and returning
            # chunk objects; accept either so a library update cannot silently
            # break synthesis.
            chunks.append(getattr(chunk, "audio_int16_bytes", chunk))
        pcm = b"".join(chunks)
        if not pcm:
            raise ValueError("Piper produced no audio")

        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.hi_sample_rate)
            wf.writeframes(pcm)
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
            return self._synthesize_hi(text)

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

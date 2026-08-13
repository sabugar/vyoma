import base64
import io
import time
import librosa
import torch
import numpy as np
import onnxruntime as ort


SAMPLE_RATE = 16000


class ASRInference:

    # -----------------------------
    # Init → Load all models once
    # -----------------------------
    def __init__(self, checkpoint_dir="checkpoints"):

        print("Loading ASR preprocessor...")
        self.preprocessor = torch.jit.load(
            f"{checkpoint_dir}/hi-conformer_preprocess.pt",
            map_location="cpu"
        ).eval()

        print("Loading ASR ONNX sessions...")
        self.sessions = {
            "hi": ort.InferenceSession(
                f"{checkpoint_dir}/hi-conformer.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            ),
            "ta": ort.InferenceSession(
                f"{checkpoint_dir}/ta-conformer.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            ),
        }

        print("ASR Models loaded successfully.")

        # -------------------------
        # Vocabs
        # -------------------------
        self.VOCAB_HI = [
            '<unk>', 'ा', 'े', 'र', 'ी', 'न', 'ि', 'ल', 'क', '्', '▁', 'स',
            'म', 'त', '▁स', 'ो', '▁द', '▁क', 'ट', 'ं', '▁अ', 'प', '▁ब',
            '▁प', 'व', 'ु', 'य', '▁है', '▁म', 'ह', '▁ज', '▁व', '▁आ',
            'ग', 'द', '▁ह', 'ू', 'श', '्र', 'ै', 'ब', '्य', '▁इ', 'ज',
            'ड', '▁न', 'र्', '▁के', '▁ल', '▁में', 'च', 'ए', 'ज़', '▁उ',
            'ख', '▁र', '▁फ', 'ों', 'ॉ', 'भ', '▁ग', 'ंग', 'ता', 'ने',
            '▁और', '▁का', 'ाइ', '्ट', '▁प्र', '▁को', '▁की', '▁कर',
            '▁हो', '▁से', '▁च', 'ध', '▁हैं', 'ई', '्s', '▁तो',
            '▁त', '▁थ', 'फ', 'थ', 'स्ट', '▁कि', 'न्ट', '▁भी'
        ]

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
    # Main Inference
    # -----------------------------
    def infer(self, audio_base64: str, language: str):

        start_time = time.time()

        if language not in self.sessions:
            raise ValueError(f"Unsupported language: {language}")

        vocab = (
            self.VOCAB_HI if language == "hi"
            else self.VOCAB_TA
        )

        # Audio
        signal, length = self.load_audio(audio_base64)

        # Preprocess
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

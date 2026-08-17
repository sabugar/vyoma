import base64

from asr.infer import ASRInference
from nmt.infer import NMTInference
from tts.infer import TTSInference



print("\nInitializing Inference Engines...\n")

asr_engine = ASRInference(
    checkpoint_dir="asr/checkpoints"
)

nmt_engine = NMTInference(
    checkpoint_root="nmt/checkpoints"
)

tts_engine = TTSInference()

print("\nAll Engines Loaded Successfully.\n")


def run_asr(audio_file_path: str, language: str):

    print(f"\nRunning ASR → {language}")

    with open(audio_file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    result = asr_engine.infer(
        audio_base64=audio_b64,
        language=language
    )

    print("ASR Output:", result["text"])

    return result



def run_nmt(text: str, src_lang: str, tgt_lang: str):

    print(f"\nRunning NMT → {src_lang} → {tgt_lang}")

    result = nmt_engine.infer(
        text=text,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    print("NMT Output:", result["translated_text"])

    return result


def run_tts(text: str, language: str, output_wav="tts_out.wav"):

    print(f"\nRunning TTS → {language}")

    result = tts_engine.infer(
        text=text,
        language=language,
        return_base64=True
    )

    # Save audio locally
    audio_bytes = base64.b64decode(
        result["audio_base64"]
    )

    with open(output_wav, "wb") as f:
        f.write(audio_bytes)

    print(f"TTS Audio saved → {output_wav}")

    return result



def run_s2s_pipeline(
    audio_path: str,
    src_lang: str,
    tgt_lang: str
):

    print("\nRunning Speech-to-Speech Pipeline...\n")

    # ASR
    asr_out = run_asr(audio_path, src_lang)
    text_src = asr_out["text"]

    # NMT
    nmt_out = run_nmt(text_src, src_lang, tgt_lang)
    text_tgt = nmt_out["translated_text"]

    # TTS
    tts_out = run_tts(text_tgt, tgt_lang)

    return {
        "asr_text": text_src,
        "translated_text": text_tgt,
        "audio_base64": tts_out["audio_base64"]
    }


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":

    # -------------------------------------
    # Example 1 → ASR only
    # -------------------------------------
    run_asr("/home/ubuntu/s2s_service/hi.wav", "hi")

    # -------------------------------------
    # Example 2 → NMT only
    # -------------------------------------
    run_nmt(
        text="भारत एक महान देश है",
        src_lang="hi",
        tgt_lang="en"
    )

    # -------------------------------------
    # Example 3 → TTS only
    # -------------------------------------
    run_tts(
        text="Hello, welcome to Bhashini.",
        language="en"
    )

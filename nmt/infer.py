import os
import time
import logging
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from inference.engine import Model, iso_to_flores


# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("nmt_inference")


# =========================================================
# CONSTANTS
# =========================================================
INDIC_LANGUAGES = set(iso_to_flores.keys())
REQUIRED_MODELS = {"en-indic", "indic-en", "indic-indic"}


# =========================================================
# INFERENCE CLASS
# =========================================================
class NMTInference:

    # -----------------------------------------------------
    # Init → Load all models once
    # -----------------------------------------------------
    def __init__(self, checkpoint_root="./checkpoints"):

        self.models = {}
        self._checkpoint_root = checkpoint_root

        if not os.path.exists(checkpoint_root):
            raise RuntimeError(
                f"Checkpoint folder not found: {checkpoint_root}"
            )

        # indic-indic (direct language-to-language, e.g. hi->ta) is lazy-loaded
        # on first actual use instead of eagerly here. This deployment always
        # routes through EN (hi->EN->hi, see _route_translate), so indic-indic
        # is never touched in practice - eagerly loading it wasted ~300-500MB
        # (its ctranslate2 runtime footprint, on top of the 326MB model file)
        # on an 8GB board already running the LLM keep-alive-forever plus
        # ASR+TTS. That headroom is what stood between steady state and the
        # slow disk swapfile, which under load stalled the system long enough
        # to trip the systemd watchdog's 2-minute unresponsive timer and force
        # a hard reset - see the 14 Aug reboot investigation.
        for folder in ("en-indic", "indic-en"):

            model_path = os.path.join(
                checkpoint_root,
                folder,
                "ct2_int8_model"
            )

            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"Missing model path: {model_path}"
                )

            logger.info(f"Loading NMT model: {folder}")

            self.models[folder] = Model(
                model_path,
                device="cuda",
                input_lang_code_format="iso",
                model_type="ctranslate2"
            )

        if not self.models:
            raise RuntimeError("No NMT models loaded")

        logger.info("Required NMT models loaded successfully (indic-indic lazy)")

    def _get_indic_indic_model(self):
        if "indic-indic" not in self.models:
            model_path = os.path.join(
                self._checkpoint_root, "indic-indic", "ct2_int8_model"
            )
            if not os.path.exists(model_path):
                raise RuntimeError(f"Missing model path: {model_path}")
            logger.info("Lazy-loading NMT model: indic-indic")
            self.models["indic-indic"] = Model(
                model_path,
                device="cuda",
                input_lang_code_format="iso",
                model_type="ctranslate2"
            )
        return self.models["indic-indic"]

    # -----------------------------------------------------
    # Translation Routing
    # -----------------------------------------------------
    def _route_translate(self, text: str, src: str, tgt: str) -> str:

        if src in INDIC_LANGUAGES and tgt in INDIC_LANGUAGES:

            return self._get_indic_indic_model() \
                .paragraphs_batch_translate__multilingual(
                    [[text, src, tgt]]
                )[0]

        elif src in INDIC_LANGUAGES and tgt == "EN":

            return self.models["indic-en"] \
                .paragraphs_batch_translate__multilingual(
                    [[text, src, "en"]]
                )[0]

        elif src == "EN" and tgt in INDIC_LANGUAGES:

            return self.models["en-indic"] \
                .paragraphs_batch_translate__multilingual(
                    [[text, "en", tgt]]
                )[0]

        else:
            raise ValueError(
                f"Unsupported translation direction {src} → {tgt}"
            )

    # -----------------------------------------------------
    # Public Inference Method
    # -----------------------------------------------------
    def infer(self, text: str, src_lang: str, tgt_lang: str):

        start_time = time.time()

        # -------------------------
        # Validation
        # -------------------------
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 5000:
            raise ValueError("Text too long")

        # -------------------------
        # Translation
        # -------------------------
        translated = self._route_translate(
            text,
            src_lang,
            tgt_lang
        )

        logger.info(
            f"NMT {src_lang}→{tgt_lang} | "
            f"Time: {time.time() - start_time:.3f}s"
        )

        return {
            "translated_text": translated,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "processing_time_sec": round(
                time.time() - start_time, 3
            )
        }

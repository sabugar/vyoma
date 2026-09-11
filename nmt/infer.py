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

        # Pay CTranslate2's first-translate cost now, not on the user's first
        # question. Loading the model is not the same as running it: the first
        # real translate sets up the compute graph and thread pool and took
        # 5.9s measured, against 0.55s once warm. ASR and TTS already warm
        # themselves this way; NMT did not, so the very first question of a
        # session - the one a demo audience actually watches - paid the whole
        # penalty twice, once per direction. Warm through _route_translate,
        # the same entry point real requests use.

    def warm_up(self):
        """Pay CTranslate2's first-translate cost before any user request.

        Must be called AFTER every other engine has loaded. Loading a model is
        not the same as running it: the first real translate selects CUDA
        kernels for the shape it is given and took 4.9s measured, against
        0.5s once warm. Warming inside __init__ did not hold - TTS builds its
        own onnxruntime session afterwards, which disturbs the state this
        warms - so the first question of a session still paid the full cost.
        The text below is a realistic full sentence on purpose: a two-token
        warm-up ("नमस्ते") completed in 0.55s and warmed nothing useful.
        """
        logger.info("Warming up NMT (both directions)...")
        try:
            self._route_translate(
                "तुरंत जन्मे बच्चे को परिवार वाले नहलाना चाहते हैं तो "
                "आशा को क्या सलाह देनी चाहिए और किन बातों का ध्यान रखना चाहिए",
                "hi", "EN")
            self._route_translate(
                "If the baby weighs less than two thousand grams it should not "
                "be bathed. Dry the baby, keep it warm, and place it against "
                "the mother's skin until a health worker can be reached.",
                "EN", "hi")
            logger.info("NMT warm-up complete")
        except Exception:
            # A warm-up failure must never stop the service from starting;
            # the first real request simply pays the cost instead.
            logger.exception("NMT warm-up failed, continuing")

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
    # English arrives as "EN" from this deployment, but a caller that sends
    # "en" would fall through the EN branches below and be routed to the
    # indic-indic model instead - silently lazy-loading 326MB that this board
    # deliberately keeps unloaded, plus ~5s on that request. Normalise once,
    # here, so no caller can trip it by case alone.
    @staticmethod
    def _norm_lang(code: str) -> str:
        return "EN" if str(code).strip().lower() == "en" else str(code).strip()

    def _route_translate(self, text: str, src: str, tgt: str) -> str:

        src = self._norm_lang(src)
        tgt = self._norm_lang(tgt)

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

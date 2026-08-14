#!/usr/bin/env python
import base64

from asr.infer import ASRInference
from nmt.infer import NMTInference
from tts.infer import TTSInference

import logging
import flask
from flask import request
import argparse
import os
import psutil
import gc
import ctypes

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


class BashiniAPIServer:
    def __init__(self, app=None, logger=None):
        self.logger = logger if logger is not None else logging.getLogger("BhashiniAPIServer")
        self.app = app if app is not None else flask.Flask(__name__)

        self.logger.info("Initializing Inference Engines...")

        process = psutil.Process(os.getpid())

        ram_base = process.memory_info().rss / (1024 ** 2)

        self.asr_engine = ASRInference(
            checkpoint_dir="asr/checkpoints"
        )
        
        ram_asr = process.memory_info().rss / (1024 ** 2)

        self.nmt_engine = NMTInference(
            checkpoint_root="nmt/checkpoints"
        )

        ram_nmt = process.memory_info().rss / (1024 ** 2)

        self.tts_engine = TTSInference()

        ram_tts = process.memory_info().rss / (1024 ** 2)
        self.logger.info(f"Baseline RAM Usage: {ram_base:.2f} MB")
        self.logger.info(f"ASR RAM Usage: {ram_asr-ram_base:.2f} MB")
        self.logger.info(f"NMT RAM Usage: {ram_nmt-ram_asr:.2f} MB")
        self.logger.info(f"TTS RAM Usage: {ram_tts-ram_nmt:.2f} MB")
        self.logger.info(f"Total RAM Usage: {ram_tts:.2f} MB")

        self.logger.info("All Engines Loaded Successfully.")

        self.app.add_url_rule("/asr", "asr", self.handle_asr, methods=["POST"])
        self.app.add_url_rule("/nmt", "nmt", self.handle_nmt, methods=["POST"])
        self.app.add_url_rule("/tts", "tts", self.handle_tts, methods=["POST"])
        self.app.add_url_rule("/s2s", "s2s", self.handle_s2s, methods=["POST"])
        self.app.add_url_rule("/health", "health", self.health, methods=["GET"])
        self.app.after_request(self._trim_memory)

    def _trim_memory(self, response):
        # ASR/NMT/TTS each allocate short-lived numpy/ctranslate2/onnxruntime
        # buffers per request. CPython's allocator and glibc's malloc both
        # tend to hold freed heap pages rather than returning them to the OS,
        # so RSS creeps up across many requests even with no real leak.
        # Forcing a collect + malloc_trim after every request returns that
        # freed memory - this board has no RAM to spare for "free later".
        if request.path != "/health":
            gc.collect()
            if _libc is not None:
                _libc.malloc_trim(0)
        return response
    
    @classmethod
    def create_app(cls):
        server = BashiniAPIServer()
        return server.app

    def health(self):
        return flask.jsonify({
            "result": "success"
        }), 200

    def handle_asr(self):
        data = flask.request.get_json()
        audio_b64 = data.get("audio_base64")
        language = data.get("language")

        if not audio_b64 or not language:
            return flask.jsonify({"error": "Missing 'audio_base64' or 'language' in request"}), 400

        result = self.run_asr(audio_b64, language)
        return flask.jsonify(result)

    def handle_nmt(self):
        data = flask.request.get_json()
        text = data.get('text')
        src_lang = data.get('src_lang')
        tgt_lang = data.get('tgt_lang')

        if not text or not src_lang or not tgt_lang:
            return flask.jsonify({"error": "Missing 'text' or 'src_lang' or 'tgt_lang' in request"}), 400

        result = self.run_nmt(text, src_lang, tgt_lang)
        return flask.jsonify(result)

    def handle_tts(self):
        data = flask.request.get_json()
        text = data.get('text')
        language = data.get("language")

        if not text or not language:
            return flask.jsonify({"error": "Missing 'text' or 'language' in request"}), 400

        result = self.run_tts(text, language)
        return flask.jsonify(result)

    def handle_s2s(self):
        data = flask.request.get_json()
        audio_b64 = data.get("audio_base64")
        src_lang = data.get('src_lang')
        tgt_lang = data.get('tgt_lang')

        if not audio_b64 or not src_lang or not tgt_lang:
            return flask.jsonify({"error": "Missing 'audio_base64' or 'src_lang' or 'tgt_lang' in request"}), 400

        result = self.run_s2s_pipeline(audio_b64, src_lang, tgt_lang)
        return flask.jsonify(result)

    def run_asr(self, audio_b64: str, language: str):
        self.logger.info(f"Running ASR → {language}")

        result = self.asr_engine.infer(
            audio_base64=audio_b64,
            language=language
        )

        self.logger.debug("ASR Output:", result["text"])

        return result

    def run_nmt(self, text: str, src_lang: str, tgt_lang: str):
        self.logger.info(f"Running NMT → {src_lang} → {tgt_lang}")

        result = self.nmt_engine.infer(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )

        self.logger.debug("NMT Output:", result["translated_text"])

        return result

    def run_tts(self, text: str, language: str):
        self.logger.info(f"Running TTS → {language}")

        result = self.tts_engine.infer(
            text=text,
            language=language,
            return_base64=True
        )

        # self.logger.debug(f"TTS Audio generated {len(result['output_base64'])} bytes")

        return result

    def run_s2s_pipeline(
        self,
        audio_b64: str,
        src_lang: str,
        tgt_lang: str
    ):
        self.logger.info("Running Speech-to-Speech Pipeline...")

        # ASR
        asr_out = run_asr(audio_b64, src_lang)
        text_src = asr_out["text"]

        # NMT
        nmt_out = run_nmt(text_src, src_lang, tgt_lang)
        text_tgt = nmt_out["translated_text"]

        # TTS
        tts_out = run_tts(text_tgt, tgt_lang, return_base64=True)

        return {
            "asr_text": text_src,
            "translated_text": text_tgt,
            "audio_base64": tts_out["audio_base64"]
        }

def cli():
    parser = argparse.ArgumentParser(description="Bhashini API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=11400, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run the server in debug mode", default=False)
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:        
        logging.basicConfig(level=logging.INFO)

    app = BashiniAPIServer.create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    cli()

import threading
import logging
import importlib


class BaseApplication:
    METADATA = {} # Will be overridden by ApplicationRegistry decorator

    def __init__(self, board, settings=None):
        self.logger = logging.getLogger(__name__)
        self.settings = self.METADATA.get("default_settings", {})
        if settings is not None:
            self.settings.update(settings)
        self.board = board
        self.thread = threading.Thread()
        self.running = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        self.logger.info("Startup")
        while self.running:
            try:
                self.run()
            except KeyboardInterrupt:
                self.logger.info("Exit")
                self.board.clear_screen()
                self.running = False
            except Exception as e:
                self.logger.exception("Error in application run loop: %s", e)

    def run(self):
        raise NotImplementedError()
    
    def stop(self):
        self.running = False
        self.thread.join()

    @classmethod
    def verify_dependencies(cls, settings=None):
        # Verify that all service dependencies are available
        settings = settings or {}
        input_language = settings.get("input_language", "en")
        output_language = settings.get("output_language", "en")
        english_not_needed = input_language != "en" and output_language != "en"
        models = cls.METADATA.get("models", {})
        for model in models:
            if english_not_needed and model in ("piper", "vosk"):
                # Piper/Vosk are only used on the English path (see e.g. HearTheWorld.start()).
                # Verifying them here would eagerly load a throwaway ONNX/model session on every
                # startup even in non-English mode, costing memory the LLM needs.
                logging.debug(f"Skipping model dependency verification for {model}: non-English settings")
                continue
            logging.debug(f"Verifying model dependency: {model}")
            package = importlib.import_module(f"pocketinfer.models.{model}")
            model_class = getattr(package, model.capitalize())
            verify_func = getattr(model_class, "verify")
            success, message = verify_func(models[model])
            if not success:
                # Try to manually update
                update_func = getattr(model_class, "update")
                update_func(models[model])
                # Re-verify
                success, message = verify_func(models[model])
                if not success:
                    raise RuntimeError(f"Model dependency verification failed for {model}: {message}")
        return True
    @classmethod

    def update_dependencies(cls):
        # Verify that all service dependencies are available
        models = cls.METADATA.get("models", {})
        for model in models:
            logging.debug(f"Updating model dependency: {model}")
            package = importlib.import_module(f"pocketinfer.models.{model}")
            model_class = getattr(package, model.capitalize())
            update_func = getattr(model_class, "update")
            update_func(models[model])
        return True

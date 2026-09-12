import requests
import time
import subprocess
from subprocess import check_output


class Tts:
    def __init__(self):
        pass

    def infer(self, text: str, language: str):
        payload = {
            "text": text,
            "language": language
        }

        response = requests.post("http://localhost:11400/tts", json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"TTS inference failed: {response.text}")

    @classmethod
    def verify(cls, args):
        # For TTS, we can do a simple health check by sending a test synthesis request
        try:
            response = requests.get("http://localhost:11400/health")
            if response.status_code == 200:
                return True, "TTS service is available."
        except requests.exceptions.ConnectionError:
            print("Connection Error, trying to launch model")
        # bhashini_models can already be starting (e.g. via systemd After=/Wants=
        # ordering) - only force a restart if it isn't already active/activating,
        # otherwise this resets an in-progress startup and makes it slower.
        state = subprocess.getoutput(
            'systemctl is-active bhashini_models.service'
        ).strip()
        if state not in ("active", "activating"):
            check_output('systemctl restart bhashini_models.service', shell=True)
        start = time.time()
        # The Hindi ASR/TTS models are large (600M param ASR, VITS TTS) and can
        # take well over 60s to load from a cold service start.
        last_status = None
        while time.time() - start < 120.0:
            try:
                response = requests.get("http://localhost:11400/health")
                last_status = response.status_code
                if response.status_code == 200:
                    return True, "TTS service is available."
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(0.25)
        return False, f"TTS service responded with status code {last_status}."
        
    @classmethod
    def update(cls, args):
        return True, "OK"

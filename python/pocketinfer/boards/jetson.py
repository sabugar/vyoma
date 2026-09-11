import time
import queue
import os

from pocketinfer.serialcomms import IOInterface
from pocketinfer.boards.base import Board
from multiprocessing import Process, Queue, Pipe, set_start_method

import Jetson.GPIO as GPIO

from threading import Thread

# NOTE: pocketinfer.ui.handheld is imported lazily inside
# PocketInferDevboardUI.__init__ (only when the screen is actually enabled).
# Importing it at module level pulls in displayio, which starts a background
# refresh thread that spins a full CPU core even with no display attached.


class NullUI:
    '''No-op UI for headless devboards (no ILI9341 screen attached).
    Every method call succeeds silently so application code can call
    statusbar()/top_text()/etc. unconditionally.'''
    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return True
        return noop

# import time
# import displayio
# import digitalio
# import board
# import fourwire
# import adafruit_ili9341
# import xpt2046_circuitpython as xpt2046


class PocketInferDevboard(Board):
    V4L_CAMERA_NAME = 'Arducam_8mp'
    ALSA_CAPTURE_NAME = 'Arducam_8mp'
    ALSA_PLAYBACK_NAME = 'USB Audio Device'
    TRIGGER_BOARD_IDX = 'GP167'  # Physical pin 7 on header

    def __init__(self, args):
        super().__init__(args)
        # Circuitpython modules may have already initialized in TEGRA_SOC mode.
        GPIO.setmode(GPIO.TEGRA_SOC)
        GPIO.setup(self.TRIGGER_BOARD_IDX, GPIO.IN)
        GPIO.add_event_detect(self.TRIGGER_BOARD_IDX, GPIO.BOTH, callback=self.trig_cb, bouncetime=100)

    def trig_cb(self, channel):
        if GPIO.input(self.TRIGGER_BOARD_IDX):
            self.trigger_button = True
            self.trigger_button_down.set()
            self.logger.debug("Trigger button down")
        else:
            self.trigger_button = False
            self.trigger_button_up.set()
            self.logger.debug("Trigger button up")

    # Recording ends when the speaker falls silent, not when the button is
    # released. This board's trigger pin cannot report a sustained press:
    # measured repeatedly with raw polling (no event detection), it goes HIGH
    # on press and drops LOW after ~7.8s (7.76 / 7.78 / 7.79 / 7.81 / 7.90 /
    # 7.90 / 7.91s) while the button is still physically held, and never
    # returns HIGH. Jetson.GPIO's pull_up_down does not change this - setting
    # PUD_UP left the idle level at LOW, i.e. the library's pull request is
    # ignored on this platform, as NVIDIA's own docs note.
    #
    # So hold-to-talk is not implementable here, and questions were being cut
    # off at 7.9s - losing the negation in "...डॉक्टर या नर्स मौजूद नहीं है"
    # and inverting the answer with it. Press once, speak, stop talking.
    # 2.0s cut people off mid-question: a normal breath between sentences
    # ("...सांस भी कमजोर चल रही है" <pause> "डॉक्टर या नर्स मौजूद नहीं है")
    # was read as the end, and the half-question that reached the LLM had no
    # question in it at all. Natural sentence-boundary pauses run 0.5-1.5s, so
    # 3.5s clears them while still ending promptly when the speaker is done.
    SILENCE_STOP_S = 3.5      # quiet this long ends the question
    SPEECH_GRACE_S = 4.0      # allow this long to start speaking at all
    MAX_RECORD_S = 60.0       # safety bound
    CALIBRATE_S = 0.5         # ambient noise sampled from the start
    POLL_S = 0.05

    def _chunk_levels(self, frames):
        """RMS per recorded chunk, as float32 to avoid int16 overflow."""
        import numpy as np
        levels = []
        for chunk in frames:
            arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            if arr.size:
                levels.append(float(np.sqrt(np.mean(arr * arr))))
        return levels

    def wait_for_trigger_button_up(self, timeout=None):
        '''Block until the speaker has stopped talking (or the button is
        pressed again, which still works as a manual stop).'''
        import numpy as np

        recorder = self.audio
        chunk_s = recorder.frames_per_buffer / float(recorder.rate)
        quiet_needed = max(1, int(round(self.SILENCE_STOP_S / chunk_s)))
        calib_chunks = max(1, int(round(self.CALIBRATE_S / chunk_s)))

        start = time.monotonic()
        deadline = start + min(self.MAX_RECORD_S,
                               timeout if timeout else self.MAX_RECORD_S)
        threshold = None
        speech_seen = False

        # The press that started the recording is still settling; ignore it so
        # it is not read as the manual stop.
        time.sleep(0.35)
        self.trigger_button_down.clear()

        while True:
            now = time.monotonic()
            if now >= deadline:
                self.logger.debug("Recording hit max duration")
                return
            if self.trigger_button_down.is_set():
                self.logger.debug("Manual stop press")
                return

            levels = self._chunk_levels(list(recorder.frames))

            if threshold is None and len(levels) >= calib_chunks:
                ambient = float(np.median(levels[:calib_chunks]))
                # Speech has to clear the room noise by a clear margin; the
                # floor keeps a silent room from setting a hair-trigger.
                threshold = max(ambient * 3.0, 250.0)
                self.logger.debug(
                    "Silence threshold %.0f (ambient %.0f)", threshold, ambient)

            if threshold is not None and levels:
                recent = levels[-quiet_needed:]
                if max(levels) > threshold:
                    speech_seen = True
                if speech_seen and len(recent) >= quiet_needed and \
                        max(recent) <= threshold:
                    self.logger.debug(
                        "Silence for %.1fs, ending recording", self.SILENCE_STOP_S)
                    return
                if not speech_seen and (now - start) > self.SPEECH_GRACE_S:
                    self.logger.debug("No speech detected within grace period")
                    return

            time.sleep(self.POLL_S)

class PocketInferDevboardUI(PocketInferDevboard):
    TOUCH_IRQ_BOARD_IDX = 'GP37_SPI3_MISO'  # Physical pin 22 on header
    ALSA_PLAYBACK_NAME = 'UACDemo'
    ALSA_PLAYBACK_CHANNEL_NAME = "PCM"

    def __init__(self, args):
        super().__init__(args)
        # This deployment's devboard has NO ILI9341 screen attached. Spawning
        # the display UI subprocess anyway wastes a full CPU core spinning in
        # displayio's refresh loop plus its memory - on an 8GB board running
        # LLM+ASR+TTS that pressure thrashes swap and hangs the whole system.
        # Headless is now the default; set POCKETINFER_UI=1 in the service
        # environment to re-enable the real screen UI.
        if os.environ.get('POCKETINFER_UI', '0') != '1':
            self.UI = NullUI()
            self.button_queue = None
            self._ui = None
            self.ui_thread_running = False
            return
        from pocketinfer.ui.handheld import IlI9341HandheldUI, ILI9341UIConfig
        cfg = ILI9341UIConfig(
            reset_pin = 'GP36_SPI3_CLK',
            pwm_pin = 'GP122',
            cs_pin = 'GP50_SPI1_CS0_N',
            dc_pin = 'GP88_PWM1',
            touch_cs = 'GP51_SPI1_CS1_N',
            touch_irq = 'GP37_SPI3_MISO',
            display_baudrate = 30000000,
            touch_baudrate = 1000000,
            width = 320,
            height = 240,
            rotation = 90
        )
        set_start_method('forkserver')
        self.parent_conn, child_conn = Pipe()
        self.button_queue = Queue()
        self._ui = Process(target=IlI9341HandheldUI.multiprocess_launch, args=(cfg, child_conn, self.button_queue))
        self._ui.start()
        self.ui_thread = Thread(target=self._process_ui_events, daemon=True)
        self.ui_thread_running = True
        self.ui_thread.start()
        self.UI = IlI9341HandheldUI.get_remote(self.parent_conn)

    def _process_ui_events(self):
        while self.ui_thread_running:
            try:
                name = self.button_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            for cb in self.ui_cbs:
                try:
                    cb(name)
                except:
                    self.logger.exception("Error in UI callback")
                    continue

    def touch_cb(self, channel):
        self.logger.debug("Touch IRQ")
        self.UI.check_touch()

    def clear_screen(self):
        self.UI.clear_screen()

    def statusbar(self, text):
        self.UI.statusbar_text(text)
        return True

    def top_text(self, text):
        self.UI.top_text(text)
        return True

    def bottom_text(self, text):
        self.UI.bottom_text(text)
        return True

    def mode_text(self, text):
        self.UI.mode_text(text)
        return True

    def memory_text(self, text):
        self.UI.memory_text(text)
        return True

class RaspiAIHat2Board(Board):
    V4L_CAMERA_NAME = 'Arducam_8mp'
    ALSA_CAPTURE_NAME = 'Arducam_8mp'
    ALSA_PLAYBACK_NAME = 'USB Audio Device'
    TRIGGER_BOARD_IDX = 7

    def __init__(self, args):
        super().__init__(args)

class PocketInferDemo(Board):
    V4L_CAMERA_NAME = 'Arducam_8mp'
    ALSA_CAPTURE_NAME = 'USB PnP Sound Device'
    ALSA_CAPTURE_RATE = 44100
    ALSA_PLAYBACK_NAME = 'USB Audio Device'

    def __init__(self, args):
        super().__init__(args)
        self.ioexp = IOInterface()
        self.ioexp.subscribe(self.ioexp_cb)
        self.ioexp.open()
        self.clear_screen()
        self.statusbar("Loading...")

    def ioexp_cb(self, msg):
        if msg == 'BT0':
            self.trigger_button = True
            self.trigger_button_down.set()
            self.logger.debug("Trigger button down")
        elif msg == 'BT1':
            self.trigger_button = False
            self.trigger_button_up.set()
            self.logger.debug("Trigger button up")
        elif msg == 'dOK':
            pass
        elif msg.startswith('C'):
            for cb in self.ui_cbs:
                try:
                    cb(msg[1:])
                except:
                    pass
        else:
            self.logger.debug("RX: "+msg)

    def button_led(self, value):
        if value:
            return self.ioexp.transact('l1') == 'lOK'
        else:
            return self.ioexp.transact('l0') == 'lOK'
        
    def rgb_led(self, r, g=None, b=None):
        if isinstance(r, str):
            if r == 'off' or r == 'black':
                r,g,b = (0,0,0)
            elif r == 'on' or r == 'white':
                r,g,b = (255,255,255)
            elif r == 'red':
                r,g,b = (255,0,0)
            elif r == 'green':
                r,g,b = (0,255,0)
            elif r == 'blue':
                r,g,b = (0,0,255)
            elif r == 'yellow':
                r,g,b = (255,200,0)
            elif r == 'purple':
                r,g,b = (100,0,255)
            elif r == 'cyan':
                r,g,b = (0,255,255)
            elif r== 'orange':
                r,g,b = (255,75,0)
        elif g is None or b is None:
            raise SyntaxError("If color is not specified, all three r,g,b values must be specified")

        if isinstance(r,float):
            r = int(r*255.0)
        if isinstance(g,float):
            g = int(g*255.0)
        if isinstance(b,float):
            b = int(b*255.0)

        if isinstance(r,bool):
            r = r*255
        if isinstance(g,bool):
            g = g*255
        if isinstance(b,bool):
            b = b*255

        return self.ioexp.transact(f'L{r},{g},{b}') == 'LOK'

    def clear_screen(self):
        self.ioexp.ser.write('''
        a0
        TT
        TB
        TS 
        TM
        tm
        '''.encode('utf-8'))

    def led_animation(self, val):
        return self.ioexp.transact(f'a{int(val)}') == 'aOK'

    def statusbar(self, text):
        return self.ioexp.transact(f'TS{text}') == 'TSOK'

    def top_text(self, text):
        return self.ioexp.transact(f'TT{text}') == 'TTOK'
    
    def bottom_text(self, text):
        return self.ioexp.transact(f'TB{text}') == 'TBOK'

    def mode_text(self, text):
        return self.ioexp.transact(f'TM{text}') == 'TMOK'

    def memory_text(self, text):
        return self.ioexp.transact(f'Tm{text}') == 'TmOK'

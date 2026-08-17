from os.path import exists
from os import system
from subprocess import run
from glob import glob
import threading
import logging
import cv2
import os
import re
from pocketinfer import audio


class CameraIterable:
    def __init__(self, board):
        self.board = board
    def __iter__(self):
        return self
    def __next__(self):
        frame = self.board.camera_frame()
        if frame is None:
            raise StopIteration
        return frame

class CameraReader:
    def __init__(self, camera_name='', camera_interface='usb', width=1280, height=720):
        self.logger = logging.getLogger(__name__)
        self.camera_name = camera_name
        self.camera_interface = camera_interface
        self.camera_idx = None
        self.width = width
        self.height = height
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.frame = None
        self.running = False
        self.frame_available = threading.Event()
        self.cap = None
        self.ready = threading.Event()
        self.error = None

    def start(self):
        self.running = True
        self.thread.start()
        if not self.ready.wait(timeout=20):
            self.running = False
            raise RuntimeError("Camera thread did not become ready within 20s")
        if self.error is not None:
            self.running = False
            raise self.error
    
    def _candidate_devices(self):
        """All /dev/videoN nodes for this camera, lowest by-id index first."""
        found = []
        for filename in sorted(glob('/dev/v4l/by-id/*')):
            match = re.match(r'(\S+)\-(\S+)\-\S+\-index(\d+)', filename)
            if match is None:
                continue
            interface, name, idx = match.groups()
            if self.camera_interface in interface and self.camera_name in name:
                found.append((int(idx), os.path.realpath(filename)))
        found.sort()
        return [path for _, path in found]

    def _open_capture(self):
        """Open the first node that actually yields a frame (skips metadata nodes)."""
        devices = self._candidate_devices()
        if not devices:
            self.logger.warning(
                "Camera '%s' (interface '%s') not found, defaulting to /dev/video0",
                self.camera_name, self.camera_interface)
            devices = ['/dev/video0']
        for device in devices:
            self.logger.debug("Trying camera device %s", device)
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                self.logger.debug("%s did not open, skipping", device)
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ret, frame = cap.read()
            if not ret or frame is None:
                self.logger.debug("%s opened but gave no frame, skipping", device)
                cap.release()
                continue
            self.logger.info("Using camera device %s", device)
            self.camera_idx = device
            return cap
        raise RuntimeError(
            "No working capture device for camera '%s' (tried: %s)"
            % (self.camera_name, ', '.join(devices)))

    def _run(self):
        try:
            self.cap = self._open_capture()
        except Exception as exc:
            self.error = exc
            self.ready.set()
            self.running = False
            return
        self.ready.set()
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                self.frame = frame
                self.frame_available.set()
        finally:
            self.running = False
            self.cap.release()

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class Board:
    V4L_CAMERA_NAME = ''
    V4L_CAMERA_INTERFACE = 'usb'
    ALSA_CAPTURE_NAME = ''
    ALSA_PLAYBACK_NAME = ''
    ALSA_CAPTURE_CHANNEL_NAME = "Mic"
    ALSA_PLAYBACK_CHANNEL_NAME = "Speaker"
    ALSA_DEVNAME_BLACKLIST = ['NVIDIA Jetson Orin Nano APE']

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.args = args
        self.trigger_button = False
        self.trigger_button_down = threading.Event()
        self.trigger_button_up = threading.Event()
        self.camera = CameraReader(
            camera_name=self.V4L_CAMERA_NAME,
            camera_interface=self.V4L_CAMERA_INTERFACE
        )
        # Select audio capture device - try to use ALSA_CAPTURE_NAME if provided, apply blasklist, fall back on any available capture device
        capture_devices = audio.alsa_devices_filtered(record=True, blacklist=self.ALSA_DEVNAME_BLACKLIST)
        capture_device = None
        if self.ALSA_CAPTURE_NAME != '':
            for dev in capture_devices:
                if self.ALSA_CAPTURE_NAME in dev['name']:
                    capture_device = dev
                    break
        if capture_device is None and len(capture_devices) > 0:
            self.logger.warning(f"Capture device '{self.ALSA_CAPTURE_NAME}' not found, defaulting to first available device '{capture_devices[0]['name']}'")
            capture_device = capture_devices[0]
        if capture_device is None:
            self.logger.error(f"No capture devices found, please check your audio input device")

        # Select audio playback device - try to use ALSA_PLAYBACK_NAME if provided, apply blasklist, fall back on any available capture device
        playback_devices = audio.alsa_devices_filtered(playback=True, blacklist=self.ALSA_DEVNAME_BLACKLIST)
        playback_device = None
        if self.ALSA_PLAYBACK_NAME != '':
            for dev in playback_devices:
                if self.ALSA_PLAYBACK_NAME in dev['name']:
                    playback_device = dev
                    break
        if playback_device is None and len(playback_devices) > 0:
            self.logger.warning(f"Playback device '{self.ALSA_PLAYBACK_NAME}' not found, defaulting to first available device '{playback_devices[0]['name']}'")
            playback_device = playback_devices[0]
        if playback_device is None:
            self.logger.error(f"No playback devices found, please check your audio output device")

        self.logger.debug('Capture device: %s, Playback device: %s', capture_device, playback_device)
        self.audio = audio.AudioRecorder(device_idx=capture_device['index'] if capture_device else 0, frames_per_buffer=4096)
        self.alsa_capture_card = capture_device['alsa_card'] if capture_device else None 
        self.alsa_playback_card = playback_device['alsa_card'] if playback_device else None
        _alsa_playback_device = playback_device['alsa_device'] if playback_device else None
        self.alsa_playback_device = f'hw:{self.alsa_playback_card},{_alsa_playback_device}'
        # Try to crank up volume on recording and playback devices
        audio.set_volume(self.alsa_capture_card, 100)
        audio.set_volume(self.alsa_playback_card, 100)
        self.ui_cbs = []

    def subscribe_to_ui(self, func):
        if func not in self.ui_cbs:
            self.ui_cbs.append(func)

    def unsubscribe_to_ui(self, func):
        if func in self.ui_cbs:
            self.ui_cbs.remove(func)
    
    def wait_for_trigger_button_down(self, timeout=None):
        self.trigger_button_down.clear()
        self.trigger_button_down.wait(timeout=timeout)
    
    def wait_for_trigger_button_up(self, timeout=None):
        self.trigger_button_up.clear()
        self.trigger_button_up.wait(timeout=timeout)

    def camera_frame(self):
        if not self.camera.running:
            self.camera.frame_available.clear()
            self.camera.start()
            self.camera.frame_available.wait(timeout=5.0)
        return self.camera.frame

    def camera_frames(self):
        return CameraIterable(self)

    def camera_frame_jpg(self):
        frame = self.camera_frame()
        if frame is None:
            return None
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            return None
        return bytearray(buffer)

    @classmethod
    def get_board(cls):
        args = {}
        if not exists('/proc/device-tree/model'):
            raise NotImplementedError('/proc/device-tree not found: Must be a linux system with modern kernel >4')
        with open('/proc/device-tree/model', 'r') as fil:
            devicetree_model = fil.read().replace('\x00', '').strip()
        if devicetree_model.startswith('NVIDIA'):
            # nv_tegra_release will only be present on NVIDIA platforms, possibly only JetPack
            if not exists('/etc/nv_tegra_release'):
                raise NotImplementedError("Only NVIDIA Tegra platforms supported "+devicetree_model)
            with open('/etc/nv_tegra_release', 'r') as fil:
                args['kernelinfo'] = fil.readline()
            # Read EEPROM data from the module and carrier board. These i2c EEPROMs should be available on all Jetson platforms
            module_ver_raw = run(['i2ctransfer', '-f', '-y', '0', 'w1@0x50', '0x14', 'r22@0x50'], capture_output=True, text=True)
            if module_ver_raw.stderr:
                raise NotImplementedError('Cannot detect nvidia platform - Error reading module eeprom: '+module_ver_raw.stderr)
            module_ver = bytearray([int(x,16) for x in module_ver_raw.stdout.split(' ')])
            carrier_ver_raw = run(['i2ctransfer', '-f', '-y', '0', 'w1@0x57', '0x14', 'r22@0x57'], capture_output=True, text=True)
            if carrier_ver_raw.stderr:
                raise NotImplementedError('Cannot detect nvidia platform - Error reading module eeprom: '+carrier_ver_raw.stderr)
            carrier_ver = bytearray([int(x,16) for x in carrier_ver_raw.stdout.split(' ')])
            if not module_ver.startswith(b'699-13767-0005'):
                raise NotImplementedError('Unsupported Jetson module: '+module_ver.decode('utf-8'))
            args['module_ver'] = module_ver
            args['carrier_ver'] = carrier_ver 
            # Load the correct board based on the carrier board
            if carrier_ver.startswith(b'699-13768-0000'):
                # We could also instantiate a PocketInferDevboard, which has no UI 
                # But automatic detection of the screen is a challenge
                from pocketinfer.boards.jetson import PocketInferDevboardUI
                return PocketInferDevboardUI(args)
            if carrier_ver.startswith(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'):
                # The seeeedstudio carrier board has an eeprom present but zero-ed out memory
                from pocketinfer.boards.jetson import PocketInferDemo
                return PocketInferDemo(args)
            raise NotImplementedError('Unsupported Carrier Board: '+carrier_ver.decode('utf-8'))
        else:
            raise NotImplementedError('Unsupported linux platform: '+devicetree_model)

    # To be overridden, ideally
    def button_led(self, value) -> bool:
        return True
        
    def rgb_led(self, r, g=None, b=None) -> bool:
        return True

    def led_animation(self, val) -> bool:
        return True

    def clear_screen(self):
        return

    def statusbar(self, text) -> bool:
        self.logger.info("Statusbar: "+text)
        return True

    def top_text(self, text) -> bool:
        self.logger.info("Top text: "+text)
        return True
    
    def bottom_text(self, text) -> bool:
        self.logger.info("Bottom text: "+text)
        return True

    def mode_text(self, text) -> bool:
        self.logger.info("Mode text: "+text)
        return True

    def memory_text(self, text) -> bool:
        return True

class DummyBoard(Board):
    def __init__(self, args):
        super().__init__(args)
        self.logger.info("Using DummyBoard - no hardware features will work")
        self.audio = audio.DummyAudioRecorder(args['audio_file'])

    def wait_for_trigger_button_down(self, timeout=None):
        self.trigger_button_down.clear()
        return
    
    def wait_for_trigger_button_up(self, timeout=None):
        self.trigger_button_up.clear()
        return
    
    def camera_frame(self):
        if 'image_file' not in self.args:
            return None
        img = self.args.get('image_file')
        if isinstance(img, str):
            if not exists(img):
                raise FileNotFoundError(f"DummyBoard image file '{img}' not found")
            return cv2.imread(img)
        if isinstance(img, bytes):
            return cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
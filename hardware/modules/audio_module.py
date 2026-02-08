import os
import sys
import queue
import logging
import sounddevice as sd
import subprocess

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    vosk = None

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

class AudioModule:
    def __init__(self, sample_rate=16000, device=None, model_path="hardware/models/vosk-model-small-en-us-0.15"):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.samplerate = sample_rate
        self.q = queue.Queue()
        
        # Check Vosk Model
        if VOSK_AVAILABLE:
            if not os.path.exists(model_path):
                self.logger.warning(f"VOSK model not found at {model_path}. Please download it.")
                self.model = None
            else:
                try:
                    self.model = vosk.Model(model_path)
                    self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
                except Exception as e:
                    self.logger.error(f"Failed to load Vosk model: {e}")
                    self.model = None
        else:
             self.logger.warning("Vosk library not found (PC Mode). STT disabled.")
             self.model = None

        # Initialize TTS Engine
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150) # Speed
            except Exception as e:
                self.logger.error(f"Failed to init TTS: {e}")
                self.tts_engine = None
        else:
            self.logger.warning("pyttsx3 library not found. TTS disabled.")
            self.tts_engine = None

    def _callback(self, indata, frames, time, status):
        """Streaming callback for sounddevice"""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def listen_continuously(self):
        """
        Generator function to yield recognized text in real-time.
        Use this in the main loop explicitly.
        """
        if not self.model:
            yield ""
            return

        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000, device=self.device, dtype='int16',
                               channels=1, callback=self._callback):
            while True:
                data = self.q.get()
                if self.recognizer.AcceptWaveform(data):
                    res = self.recognizer.Result()
                    text = eval(res)['text']
                    if text:
                        yield text
                else:
                    # Partial result
                    pass

    def speak(self, text):
        """
        Speak text using TTS engine.
        Route audio via Bluetooth if connected.
        """
        if self.is_bluetooth_connected():
            self.logger.info("Routing audio to Bluetooth device...")
            # Ideally switch pulseaudio sink here, but espeak usually follows system default sink
            pass
        
        if self.tts_engine:
            self.logger.info(f"Speaking: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        else:
            self.logger.warning("TTS Engine not available.")

    def is_bluetooth_connected(self):
        """
        Check if a Bluetooth audio device is connected using bluetoothctl
        """
        try:
            output = subprocess.check_output(["bluetoothctl", "info"]).decode("utf-8")
            if "Connected: yes" in output and "Audio Sink" in output: # Simplistic check
                return True
        except Exception:
            pass
        return False

import queue
import sys
import sounddevice as sd
import json

from typing import List, Callable
from vosk import Model, KaldiRecognizer

class SpeechRecognizer:
    """Input class for handling audio input"""

    def __init__(self) -> None:
        self.q = queue.Queue()
        try:
            default_input_device = sd.default.device[0]
            device_info = sd.query_devices(default_input_device, "input")
            sample_rate = int(device_info["default_samplerate"])
            
            self.model = Model(lang="en-us")
            self._input_stream = sd.RawInputStream(samplerate=sample_rate, blocksize=8000, dtype="int16", channels=1, callback=self._callback)
            self.rec = KaldiRecognizer(self.model, sample_rate)
            self._listen = False
            self._capturing = False
        except Exception: pass

    def _int_or_str(self, text):
        """Helper function for argument parsing."""
        try:
            return int(text)
        except ValueError:
            return text

    def _callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def is_active(self):
        return self._listen

    def kill(self):
        try:
            self._listen = False
            self._input_stream.abort()
        except Exception: pass

    def is_capturing(self):
        return self._capturing

    def start(self, subscribers: List[Callable]):
        self._listen = True
        self._input_stream.start()
        while self._listen:
            try:
                data = self.q.get()
                if not data:
                    self._capturing = False
                    continue

                if self.rec.AcceptWaveform(data):
                    d = self.rec.Result()
                    parsed = json.loads(d)
                    text = parsed.get("text")
                    if not text:
                        self._capturing = False
                        continue

                    for sub in subscribers:
                        sub(text)
                else:
                    d = self.rec.PartialResult()
                    parsed = json.loads(d)
                    if not parsed:
                        self._capturing = False
                        continue
                    if not parsed.get("partial"):
                        self._capturing = False
                        continue
                    text = parsed.get("partial")
                    if not text:
                        self._capturing = False
                        continue
                    self._capturing = True
            except Exception:
                continue

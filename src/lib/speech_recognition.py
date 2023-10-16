import sys
import sounddevice as sd
import json
import asyncio

from typing import List, Callable
from vosk import Model, KaldiRecognizer

class SpeechRecognizer:
    """Input class for handling audio input"""

    def __init__(self) -> None:
        self.q = asyncio.Queue()
        default_input_device = sd.default.device[0]
        device_info = sd.query_devices(default_input_device, "input")
        sample_rate = int(device_info["default_samplerate"])
        
        self.model = Model(lang="en-us")
        self._input_stream = sd.RawInputStream(samplerate=sample_rate, blocksize=8000, dtype="int16", channels=1, callback=self._callback)
        self.rec = KaldiRecognizer(self.model, sample_rate)
        self._capturing = False
        self._loop = None

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
        b = bytes(indata)
        loop = self._loop
        asyncio.run_coroutine_threadsafe(self.q.put(b), loop)

    def kill(self):
        print("Killing speech recognizer...")
        self._capturing = False
        self._input_stream.abort()

    def is_capturing(self):
        return self._capturing

    async def start(self, subscribers: List[Callable]):
        print("ASR starting...")
        try:
            self._loop = asyncio.get_event_loop()
            self._input_stream.start()
            while True:
                data = await self.q.get()
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
                        await sub(text)
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
        except asyncio.CancelledError as e:
            print(f"asr asyncio.CancelledError: {e}")
            self.kill()
            return
        except Exception as e:
            print(f"asr Exception: {e}")
            self.kill()
            return

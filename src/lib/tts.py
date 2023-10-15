import time
import asyncio
import torch
from queue import Queue
from TTS.api import TTS as TextToSpeech

from lib.sound import Audio
# from lib.spacy import chunk_words

device = "cuda" if torch.cuda.is_available() else "cpu"

class TTS:
    def __init__(self, model_name="tts_models/en/ljspeech/vits", vocoder_name="vocoder_models/en/ljspeech/hifigan_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TextToSpeech(model_name=model_name, progress_bar=False).to(self.device)
        self.queue = Queue()
        self._playing = False
        asyncio.create_task(self.play_task())

    async def play_task(self):
        while True:
            wav = self.queue.get()
            if not wav: continue
            self._playing = True
            Audio.play_audio(wav, sample_rate=22050, volume=0.6)
            self._playing = False

    def is_playing(self):
        return self._playing

    def text_to_speech(self, text):
        wav = self.tts.tts(text)
        self.queue.put(wav)

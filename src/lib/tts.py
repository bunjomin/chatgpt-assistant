import os
import asyncio
import requests
import zipfile
import torch
from TTS.api import TTS as TextToSpeech

from lib.sound import Audio

device = "cuda" if torch.cuda.is_available() else "cpu"

_DIR = os.path.dirname(os.path.abspath(__file__))

class TTS:
    class _URLS:
        VITS = "https://github.com/coqui-ai/TTS/releases/download/v0.6.1_models/tts_models--en--ljspeech--vits.zip"

    class _PATHS:
        MODELS = os.path.normpath(os.path.join(_DIR, "../data/models"))
        VITS = os.path.normpath(os.path.join(_DIR, "../data/models/vits"))

    def _ensure_model(self):
        if not os.path.exists(self._PATHS.VITS):
            self.download_and_unzip()

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._ensure_model()

        model_path = os.path.normpath(os.path.join(self._PATHS.VITS, "model_file.pth"))
        model_config = os.path.normpath(os.path.join(self._PATHS.VITS, "config.json"))

        self.tts = TextToSpeech(model_path=model_path, config_path=model_config, progress_bar=False).to(self.device)
    
    def download_and_unzip(self):
        """
        Download and unzip a zip file.
        
        :param url: The URL of the zip file.
        :param dest_folder: The destination folder where the content should be unzipped.
        """
        response = requests.get(self._URLS.VITS, stream=True)
        zip_path = os.path.join(self._PATHS.MODELS, "temp.zip")
        with open(zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self._PATHS.MODELS)
        os.remove(zip_path)
        os.rename(os.path.join(self._PATHS.MODELS, "tts_models--en--ljspeech--vits"), self._PATHS.VITS)

    async def text_to_speech(self, text):
        wav = self.tts.tts(text)
        await asyncio.to_thread(Audio.play_audio, wav, 22050, 0.6)
        # q = asyncio.Queue()
        # for chunk in chunk_words(text, 12, 6):
        #     wav = self.tts.tts(chunk)
        #     q.put_nowait(wav)
        # while not q.empty():
        #     wav = await q.get()
        #     if not wav: continue
        #     await asyncio.to_thread(Audio.play_audio, wav, 22050, 0.6)

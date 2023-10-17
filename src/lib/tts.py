import os
import requests
import sounddevice as sd

from piper import PiperVoice

_DIR = os.path.dirname(os.path.abspath(__file__))


class TTS:
    class _PATHS:
        MODELS = os.path.normpath(os.path.join(_DIR, "../data/models"))
        AMY = os.path.normpath(os.path.join(_DIR, "../data/models/amy"))

    class _URLS:
        AMY_MODEL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx"
        AMY_CONFIG = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json"

    def _ensure_files(self):
        if not os.path.exists(self._PATHS.AMY):
            os.makedirs(self._PATHS.AMY)
        if not os.path.exists(
            os.path.normpath(os.path.join(self._PATHS.AMY, "en_US-amy-medium.onnx"))
        ) or not os.path.exists(
            os.path.normpath(os.path.join(self._PATHS.AMY, "config.json"))
        ):
            self.download_and_unzip()

    def __init__(self):
        self._ensure_files()
        model_path = os.path.normpath(
            os.path.join(self._PATHS.AMY, "en_US-amy-medium.onnx")
        )
        config_path = os.path.normpath(os.path.join(self._PATHS.AMY, "config.json"))

        self.tts = PiperVoice.load(model_path, config_path, use_cuda=False)

    def download_and_unzip(self):
        response = requests.get(self._URLS.AMY_MODEL, stream=True)
        with open(
            os.path.normpath(os.path.join(self._PATHS.AMY, "en_US-amy-medium.onnx")),
            "wb",
        ) as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        response = requests.get(self._URLS.AMY_CONFIG, stream=True)
        with open(
            os.path.normpath(os.path.join(self._PATHS.AMY, "config.json")), "wb"
        ) as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)

    def text_to_speech(self, text):
        stream = sd.RawOutputStream(samplerate=22050, dtype="int16", channels=1)
        stream.start()
        for bytes in self.tts.synthesize_stream_raw(text):
            stream.write(bytes)
        stream.close()
        return

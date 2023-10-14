import pyaudio

class PyAudioSingleton:
    _instance = None
    paInt16 = pyaudio.paInt16
    paFloat32 = pyaudio.paFloat32
    paContinue = pyaudio.paContinue

    @staticmethod
    def get_instance():
        if PyAudioSingleton._instance is None:
            PyAudioSingleton._instance = pyaudio.PyAudio()
        return PyAudioSingleton._instance

    @staticmethod
    def terminate():
        if PyAudioSingleton._instance is not None:
            PyAudioSingleton._instance.terminate()
            PyAudioSingleton._instance = None

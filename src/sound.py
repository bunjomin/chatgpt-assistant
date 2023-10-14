import os
import wave
import numpy as np

from audio import PyAudioSingleton

class Sound:
    @staticmethod
    def play_audio(audio, sample_rate=22050):
        p = PyAudioSingleton.get_instance()
        stream = p.open(format=PyAudioSingleton.paFloat32,
                        channels=1,
                        rate=sample_rate,
                        output=True)
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()

    @staticmethod
    def play_sound_file(filename, extension="wav"):
        dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.normpath(os.path.join(dir, "./assets/", f"{filename}.{extension}"))
        wf = wave.open(file_path, 'rb')

        # Read the entire audio data
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0  # Convert to float32

        # Get the sample rate
        sample_rate = wf.getframerate()

        Sound.play_audio(audio_array, sample_rate)

import sounddevice as sd
import numpy as np
import os
import wave

class Audio:

    @staticmethod
    def play_audio(audio, sample_rate=22050, volume=1.0):
        # Ensure audio is in the correct float32 format with values between -1 and 1
        if isinstance(audio, (list, np.ndarray)):
            audio_data = []
            for n in audio:
                if isinstance(n, np.ndarray):
                    adjusted_audio = np.float32(n.astype(np.float32) * volume)
                    audio_data.append(np.clip(adjusted_audio, -1.0, 1.0))
                elif isinstance(n, (np.float32, float)):
                    adjusted_audio = np.float32(n * volume)
                    audio_data.append(np.clip(np.array([adjusted_audio]), -1.0, 1.0))
                elif isinstance(n, int):
                    float_val = np.float32(n / float(2**15))  # Convert 16-bit int to float between -1 and 1
                    adjusted_audio = np.float32(float_val * volume)
                    audio_data.append(np.clip(np.array([adjusted_audio]), -1.0, 1.0))
            audio_data = np.concatenate(audio_data)
        elif isinstance(audio, np.ndarray):
            adjusted_audio = np.float32(audio.astype(np.float32) * volume)
            audio_data = np.clip(adjusted_audio, -1.0, 1.0)

        # Play the audio
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()

    @staticmethod
    def play_sound_file(filename, extension="wav"):
        dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.normpath(os.path.join(dir, "../assets/", f"{filename}.{extension}"))
        wf = wave.open(file_path, 'rb')

        # Read the entire audio data
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0  # Convert to float32

        # Get the sample rate
        sample_rate = wf.getframerate()

        Audio.play_audio(audio_array, sample_rate, 0.6)

import os
import numpy as np
import time
import re

from abc import ABC, abstractmethod

# Local libs
from chatgpt import ChatGPT
from tts import tts
from model_asr import asr
from sound import Sound
from audio import PyAudioSingleton
from lib.spellchecker import SpellChecker

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class BaseListener(ABC):
    def __init__(self, listener_context, timeout=1.0) -> None:
        self.chunks = []
        self.last_spoken = None
        self.empty_counter = 0
        self.context = listener_context
        self.timeout = timeout

    # Super simple way to calculate a numeric similarity
    # between two strings. It's not _great_, but it seems
    # to do okay and couldn't be more lightweight.
    # May want to replace with a lib if this doesn't end
    # up being good enough.
    # https://en.wikipedia.org/wiki/Jaccard_index
    @staticmethod
    def jaccard_similarity(str1, str2):
        set1 = set(str1)
        set2 = set(str2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    @abstractmethod
    def empty(self): pass

    @abstractmethod
    def appended(self): pass

    @abstractmethod
    def done(self): pass

    def evaluate(self, text):
        if len(text):
            self.chunks.append(text)
            self.empty_counter = asr.offset
            self.last_spoken = time.time()
            return self.appended()

        if self.empty_counter <= 0: return

        self.empty_counter -= 1
        if self.empty_counter > 0: return

        if self.last_spoken == None: return self.empty()
        if time.time() - self.last_spoken > self.timeout:
            return self.done()
        if not self.chunks or not self.chunks[-1] or type(self.chunks[-1]) is not str: return self.empty()
        self.chunks.append(" ")
        self.empty()

class CommandListener(BaseListener):
    _SLEEP_PHRASES = ["stop", "quit", "exit", "shut down"]

    def __init__(self, listener_context):
        super().__init__(listener_context)
        self.chat_gpt = ChatGPT({ "api_key": f"{OPENAI_API_KEY}" })

    def submit(self):
        joined = ' '.join(self.chunks)
        print (f"\nSUBMITTING: {joined}")
        # TODO: figure out how to make a loading sound work async
        Sound.play_sound_file("sleep")
        response = self.chat_gpt.chat(joined)
        print(f"\nRESPONSE: {response}")
        tts(response)
        Sound.play_sound_file("awake")
        self.chunks.clear()

    def empty(self): pass

    def appended(self): pass
    
    def done(self):
        for sleep_phrase in self._SLEEP_PHRASES:
            if BaseListener.jaccard_similarity(' '.join(self.chunks).strip(), sleep_phrase) > 0.7:
                Sound.play_sound_file("sleep")
                self.context.sleep()
                return
        self.submit()
        self.chunks.clear()

    def evaluate(self, text):
        if not self.context.awake: return
        super().evaluate(text)

class WakeListener(BaseListener):
    _WAKE_PHRASES = ["hey gpt", "ok gpt", "chatgpt"]
    _QUIT_PHRASES = ["stop", "quit", "exit", "shut down"]
    _SIMILARITY_THRESHOLD = 0.7

    def __init__(self, listener_context, timeout=1.0):
        super().__init__(listener_context, timeout)
        print('Listening...')

    def empty(self): pass

    def appended(self):
        joined = " ".join(self.chunks)
        print(f"joined: {joined}")
        for wake_phrase in self._WAKE_PHRASES:
            similarity = self.jaccard_similarity(wake_phrase, joined)
            if similarity < WakeListener._SIMILARITY_THRESHOLD: continue
            Sound.play_sound_file("awake")
            print("awake")
            self.context.awaken()
            break
        for quit_phrase in self._QUIT_PHRASES:
            similarity = self.jaccard_similarity(quit_phrase, joined)
            if similarity < WakeListener._SIMILARITY_THRESHOLD: continue
            self.context.quit()
            break

    def done(self):
        self.chunks = []

    def evaluate(self, text):
        if self.context.awake: return None
        super().evaluate(text)

class ListenerContext:
    def __init__(self):
        self.awake = False
        self.active_listener = None
        self.set_awake(False)

    def set_awake(self, bool):
        self.awake = bool
        if bool: self.active_listener = CommandListener(self)
        else: self.active_listener = WakeListener(self)

    def awaken(self):
        self.set_awake(True)

    def sleep(self):
        self.set_awake(False)
    
    def set_recognition(self, recognition):
        self.recognition = recognition

    def quit(self):
        Sound.play_sound_file("quit")
        self.recognition.quit()

class Recognition:
    def __init__(self, listener_context=ListenerContext()):
        self.p = PyAudioSingleton.get_instance()
        Sound.play_sound_file("startup")
        input_device = self.p.get_default_input_device_info()
        device_idx = input_device["index"]
        self.listener_context = listener_context
        self.listener_context.set_recognition(self)
        self.stream = self.p.open(format=PyAudioSingleton.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        input_device_index=device_idx,
                        stream_callback=self.callback,
                        frames_per_buffer=16000)
        self.should_quit = False
    
    def _clean_text(self, text):
        return SpellChecker.correct_consonants(re.sub(r'\s{2,}', ' ', text))

    def quit(self):
        self.should_quit = True

    def callback(self, in_data, frame_count, time_info, status):
        if not self.stream.is_active() or self.stream.is_stopped(): return
        signal = np.frombuffer(in_data, dtype=np.int16)
        text = asr.transcribe(signal)
        if isinstance(self.listener_context.active_listener, CommandListener):
            text = self._clean_text(text)
        self.listener_context.active_listener.evaluate(text)
        return (in_data, PyAudioSingleton.paContinue)
    
    def listen(self):
        self.stream.start_stream()

        try:
            while not self.should_quit and self.stream.is_active():
                time.sleep(0.1)
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            exit()

Recognition().listen()

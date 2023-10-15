import os
import time
import asyncio

from chatgpt import ChatGPT
from lib.sound import Audio
from lib.tts import TTS
from lib.speech_recognition import SpeechRecognizer

class Assistant:
    _QUIT_PHRASES = ["stop", "quit", "exit", "shut down"]
    _WAKE_PHRASES = ["hey gpt", "ok gpt", "chatgpt", "okay gpt"]
    _SIMILARITY_THRESHOLD = 0.7

    @staticmethod
    def jaccard_similarity(str1, str2):
        set1 = set(str1)
        set2 = set(str2)
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def __init__(self):
        self._awake = False
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key or len(api_key) == 0:
            raise ValueError("You must provide an OpenAI API key with the environment variable OPENAI_API_KEY.")
        self._api_key = api_key
        self.tts = TTS()
        self.chat_gpt = ChatGPT({ "api_key": f"{self._api_key}" })
        self.speech_recognizer = SpeechRecognizer()
        self._last_speech_timestamp = None

    def chat(self, text):
        self._last_speech_timestamp = None
        response = self.chat_gpt.chat(text)
        print(f"\nRESPONSE: {response}")
        self.tts.text_to_speech(response)
        while self.tts.is_playing():
            time.sleep(0.1)
        Audio.play_sound_file("awake")
        self._last_speech_timestamp = round(time.time(), 2)

    def sleep(self):
        Audio.play_sound_file("sleep")
        self._awake = False
        self._last_speech_timestamp = None

    def awaken(self):
        Audio.play_sound_file("awake")
        self._awake = True
        self._last_speech_timestamp = round(time.time(), 2)

    def handle_awake(self, text):
        print(f"awake text: {text}")
        for quit_phrase in self._QUIT_PHRASES:
            if Assistant.jaccard_similarity(quit_phrase, text) > self._SIMILARITY_THRESHOLD:
                self.sleep()
                self.chat_gpt.reset()
                return
        self.chat(text)

    def handle_asleep(self, text):
        print(f"sleep text: {text}")
        for quit_phrase in self._QUIT_PHRASES:
            if Assistant.jaccard_similarity(quit_phrase, text) > self._SIMILARITY_THRESHOLD:
                self.quit()
                return
        for wake_phrase in self._WAKE_PHRASES:
            if Assistant.jaccard_similarity(wake_phrase, text) > self._SIMILARITY_THRESHOLD:
                self.awaken()
                return

    def handle_speech(self, text):
        # Weird bug where the asr returns "huh" for various air sounds
        if text == "huh": return
        self._last_speech_timestamp = round(time.time(), 2)
        if self._awake:
            self.handle_awake(text)
        else:
            self.handle_asleep(text)

    async def check_awake_status(self):
        while True:
            await asyncio.sleep(0.5)
            if self._last_speech_timestamp is None:
                print("No speech yet")
                continue
            if self.speech_recognizer.is_capturing():
                print("Currently capturing...")
                self._last_speech_timestamp = round(time.time(), 2)
                continue
            time_since_last_speech = round(round(time.time(), 2) - self._last_speech_timestamp, 2)
            if time_since_last_speech > 3.0 and self._awake:
                print(f"Sleeping due to inactivity ({time_since_last_speech} seconds)")
                self.sleep()
            print(f"Time since last speech: {time_since_last_speech} seconds")

    async def start(self):
        Audio.play_sound_file("startup")
        print("Listening...\n")
        asyncio.create_task(self.check_awake_status())  # Start the asynchronous method
        await asyncio.to_thread(self.speech_recognizer.start, [self.handle_speech])

    def quit(self):
        Audio.play_sound_file("quit")
        self.speech_recognizer.kill()
        for task in asyncio.all_tasks():
            task.cancel()
        exit()


async def main():
    try:
        assistant = Assistant()
        await assistant.start()
    except KeyboardInterrupt:
        assistant.quit()
    finally:
        assistant.quit()

if __name__ == "__main__":
    asyncio.run(main())

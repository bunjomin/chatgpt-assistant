import os
import time
import sys
import asyncio
import numpy as np
import re
from dotenv import load_dotenv

from src.lib.sound import Audio
from src.lib.speech_recognition import SpeechRecognizer
from src.lib.chatgpt import ChatGPT
from src.lib.tts import TTS

load_dotenv()
os.environ["PA_ALSA_PLUGHW"] = "1"


async def shutdown(assistant):
    await asyncio.to_thread(Audio.play_sound_file, "quit")
    if assistant:
        await assistant.quit()


def convert_24bit_wav_to_float32(audio_data):
    audio_bytes = np.frombuffer(audio_data, dtype=np.uint8)
    audio_int32 = np.zeros(len(audio_bytes) // 3, dtype=np.int32)
    audio_int32 += (
        audio_bytes[::3].astype(np.int32) << 8
    )  # Shift left by 8 bits for the least significant byte
    audio_int32 += (
        audio_bytes[1::3].astype(np.int32) << 16
    )  # Shift left by 16 bits for the middle byte
    audio_int32 += (
        audio_bytes[2::3].astype(np.int32) - 128
    ) << 24  # Subtract 128 (to make it signed) and shift left by 24 bits for the most significant byte
    audio_float32 = audio_int32 / (2**23 - 1)

    return audio_float32


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
        api_key = os.environ.get("OPENAI_API_KEY")
        self._api_key = api_key
        self.chat_gpt = ChatGPT({"api_key": self._api_key})
        self.tts = TTS()
        self.speech_recognizer = SpeechRecognizer()
        self._last_speech_timestamp = None
        self.current_conversation = []

    async def resume(self):
        self.speech_recognizer.resume()
        self._last_speech_timestamp = round(time.time(), 2)
        await asyncio.to_thread(Audio.play_sound_file, "awake")

    async def chat(self, text):
        try:
            self._last_speech_timestamp = None
            self.speech_recognizer.pause()
            full_response = ""
            chunks = []
            async for chunk in self.chat_gpt.chat(text):
                chunks.append(chunk)
                if (
                    len(chunks) > 10
                    and not re.match(r"\d+[.,]$", chunk.strip())
                    and not re.match(r"\d+[.,]$", chunks[-1].strip())
                ):
                    for terminator in [",", ".", ";", "and", "or"]:
                        if chunk.strip().endswith(terminator):
                            joined = "".join(chunks)
                            print(f"playing segment: {joined}")
                            await asyncio.to_thread(self.tts.text_to_speech, joined)
                            full_response += joined
                            chunks = []

            if len(chunks) > 0:
                joined = "".join(chunks)
                print(f"playing final segment: {joined}")
                await asyncio.to_thread(self.tts.text_to_speech, joined)
                full_response += joined
                chunks = []

            self.current_conversation.append(
                {
                    "role": "user",
                    "content": text,
                }
            )

            self.current_conversation.append(
                {
                    "role": "assistant",
                    "content": full_response,
                }
            )

            await self.resume()

        except Exception as e:
            print(f"chat: Exception: {e}")
            await self.resume()
            return

    async def sleep(self):
        await asyncio.to_thread(Audio.play_sound_file, "sleep")
        self._awake = False
        self._last_speech_timestamp = None

    async def awaken(self):
        await asyncio.to_thread(Audio.play_sound_file, "awake")
        self._awake = True
        self._last_speech_timestamp = round(time.time(), 2)

    async def handle_awake(self, text):
        print(f"awake text: {text}")
        for quit_phrase in self._QUIT_PHRASES:
            if (
                Assistant.jaccard_similarity(quit_phrase, text)
                > self._SIMILARITY_THRESHOLD
            ):
                await self.sleep()
                self.current_conversation = []
                self.chat_gpt.reset()
                return
        await self.chat(text)

    async def handle_asleep(self, text):
        for quit_phrase in self._QUIT_PHRASES:
            if (
                Assistant.jaccard_similarity(quit_phrase, text)
                > self._SIMILARITY_THRESHOLD
            ):
                await self.quit()
                return
        for wake_phrase in self._WAKE_PHRASES:
            if (
                Assistant.jaccard_similarity(wake_phrase, text)
                > self._SIMILARITY_THRESHOLD
            ):
                await self.awaken()
                return

    async def handle_speech(self, text):
        # Weird bug where the asr returns "huh" for various air sounds
        if text == "huh":
            return
        self._last_speech_timestamp = round(time.time(), 2)
        if self._awake:
            await self.handle_awake(text)
        else:
            await self.handle_asleep(text)

    async def check_awake_status(self):
        try:
            while True:
                await asyncio.sleep(0.5)
                if self._last_speech_timestamp is None:
                    continue
                if self.speech_recognizer.is_capturing():
                    self._last_speech_timestamp = round(time.time(), 2)
                    continue
                time_since_last_speech = round(
                    round(time.time(), 2) - self._last_speech_timestamp, 2
                )
                if time_since_last_speech > 4.0 and self._awake:
                    print(
                        f"Sleeping due to inactivity ({time_since_last_speech} seconds)"
                    )
                    await self.sleep()
        except asyncio.CancelledError:
            print("check_awake_status: asyncio.CancelledError")
            return
        except Exception as e:
            print(f"check_awake_status: Exception: {e}")
            return

    async def quit(self):
        for task in asyncio.all_tasks():
            print(f"task: {task}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.speech_recognizer.kill()
        sys.exit()

    async def start(self):
        print("Starting assistant...\n")
        try:
            await asyncio.to_thread(Audio.play_sound_file, "startup")
            await asyncio.gather(
                self.check_awake_status(),
                self.speech_recognizer.start([self.handle_speech]),
            )
        except asyncio.CancelledError:
            print("start: asyncio.CancelledError")
            await self.quit()
        except Exception as e:
            print(f"start: Exception: {e}")
            await self.quit()
            return


if __name__ == "__main__":
    assistant = None
    try:
        assistant = Assistant()
        asyncio.run(assistant.start())
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Shutting down.")
        if assistant:
            asyncio.run(shutdown(assistant))
    except Exception as e:
        print(f"main Exception: {e}")
        if assistant:
            asyncio.run(shutdown(assistant))

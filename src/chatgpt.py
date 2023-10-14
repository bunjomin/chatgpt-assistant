import requests
import copy

# TODO:
# Handle streamed responses so we can start
# doing TTS more quickly.

class ChatGPT:
    __context_messages = [
        {
            "role": "user",
            # Eventually we'll have a screen
            "content": "I am chatting via voice and TTS. Only respond to me with plain text.",
        },
    ]

    __API_BASE = "https://api.openai.com/v1"

    def __init__(self, config, messages=[]):
        api_key = config.get("api_key")
        if not api_key: raise ValueError("You must provide an OpenAI API key.")

        top_p = config.get("top_p") or 0.1
        self.top_p = top_p

        max_tokens = config.get("max_tokens") or 100
        self.max_tokens = max_tokens

        model = config.get("model") or "gpt-3.5-turbo"
        self.model = model

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        if len(messages):
            self.messages = messages
        else:
            self.messages = copy.copy(self.__context_messages)

    def chat(self, message):
        self.messages.append({
            "role": "user",
            "content": message,
        })
        body = {
            "model": self.model,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
        }
        try:
            r = requests.post(f"{ChatGPT.__API_BASE}/chat/completions", headers=self.headers, json=body)
        except requests.ConnectionError:
            raise KeyboardInterrupt
        except requests.Timeout:
            raise KeyboardInterrupt
        if r.status_code == 200:
            response = r.json()
            message_response = response["choices"][0]["message"]
            if not message_response or not message_response["content"]: return None
            self.messages.append({
                "role": "assistant",
                "content": message_response["content"],
            })
            return message_response["content"]
        elif r.status_code == 400:
            raise EOFError
        elif r.status_code == 401:
            print(r.json())
            raise EOFError
        elif r.status_code == 429:
            raise KeyboardInterrupt
        elif r.status_code == 502 or r.status_code == 503:
            raise KeyboardInterrupt
        else:
            raise EOFError


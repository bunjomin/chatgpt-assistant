import aiohttp
import copy
import json

class ChatGeneratorEnd(Exception):
    def __init__(self, final_value):
        self.final_value = final_value

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

        if messages and len(messages):
            self.messages = messages
        else:
            self.messages = copy.copy(self.__context_messages)

    async def chat(self, message, raiseFullResult=False):
        self.messages.append({
            "role": "user",
            "content": message,
        })
        body = {
            'messages': self.messages,
            'model': self.model,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'stream': True,
        }
        
        headers = self.headers

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{ChatGPT.__API_BASE}/chat/completions", headers=headers, json=body) as res:
                try:
                    if not res.content:
                        raise ValueError("No response body")
                    
                    chunks = []
                    async for chunk in res.content.iter_any():
                        stringed = chunk.decode()
                        split_string = stringed.split("data: ")
                        split = split_string[1:] if len(split_string) > 1 else None
                        
                        if not split:
                            continue
                        if split == "[DONE]":
                            break
                        
                        for line in split:
                            try:
                                parsed = json.loads(line)
                            except json.JSONDecodeError:
                                parsed = None
                            
                            if not parsed:
                                continue
                            if parsed.get("choices", [{}])[0].get("finish_reason", None) is not None:
                                break
                            delta = parsed.get("choices", [{}])[0].get("delta", {}).get("content", None)
                            if not delta:
                                continue
                            chunks.append(delta)
                            yield delta
                except Exception as e:
                    print(f"chat: Exception: {e}")

        if raiseFullResult:
            raise ChatGeneratorEnd(chunks)

    def reset(self):
        self.messages = copy.copy(self.__context_messages)

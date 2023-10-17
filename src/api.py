import os
import wave
import io
import asyncio
import base64
import numpy as np

from flask import Flask, jsonify, request, Response
from flask_httpauth import HTTPTokenAuth
from lib.tts import TTS
from lib.chatgpt import ChatGPT

tts = TTS()
app = Flask(__name__)
api_key = os.environ.get('OPENAI_API_KEY')
auth = HTTPTokenAuth(scheme='Bearer')

TOKENS = {
    "super-secret-test-token": "me"
}

ROLES = {
    "user": "user",
    "assistant": "assistant",
}

def normalize_audio(wav):
    if isinstance(wav, (list, np.ndarray)):
        audio_data = []
        for n in wav:
            if isinstance(n, np.ndarray):
                adjusted_audio = np.float32(n.astype(np.float32))
                audio_data.append(np.clip(adjusted_audio, -1.0, 1.0))
            elif isinstance(n, (np.float32, float)):
                adjusted_audio = np.float32(n)
                audio_data.append(np.clip(np.array([adjusted_audio]), -1.0, 1.0))
            elif isinstance(n, int):
                float_val = np.float32(n / float(2**15))  # Convert 16-bit int to float between -1 and 1
                adjusted_audio = np.float32(float_val)
                audio_data.append(np.clip(np.array([adjusted_audio]), -1.0, 1.0))
        audio_data = np.concatenate(audio_data)
        return audio_data
    elif isinstance(wav, np.ndarray):
        adjusted_audio = np.float32(wav.astype(np.float32))
        audio_data = np.clip(adjusted_audio, -1.0, 1.0)
        return audio_data

@auth.verify_token
def verify_token(token):
    if token in TOKENS:
        return TOKENS[token]
    return None

@app.route('/tts', methods=['POST'])
@auth.login_required
async def ttsify():
    body = request.json
    if not body.get("messages") or not len(body["messages"]): return jsonify({"message": "No conversation provided."}), 400
    raw_messages = body["messages"]
    messages = []
    for msg in raw_messages:
        if not msg.get("role"): return jsonify({"message": "Each message must have a role."}), 400
        if not msg.get("content"): return jsonify({"message": "Each message must have content."}), 400
        content = msg["content"]
        role = msg["role"]
        if role not in ROLES: return jsonify({"message": "Invalid role."}), 400
        if not isinstance(content, str): return jsonify({"message": "Content must be a string."}), 400
        if len(content) > 1000: return jsonify({"message": "Content must be less than 1000 characters."}), 400
        if content.strip() == "": return jsonify({"message": "Content must not be empty."}), 400
        messages.append({ "content": content, "role": role })

    message = messages.pop().get("content")
    if not message or not len(message) or not isinstance(message, str): return jsonify({"message": "Invalid message."}), 400
    chat_gpt = ChatGPT({ "api_key": f"{api_key}" }, messages=messages)
    text_response = await asyncio.to_thread(chat_gpt.chat, message)
    encoded = base64.b64encode(text_response.encode("utf-8")).decode("utf-8")
    wav = await tts.text_to_speech(text_response, play=False)
    print("got wav")
    wav = normalize_audio(wav)
    # Convert float32 data to int16
    wav_int16 = np.int16(wav * 32767)
    print("int16")

    # Write to an in-memory WAV file using the wave module
    output = io.BytesIO()
    print("created output")
    with wave.open(output, 'wb') as wf:
        wf.setnchannels(1)  # Assuming mono audio
        print("set channels")
        wf.setsampwidth(2)  # 16-bit PCM
        print("set sampwidth")
        wf.setframerate(22050)  # Your sample rate
        print("set framerate")
        wf.writeframes(wav_int16.tobytes())
        print("wrote frames")

    # Prepare and send the response
    output.seek(0)
    print("seeked")
    response = Response(output.read(), mimetype="audio/wav")
    print("created response")
    response.headers["x-response"] = encoded
    print("set headers")
    print(f"encoded: {encoded}")
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

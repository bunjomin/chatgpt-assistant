import torch

from math import ceil
from nemo.collections.common.parts.preprocessing import parsers
from models import models
from sound import Sound
from lib.spacy import chunk_words

# TODO: Figure out how to get the TTS to sound less robotic

asr_model = models["asr"]
tts_model_spectrogram = models["tts_spectrogram"]
tts_model_vocoder = models["tts_vocoder"]
models = [asr_model, tts_model_spectrogram, tts_model_vocoder]

if torch.cuda.is_available():
    for i, m in enumerate(models):
        models[i] = m.cuda()
for m in models:
    m.eval()

asr_model, tts_model_spectrogram, tts_model_vocoder = models

parser = parsers.make_parser(
    labels=asr_model.decoder.vocabulary, name="en", unk_id=-1, blank_id=-1, do_normalize=True,
)

def tts(text):
    labels_map = dict([(i, asr_model.decoder.vocabulary[i]) for i in range(len(asr_model.decoder.vocabulary))])

    segments = chunk_words(text)
    
    for segment in segments:
        tts_input = []
        asr_references = []
        tts_parsed_input = tts_model_spectrogram.parse(segment)
        tts_input.append(tts_parsed_input.squeeze())

        asr_parsed = parser(segment)
        asr_parsed = ''.join([labels_map[c] for c in asr_parsed])
        asr_references.append(asr_parsed)

        tts_input = torch.stack(tts_input)
        if torch.cuda.is_available():
            tts_input = tts_input.cuda()
        specs = tts_model_spectrogram.generate_spectrogram(tokens=tts_input)
        audio = []
        step = ceil(len(specs) / 4)
        num_chunks = len(specs) // step

        for i in range(num_chunks):
            current_spec = specs[i * step : (i + 1) * step]
            if current_spec.shape[0] == 0:
                continue
            audio.append(tts_model_vocoder.convert_spectrogram_to_audio(spec=current_spec))

        audio = [item for sublist in audio for item in sublist]

        for aud in audio:
            aud = aud.cpu().numpy()
        Sound.play_audio(aud)

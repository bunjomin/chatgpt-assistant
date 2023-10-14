import os
import nemo
import copy

import nemo.collections.asr as nemo_asr
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

from omegaconf import OmegaConf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(CURRENT_DIR, './data/models'))

class MODEL_NAME_MAP:
     ASR = "QuartzNet15x5Base-En"
     SPECTROGRAM = "tts_en_tacotron2"
     VOCODER = "tts_en_waveglow_88m"

class ModelLoader:
     def __init__(self, name, base):
          self.base = base
          self.path = os.path.normpath(os.path.join(MODEL_DIR, f'./{name}'))
          if os.path.exists(self.path) and not os.path.isdir(self.path):
               self.model = self.base.restore_from(self.path)
          else:
               self.model = self.base.from_pretrained(name)
               self.model.save_to(self.path)

class Models:
     asr = ModelLoader(MODEL_NAME_MAP.ASR, nemo_asr.models.EncDecCTCModel)
     spectrogram = ModelLoader(MODEL_NAME_MAP.SPECTROGRAM, SpectrogramGenerator)
     vocoder = ModelLoader(MODEL_NAME_MAP.VOCODER, Vocoder)

asr_model = Models.asr.model
tts_model_spectrogram = Models.spectrogram.model
tts_model_vocoder = Models.vocoder.model

cfg = copy.deepcopy(asr_model._cfg)

# Make config overwrite-able
OmegaConf.set_struct(cfg.preprocessor, False)

# some changes for streaming scenario
cfg.preprocessor.dither = 0.0
cfg.preprocessor.pad_to = 0

# spectrogram normalization constants
normalization = {}
normalization['fixed_mean'] = [
     -14.95827016, -12.71798736, -11.76067913, -10.83311182,
     -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
     -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
     -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
     -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
     -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
     -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
     -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
     -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
     -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
     -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
     -10.10687659, -10.14536695, -10.30828702, -10.23542833,
     -10.88546868, -11.31723646, -11.46087382, -11.54877829,
     -11.62400934, -11.92190509, -12.14063815, -11.65130117,
     -11.58308531, -12.22214663, -12.42927197, -12.58039805,
     -13.10098969, -13.14345864, -13.31835645, -14.47345634]
normalization['fixed_std'] = [
     3.81402054, 4.12647781, 4.05007065, 3.87790987,
     3.74721178, 3.68377423, 3.69344,    3.54001005,
     3.59530412, 3.63752368, 3.62826417, 3.56488469,
     3.53740577, 3.68313898, 3.67138151, 3.55707266,
     3.54919572, 3.55721289, 3.56723346, 3.46029304,
     3.44119672, 3.49030548, 3.39328435, 3.28244406,
     3.28001423, 3.26744937, 3.46692348, 3.35378948,
     2.96330901, 2.97663111, 3.04575148, 2.89717604,
     2.95659301, 2.90181116, 2.7111687,  2.93041291,
     2.86647897, 2.73473181, 2.71495654, 2.75543763,
     2.79174615, 2.96076456, 2.57376336, 2.68789782,
     2.90930817, 2.90412004, 2.76187531, 2.89905006,
     2.65896173, 2.81032176, 2.87769857, 2.84665271,
     2.80863137, 2.80707634, 2.83752184, 3.01914511,
     2.92046439, 2.78461139, 2.90034605, 2.94599508,
     2.99099718, 3.0167554,  3.04649716, 2.94116777]

cfg.preprocessor.normalize = normalization

# Disable config overwriting
OmegaConf.set_struct(cfg.preprocessor, True)

asr_model.preprocessor = asr_model.from_config_dict(cfg.preprocessor)

# Set model to inference mode
asr_model.eval()

models = { "asr": asr_model, "tts_spectrogram": tts_model_spectrogram, "tts_vocoder": tts_model_vocoder }

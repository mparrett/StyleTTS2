import os

os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'
#os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import nltk
nltk.download('punkt')

import torch
torch.manual_seed(0)

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
import numpy as np
import torch
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

from models import *
from utils import *

from contextlib import contextmanager

from test_utils import Timed

from torch.profiler import profile, record_function, ProfilerActivity

USE_MPS = int(os.getenv("USE_MPS", 0))
GPU_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if USE_MPS != 0:
  GPU_DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'  # TODO
#device_cpu = 'cpu'


from text_utils import TextCleaner
textclenaer = TextCleaner()

# %matplotlib inline

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(GPU_DEVICE)
    #display(mel_tensor)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        #display(ref_s)
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
        #display(ref_p)

    return torch.cat([ref_s, ref_p], dim=1)


# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])

FP16_ENABLE = False # False
USE_FP16 = USE_MPS and FP16_ENABLE
USE_DTYPE=torch.float16 if USE_FP16 else torch.float32

model = build_model(model_params, text_aligner, pitch_extractor, plbert, use_fp16=USE_FP16)

def update_model_params(model):
    for key in model:
        _ = model[key].eval()
        if key != 'text_encoder':
            _ = model[key].to(GPU_DEVICE)
        if key == 'decoder':
            _ = model[key]
            #import code; code.interact(local=locals())

    params_whole = torch.load("Models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')  # not sure about map location cpu here
    params = params_whole['net']
    def fix_state_dict(state_dict):
        new_state_dict = OrderedDict()
        for style_k, v in state_dict.items():
            name = style_k[7:] # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except Exception as e:
                print(e) # e.g. Error(s) in loading state_dict for StyleEncoder; Missing key(s) in state_dict: "shared.0.weight_orig", "shared.0.weight", "shared.0.weight_u"...
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = fix_state_dict(state_dict)
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    #             except:
    #                 _load(params[key], model[key])

    # set eval; half if using fp16 for decoder
    _ = [model[key].eval() for key in model]
    for key in model:
        if key == 'decoder':
            #import code; code.interact(local=locals())
            if USE_FP16:
                model[key].half()  # ?? 

update_model_params(model)

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

@Timed()
def tokenize(text, gpu_device):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    ps = ps.replace('``', '"')
    ps = ps.replace("''", '"')

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    #tokens = torch.LongTensor(tokens).to(gpu_device).unsqueeze(0)
    tokens = torch.LongTensor(tokens).unsqueeze(0)  #.to(gpu_device) # don't move to GPU because decoder is on CPU (mps-related)
    return tokens

@Timed()
def get_input_for_prosody(d_in, pred_aln_trg, dtype):
    """Get input for prosody. The method supports HiFi-GAN decoder adjustments."""
    en = (d_in @ pred_aln_trg.unsqueeze(0).to(device=GPU_DEVICE, dtype=dtype))
    if model_params.decoder.type == "hifigan":
        asr_new = torch.zeros_like(en)
        asr_new[:, :, 0] = en[:, :, 0]
        asr_new[:, :, 1:] = en[:, :, 0:-1]
        en = asr_new
    return en

@Timed()
def blend_predict_and_reference(s_pred, ref_s, alpha, beta):
    """Blend predicted and reference speech signals using alpha and beta weighting coefficients."""
    s, ref = s_pred[:, 128:], s_pred[:, :128]
    ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
    s = beta * s + (1 - beta)  * ref_s[:, 128:]
    return s, ref

@Timed()
def update_align_target(pred_aln_trg, pred_dur):
    """Update the alignment target tensor based on the predicted durations."""
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
        c_frame += int(pred_dur[i].data)
    return pred_aln_trg

@Timed()
def calc_predicted_durations(model, x):
    """Calculate predicted durations from the model's output."""
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1)
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    return pred_dur

@Timed()
def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    tokens = tokenize(text, GPU_DEVICE)
    
    with torch.no_grad():
        input_lengths = torch.Tensor([tokens.shape[-1]]).to(dtype=torch.int64) #.to(gpu_device)
        text_mask = length_to_mask(input_lengths)  # .to(gpu_device)
        with Timed("text encoder 1"):
            t_en = model.text_encoder(tokens, input_lengths, text_mask)

        # move all of these to the gpu device
        tokens = tokens.to(GPU_DEVICE)
        text_mask = text_mask.to(GPU_DEVICE)
        input_lengths = input_lengths.to(GPU_DEVICE)
        t_en = t_en.to(GPU_DEVICE)

        with Timed("bert duration"):
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        noise = torch.randn((1, 256)).unsqueeze(1).to(GPU_DEVICE)
        # ref_s: reference from the same speaker as the embedding
        with Timed("sampler"):
            s_pred = sampler(noise = noise, embedding=bert_dur, embedding_scale=embedding_scale, 
                             features=ref_s, num_steps=diffusion_steps).squeeze(1)

        s, ref = blend_predict_and_reference(s_pred, ref_s, alpha, beta)
        with Timed("text encoder 2"):
            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        with Timed("predictor.lstm"):
            x, _ = model.predictor.lstm(d)

        pred_dur = calc_predicted_durations(model, x)
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        pred_aln_trg = update_align_target(pred_aln_trg, pred_dur)

        # encode prosody
        with Timed("Prosody"):
            en = get_input_for_prosody(d.transpose(-1, -2), pred_aln_trg, torch.float32)
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            asr = get_input_for_prosody(t_en, pred_aln_trg, USE_DTYPE)

        with Timed("model.decoder"):
            out = model.decoder(asr, F0_pred.to(USE_DTYPE), N_pred.to(USE_DTYPE),
                                ref.squeeze().unsqueeze(0).to(USE_DTYPE))  ## this takes ~2000ms of 3500ms total

    return out.type(torch.float32).squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later


@Timed("LFInference")
def LFinference(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1):
  tokens = tokenize(text, GPU_DEVICE)

  with torch.no_grad():
      input_lengths = torch.LongTensor([tokens.shape[-1]]).to(GPU_DEVICE)
      text_mask = length_to_mask(input_lengths).to(GPU_DEVICE)

      t_en = model.text_encoder(tokens, input_lengths, text_mask)
      bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
      d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

      noise = torch.randn((1, 256)).unsqueeze(1).to(GPU_DEVICE)
      with Timed("Sampler"):
        s_pred = sampler(
            noise = noise,
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s, # reference from the same speaker as the embedding
            num_steps=diffusion_steps).squeeze(1)

      if s_prev is not None:
          # convex combination of previous and current style
          s_pred = t * s_prev + (1 - t) * s_pred

      s, ref = blend_predict_and_reference(s_pred, ref_s, alpha, beta)
      s_pred = torch.cat([ref, s], dim=-1)
      d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
      x, _ = model.predictor.lstm(d)
      duration = model.predictor.duration_proj(x)

      duration = torch.sigmoid(duration).sum(axis=-1)
      pred_dur = torch.round(duration.squeeze()).clamp(min=1)

      pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
      c_frame = 0
      for i in range(pred_aln_trg.size(0)):
          pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
          c_frame += int(pred_dur[i].data)

      # encode prosody
      en = get_input_for_prosody(d.transpose(-1, -2), pred_aln_trg, torch.float32)
      F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
      en = get_input_for_prosody(t_en, pred_aln_trg, USE_DTYPE)
      asr = (t_en @ pred_aln_trg.unsqueeze(0).to(GPU_DEVICE))
      with Timed("model.decoder"):
        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
  return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

@Timed()
def tokenize_st(text, gpu_device):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(gpu_device).unsqueeze(0)
    return tokens

@Timed("STinference")
def STinference(text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    tokens = tokenize_st(text, GPU_DEVICE)
    ref_tokens = tokenize_st(ref_text, GPU_DEVICE)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(GPU_DEVICE)
        text_mask = length_to_mask(input_lengths).to(GPU_DEVICE)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(GPU_DEVICE)
        #ref_text_mask = length_to_mask(ref_input_lengths).to(gpu_device)
        #ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
        noise = torch.randn((1, 256)).unsqueeze(1).to(GPU_DEVICE)
        with Timed("Sampler"):
            s_pred = sampler(
                noise=noise,
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, # reference from the same speaker as the embedding
                num_steps=diffusion_steps).squeeze(1)

        s, ref = blend_predict_and_reference(s_pred, ref_s, alpha, beta)

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))

        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = get_input_for_prosody(d.transpose(-1, -2), pred_aln_trg, torch.float32)
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        asr = get_input_for_prosody(t_en, pred_aln_trg, USE_DTYPE)
        with Timed("model.decoder"):
            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later


###

def main():
  """### Synthesize speech

  #### Basic synthesis (5 diffusion steps, seen speakers)
  """

  text = "This is a mac computer! And StyleTTS 2 is a text to speech model that " + \
         "leverages style diffusion and adversarial training with large speech language models " + \
        "to achieve human level text to speech synthesis." # @param {type:"string"}
  text = " This is a test of the emergency broadcast system. "

  import subprocess
  reference_dicts = {}
  #reference_dicts['696_92939'] = "Demo/reference_audio/696_92939_000016_000006.wav"
  #reference_dicts['1789_142896'] = "Demo/reference_audio/1789_142896_000022_000005.wav"
  #reference_dicts['1221-135767'] = "Demo/reference_audio/1221-135767-0014.wav"
  #reference_dicts['908-157963-0027'] = "Demo/reference_audio/908-157963-0027.wav"
  #reference_dicts['matt2'] = "matt2.wav"
  #reference_dicts['attenb2'] = "audios/attenb2.wav"
  #reference_dicts['trent_84_6s'] = "audios/trent_84_6s.wav"
  reference_dicts['eminem'] = "audios/eminem3.wav"

  #This setting uses 70% of the reference timbre and 30% of the reference prosody
  # alpha=0.3, beta=0.7,
  print(GPU_DEVICE)
  styles = {}
  import soundfile as sf
  style_k = list(reference_dicts.keys())[0]
  style_path = reference_dicts[style_k]
  texts = [text]
  while True:
    # for style_k, style_path in reference_dicts.items()
    for text in texts:
        print(style_k)
        with Timed("synthesize audio from text"):
            ref_s = styles.get(style_k)
            if ref_s is None:
                ref_s = compute_style(style_path)
            
            start = time.time()
            if style_k == 'matt':
                wav = inference(text, ref_s, alpha=0.1, beta=0.3, diffusion_steps=10, embedding_scale=1)
            else:
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=False) as prof:
                    with record_function("model_inference"):
                        wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)

            inference_s = (time.time() - start)
            duration_maybe = len(wav) / 24000
            rtf = inference_s / duration_maybe
            print(f"infer:duration ratio = {rtf:.2f} ; {inference_s:.2f} ; duration? {duration_maybe:.2f}")

            #torchaudio.save("./synthesized.wav", wav, 24000)

            sf.write("./synthesized.wav", wav, 24000)
            print(style_k + ' Synthesized:')
            #display(ipd.Audio(wav, rate=24000, normalize=False))
            print('Reference:')
            #display(ipd.Audio(path, rate=24000, normalize=False))
            # subprocess to afplay
        

        subprocess.run(["afplay", "./synthesized.wav"])
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=12))
        #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=12))
        print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
        prof.export_chrome_trace("trace.json")


    user_input = input("Enter text: ")
    parts = user_input.split('|')
    if len(parts) > 1:
        style_path = parts[0]
        style_k = style_path.split(".")[0]
        text = parts[1]
    else:
        text = parts[0]
    texts = text.strip().split('. ')

if __name__ == "__main__":
    Timed.setup_interrupt_handler()
    main()
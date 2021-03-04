import os
import numpy as np
import scipy.io.wavfile as wavfile
import librosa
import torch
import hifigan
import json

from hparams import create_hparams
from model import load_model
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
from data.sentences import sentences

hparams = create_hparams()

stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                    hparams.mel_fmax)

def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2) / 2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2) / 2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]


def load_mel(path):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec

def get_vocoder():
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("hifigan/generator_universal.pth.tar")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.cuda()

    return vocoder


def vocoder_infer(mel, vocoder, path):
    wav = vocoder(mel).squeeze(1)

    wav = (
        wav.squeeze().cpu().numpy()
        * hparams.max_wav_value
    ).astype("int16")

    wavfile.write(path, hparams.sampling_rate, wav)

    return wav


def inference(dirname):
    # 멜로트론 로딩
    checkpoint_path = "models/checkpoint_171000"
    mellotron = load_model(hparams).cuda().eval()
    mellotron.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    # 보코더 로딩
    vocoder = get_vocoder()

    # 오디오 filelist 로딩
    arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
    audio_paths = f'data/{dirname}.txt'
    dataloader = TextMelLoader(audio_paths, hparams)
    datacollate = TextMelCollate(1)
    os.makedirs(f'sample/{dirname}', exist_ok=True)

    for sent_txt in sentences:
        for file_idx in range(len(dataloader)):
            audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]
            text = sent_txt

            # get audio path, encoded text, pitch contour and mel for gst
            text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
            pitch_contour = dataloader[file_idx][3][None].cuda()
            mel = load_mel(audio_path)
            print(audio_path, text)

            # load source data to obtain rhythm using tacotron 2 as a forced aligner
            x, y = mellotron.parse_batch(datacollate([dataloader.get_data((audio_path, text, sid))]))

            # 스피커 id
            speaker_id = torch.LongTensor([int(sid)]).cuda()

            # 멜로트론 합성
            with torch.no_grad():
                # get rhythm (alignment map) using tacotron 2
                mel_outputs, mel_outputs_postnet, gate_outputs, rhythm = mellotron.forward(x)
                rhythm = rhythm.permute(1, 0, 2)

                # mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
                #     (text_encoded, mel, speaker_id, pitch_contour, rhythm))
                mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference(
                    (text_encoded, mel, speaker_id, pitch_contour))

                # wav 합성
                sample_name = f'{os.path.splitext(os.path.basename(audio_path))[0]}-{text}.wav'
                vocoder_infer(mel_outputs_postnet, vocoder, f'sample/{dirname}/{sample_name}')

if __name__ == '__main__':
    inference('VCTK_val_reference')
    inference('VCTK_val_reference_noisy')
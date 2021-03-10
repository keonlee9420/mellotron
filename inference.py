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

speaker_id_map = {
    100000: 0, 100001: 1, 100002: 2, 100003: 3, 100005: 4, 100006: 5, 100007: 6, 100008: 7, 100009: 8, 100011: 9,
    100012: 10, 100014: 11, 100015: 12, 100016: 13, 100017: 14, 100018: 15, 100019: 16, 100020: 17, 100021: 18,
    100022: 19, 100023: 20, 100024: 21, 100025: 22, 100026: 23, 100027: 24, 100028: 25, 100029: 26, 100030: 27,
    100031: 28, 100032: 29, 100033: 30, 100034: 31, 100035: 32, 100036: 33, 100037: 34, 100038: 35, 100039: 36,
    100041: 37, 100043: 38, 100044: 39, 100045: 40, 100046: 41, 100047: 42, 100048: 43, 100049: 44, 100050: 45,
    100051: 46, 100054: 47, 100055: 48, 100056: 49, 100057: 50, 100059: 51, 100060: 52, 100061: 53, 100062: 54,
    100063: 55, 100065: 56, 100066: 57, 100067: 58, 100068: 59, 100069: 60, 100071: 61, 100073: 62, 100074: 63,
    100075: 64, 100076: 65, 100077: 66, 100079: 67, 100080: 68, 100081: 69, 100082: 70, 100084: 71, 100086: 72,
    100088: 73, 100089: 74, 100091: 75, 100092: 76, 100093: 77, 100095: 78, 100097: 79, 100099: 80, 100100: 81,
    100102: 82, 100104: 83, 100105: 84, 100106: 85, 100107: 86, 100108: 87, 100109: 88}

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


def inference(dirname, outdir, checkpoint_path, parallel=False):
    # 멜로트론 로딩
    mellotron = load_model(hparams).cuda().eval()
    mellotron.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    # 보코더 로딩
    vocoder = get_vocoder()

    # 오디오 filelist 로딩
    arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
    audio_paths = f'data/{dirname}.txt'
    dataloader = TextMelLoader(audio_paths, hparams, speaker_ids=speaker_id_map)
    os.makedirs(f'{outdir}/{os.path.basename(checkpoint_path)}/{dirname}', exist_ok=True)

    with open('data/VCTK/speaker-dict.json') as f:
        speakers = json.load(f)

    new_filelist = []
    for file_idx in range(len(dataloader)):
        audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]
        print(sid)
        if not parallel:
            for sent_txt in sentences:
                text = sent_txt

                # get audio path, encoded text, pitch contour and mel for gst
                text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
                pitch_contour = dataloader[file_idx][3][None].cuda()
                mel = load_mel(audio_path)
                print(audio_path, text)

                # 스피커 id
                # speaker_name = os.path.basename(audio_path).split('_')[1]
                # speaker_id = speakers.index(speaker_name)
                speaker_id = int(sid)
                speaker_id_mapped = speaker_id_map[speaker_id]
                speaker_id = torch.LongTensor([speaker_id_mapped]).cuda()

                # 멜로트론 합성
                with torch.no_grad():
                    mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference(
                        (text_encoded, mel, speaker_id, pitch_contour))

                    # wav 합성
                    sample_name = f'{os.path.splitext(os.path.basename(audio_path))[0]}-{text}.wav'
                    vocoder_infer(mel_outputs_postnet, vocoder,
                                  f'{outdir}/{os.path.basename(checkpoint_path)}/{dirname}/{sample_name}')

                new_filelist.append(f'{outdir}/{os.path.basename(checkpoint_path)}/{dirname}/{sample_name}\n')
        else:
            # get audio path, encoded text, pitch contour and mel for gst
            text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()
            pitch_contour = dataloader[file_idx][3][None].cuda()
            mel = load_mel(audio_path)
            print(audio_path, text)

            # 스피커 id
            # speaker_name = os.path.basename(audio_path).split('_')[1]
            # speaker_id = speakers.index(speaker_name)
            speaker_id = int(sid)
            speaker_id_mapped = speaker_id_map[speaker_id]
            speaker_id = torch.LongTensor([speaker_id_mapped]).cuda()

            # 멜로트론 합성
            with torch.no_grad():
                mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference(
                    (text_encoded, mel, speaker_id, pitch_contour))

                # wav 합성
                sample_name = f'{os.path.splitext(os.path.basename(audio_path))[0]}-{text}.wav'
                vocoder_infer(mel_outputs_postnet, vocoder,
                              f'{outdir}/{os.path.basename(checkpoint_path)}/{dirname}/{sample_name}')

            new_filelist.append(f'{outdir}/{os.path.basename(checkpoint_path)}/{dirname}/{sample_name}\n')

    with open(f'{outdir}/{os.path.basename(checkpoint_path)}/{dirname}.txt', 'w') as f:
        f.writelines(new_filelist)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='models/checkpoint_550000')
    parser.add_argument('--outdir', type=str, default='sample')
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    for dirname in ['VCTK_val_reference', 'VCTK_val_reference_noisy']:
        inference(dirname, args.outdir, args.checkpoint_path, args.parallel)

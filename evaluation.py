import os
from librosa.sequence import dtw
from scipy.io.wavfile import read
import numpy as np
import torch
from yin import compute_yin
from hparams import create_hparams
from layers import TacotronSTFT

class Evaluator:
    hparams = create_hparams()
    filter_length = hparams.filter_length
    hop_length = hparams.hop_length
    win_length = hparams.win_length
    n_mel_channels = hparams.n_mel_channels
    sampling_rate = hparams.sampling_rate
    mel_fmin = hparams.mel_fmin
    mel_fmax = hparams.mel_fmax
    f0_min = hparams.f0_min
    f0_max = hparams.f0_max
    harm_thresh = hparams.harm_thresh
    max_wav_value = hparams.max_wav_value
    stft = TacotronSTFT(filter_length, hop_length, win_length,
                        n_mel_channels, sampling_rate, mel_fmin, mel_fmax)

    @staticmethod
    def load_wav_to_torch(full_path):
        sampling_rate, data = read(full_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate

    @staticmethod
    def get_f0(audio, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=100, f0_max=300, harm_thresh=0.1):
        f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh)
        pad = int((frame_length / hop_length) / 2)
        f0 = [0.0] * pad + f0 + [0.0] * pad

        f0 = np.array(f0, dtype=np.float32)
        return f0

    def get_mel_and_f0(self, filepath):
        audio, sampling_rate = self.load_wav_to_torch(filepath)
        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        f0 = self.get_f0(audio.cpu().numpy(), sampling_rate,
                         self.filter_length, self.hop_length, self.f0_min,
                         self.f0_max, self.harm_thresh)
        f0 = torch.from_numpy(f0)[None]
        f0 = f0[:, :melspec.size(1)]

        return melspec, f0

    @staticmethod
    def align_dtw(true, gst, dim):
        _, idx = dtw(X=true, Y=gst, backtrack=True)
        idx_t = idx.transpose()
        true_idx = np.flip(idx_t[0])
        gst_idx = np.flip(idx_t[1])
        true = true.transpose(0, 1)
        gst = gst.transpose(0, 1)

        warped_true_mel = np.zeros((len(idx_t[0]), dim))
        warped_gst_mel = np.zeros((len(idx_t[0]), dim))
        for i in range(len(idx_t[0])):
            warped_true_mel[i] = true[true_idx[i]]
            warped_gst_mel[i] = gst[gst_idx[i]]
        return warped_true_mel, warped_gst_mel  # (B, T, dim) not padded yet

    @staticmethod
    def calculate_gpe(f0_target, f0_out):
        f0_out_tensor = torch.from_numpy(f0_out).squeeze()
        f0_target_tensor = torch.from_numpy(f0_target).squeeze()

        out_voiced_mask = f0_out_tensor != 0
        target_voiced_mask = f0_target_tensor != 0
        diff_abs = (f0_out_tensor - f0_target_tensor).abs()
        erronous_prediction_mask = diff_abs > (0.2 * f0_target_tensor)

        denominator = out_voiced_mask * target_voiced_mask
        numerator = denominator * erronous_prediction_mask
        # denominator = out_voiced_mask * target_voiced_mask

        numerator = torch.FloatTensor([numerator.sum()])
        denominator = torch.FloatTensor([denominator.sum()])
        loss = numerator / (denominator + 1e-9)
        return loss

    @staticmethod
    def calculate_vde(f0_target, f0_out):
        f0_out_tensor = torch.from_numpy(f0_out).squeeze()
        f0_target_tensor = torch.from_numpy(f0_target).squeeze()

        out_voicing_decision = f0_out_tensor != 0
        target_voicing_decision = f0_target_tensor != 0

        mismatched_voicing_decision_mask = out_voicing_decision != target_voicing_decision
        numerator = torch.FloatTensor([mismatched_voicing_decision_mask.sum()])

        denominator = torch.FloatTensor([f0_target.shape[0]])

        loss = numerator / denominator
        return loss

    @staticmethod
    def calculate_ffe(f0_target, f0_out):
        f0_out = torch.from_numpy(f0_out).squeeze()
        f0_target = torch.from_numpy(f0_target).squeeze()
        out_voiced_mask = f0_out != 0
        target_voiced_mask = f0_target != 0
        diff_abs = (f0_out - f0_target).abs()
        erronous_prediction_mask = diff_abs > 0.2 * f0_target

        denominator = torch.FloatTensor([f0_target.shape[0]])
        numerator1 = out_voiced_mask * target_voiced_mask * erronous_prediction_mask
        numerator1 = numerator1.sum()
        numerator2 = out_voiced_mask != target_voiced_mask
        numerator2 = numerator2.sum()
        numerator = torch.FloatTensor([numerator1 + numerator2])
        loss = numerator / (
            denominator)  # removed adding 1e-3 to denominator because it seems unlikely for denominator to be zero
        return loss

    @staticmethod
    def calculate_mcd(target_mel, out_mel):
        # MCD13: mse along 13 dims. Exclude 0th mel to make it indifferent of overall energy scale.
        # Use unpadded true lens for denominator
        out_mel = torch.from_numpy(out_mel).squeeze()[:, 1:14]
        target_mel = torch.from_numpy(target_mel).squeeze()[:, 1:14]
        diff = out_mel - target_mel
        diff_sq = diff ** 2
        tmp = diff_sq.sum(dim=-1).squeeze()
        tmp = torch.sqrt(tmp)
        tmp = tmp.sum()
        numerator = tmp
        denominator = torch.FloatTensor([target_mel.shape[0]])
        mcd = numerator / denominator
        # Google's work does not multiply K
        # "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
        # K = 10 / np.log(10) * np.sqrt(2)
        return mcd

    def evaluate(self, ref_filelist, pred_filelist):
        # Load reference wavs
        with open(ref_filelist) as f:
            ref_wavs = f.readlines()
            ref_wavs = [r.split('|')[0].strip() for r in ref_wavs]

        # Load predicted wavs
        with open(pred_filelist) as f:
            pred_wavs = f.readlines()
            pred_wavs = [r.split('|')[0].strip() for r in pred_wavs]

        # Compare predicted wav with reference wav
        all_gpe, all_vde, all_ffe, all_mcd = [], [], [], []
        cnt = 0
        for pred_wav_path in pred_wavs:
            ref_wav_valid = [r for r in ref_wavs if os.path.splitext(os.path.basename(r))[0] in pred_wav_path]
            assert len(ref_wav_valid) == 1
            ref_wav_path = ref_wav_valid[0]
            print(pred_wav_path, ref_wav_path)

            pred_mel, pred_f0 = self.get_mel_and_f0(pred_wav_path)
            ref_mel, ref_f0 = self.get_mel_and_f0(ref_wav_path)

            warped_ref_mel, warped_pred_mel = self.align_dtw(ref_mel, pred_mel, 80)
            warped_ref_f0, warped_pred_f0 = self.align_dtw(ref_f0, pred_f0, 1)

            gpe = self.calculate_gpe(warped_ref_f0, warped_pred_f0)
            vde = self.calculate_vde(warped_ref_f0, warped_pred_f0)
            ffe = self.calculate_ffe(warped_ref_f0, warped_pred_f0)
            mcd = self.calculate_mcd(warped_ref_mel, warped_pred_mel)

            all_gpe.append(gpe.item())
            all_vde.append(vde.item())
            all_ffe.append(ffe.item())
            all_mcd.append(mcd.item())
            cnt += 1

        print(f"GPE: {sum(all_gpe) / cnt:.4f}")
        print(f"VDE: {sum(all_vde) / cnt:.4f}")
        print(f"FFE: {sum(all_ffe) / cnt:.4f}")
        print(f"MCD: {sum(all_mcd) / cnt:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='data/VCTK_val_reference.txt')
    parser.add_argument('--pred', type=str, default='sample_parallel/VCTK_val_reference.txt')

    args = parser.parse_args()

    evaluator = Evaluator()
    evaluator.evaluate(args.ref, args.pred)

import argparse
import os
import random
from glob import glob

def build_vctk_filelist(path):
    with open(os.path.join(path, 'speaker-info.txt')) as f:
        speaker_lines = f.readlines()[1:]

    speakers = [l.split()[0].strip() for l in speaker_lines]
    speakers = [s for s in speakers if s.startswith('p') or s.startswith('s')]

    wavs = glob(os.path.join(path, '**/*.wav'), recursive=True)
    with open('vctk/train.txt', 'r') as f:
        train_list = f.readlines()
    with open('vctk/val.txt', 'r') as f:
        val_list = f.readlines()
    with open('vctk/test.txt', 'r') as f:
        test_list = f.readlines()
    train_list = [x.split('|')[0].strip() for x in train_list]
    val_list = [x.split('|')[0].strip() for x in val_list]
    test_list = [x.split('|')[0].strip() for x in test_list]

    flist_train, flist_val, flist_test = [], [], []
    for wav in wavs:
        try:
            with open(wav.replace('.wav', '.txt')) as txtf:
                script = txtf.read().strip()
        except FileNotFoundError:
            continue
        speaker_id = os.path.basename(wav).split('_')[0].strip()
        line = f'{os.path.abspath(wav)}|{script}|{speakers.index(speaker_id)}\n'

        wav_name = os.path.splitext(os.path.basename(wav))[0]
        if wav_name in train_list:
            flist_train.append(line)
        elif wav_name in val_list:
            flist_val.append(line)
        elif wav_name in test_list:
            flist_test.append(line)
        else:
            pass

    with open('./vctk_filelist_train.txt', 'w') as f:
        f.writelines(flist_train)
    with open('./vctk_filelist_val.txt', 'w') as f:
        f.writelines(flist_val)
    with open('./vctk_filelist_test.txt', 'w') as f:
        f.writelines(flist_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()

    build_vctk_filelist(args.path)

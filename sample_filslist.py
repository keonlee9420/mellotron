import os
import json
from glob import glob

def create_file(filelist, wavdir):
    wavs = glob(os.path.join(wavdir, '*.wav'))

    with open('data/VCTK/speaker-dict.json') as f:
        speaker_dict = json.load(f)

    lines = []
    for wav in wavs:
        txt = wav.replace('.wav', '.txt')
        try:
            with open(txt) as t:
                script = t.read().strip()
        except FileNotFoundError:
            script = 'hello world, empty script'

        speaker_id = 0
        if 'vctk' in wav:
            for spk in speaker_dict.keys():
                if spk in wav:
                    speaker_id = speaker_dict[spk]
                    break

        line = f'{wav}|{script}|{speaker_id}\n'
        lines.append(line)

    with open(filelist, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    create_file('sample/filelist_vctk_val.txt', 'sample/vctk/')
    create_file('sample/filelist_nonparallel.txt', 'sample/NonParallelRefs/')




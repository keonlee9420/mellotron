import os
from glob import glob

def create_file(filelist, wavdir):
    wavs = glob(os.path.join(wavdir, '*.wav'))

    with open('data/vctk-speaker-info.txt') as f:
        speaker_lines = f.readlines()[1:]

    speakers = [l.split()[0].strip() for l in speaker_lines]

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
            for spk in speakers:
                if spk in wav:
                    speaker_id = speakers.index(spk)
                    break

        line = f'{wav}|{script}|{speaker_id}\n'
        lines.append(line)

    with open(filelist, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    create_file('sample/filelist_vctk_val.txt', 'sample/vctk/')
    create_file('sample/filelist_nonparallel.txt', 'sample/NonParallelRefs/')




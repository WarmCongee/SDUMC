#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, glob, argparse, shutil
from scipy.io import wavfile

def replace():
    os.makedirs('/disk3/htwang/MER2023-Baseline-master/dataset-process/audio-test2-pl-inter')
    all_wav_files = glob.glob(os.path.join('/disk3/htwang/MER2023-Baseline-master/dataset-process/audio-test2-clean', '*_inter.wav'))
    for wav_file in all_wav_files:
        new_wav_file = wav_file.replace('audio-test2-clean', 'audio-test2-pl-inter').replace('_inter.wav', '.wav')
        shutil.copyfile(wav_file, new_wav_file)
    return None

def index_wav(wav_dir):
    all_wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
    index_dic = {}
    for wav_file in all_wav_files:
        name = os.path.splitext(os.path.split(wav_file)[-1])[0]
        index_dic[name] = wav_file
    return index_dic

def average_fusion_wav(fused_wav, base_wav, fusing_wavs):
    fused_dir = os.path.split(fused_wav)[0]
    if not os.path.exists(fused_dir):
        os.makedirs(fused_dir, exist_ok=True)
    if len(fusing_wavs) == 0:
        shutil.copyfile(base_wav, fused_wav)
    else:
        print('fusing')
        fusing_num = 1 + len(fusing_wavs)
        base_fps, base_data = wavfile.read(base_wav)
        base_data = base_data.astype('float64') / fusing_num
        for fusing_wav in fusing_wavs:
            fusing_fps, fusing_data = wavfile.read(fusing_wav)
            fusing_data = fusing_data.astype('float64') / fusing_num
            assert base_fps == fusing_fps
            base_data = base_data + fusing_data
        wavfile.write(filename=fused_wav, rate=base_fps, data=base_data.astype('int16'))
    return None


def main_fusing(fused_dir, base_dir, fusing_dirs):
    base_index_dic = index_wav(base_dir)
    keys = [*base_index_dic.keys()]
    fusing_index_dics = []
    
    for fusing_dir in fusing_dirs:
        fusing_index_dic = index_wav(fusing_dir)
        assert set(keys) == set([*fusing_index_dic])
        fusing_index_dics.append(fusing_index_dic)
    
    for key in keys:
        average_fusion_wav(os.path.join(fused_dir, '{}.wav'.format(key)), base_wav=base_index_dic[key], fusing_wavs=[i[key] for i in fusing_index_dics])
    return None


if __name__ == '__main__':
    main_fusing(
        fused_dir='/disk3/htwang/MER2023-Baseline-master/dataset-process/audio-test2-mease+mtmease+plmease', 
        base_dir='/disk3/htwang/MER2023-Baseline-master/dataset-process/audio-test2-mease', 
        fusing_dirs=['/disk3/htwang/MER2023-Baseline-master/dataset-process/audio-test2-mtmease', 
                     '/disk3/htwang/MER2023-Baseline-master/dataset-process/audio-test2-plmease'])
    # replace()

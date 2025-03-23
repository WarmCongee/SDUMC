#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, argparse, random, json, codecs, glob, tqdm
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def crop_one_sample(pic_dir, store_name, store_dir):
    frame_paths = sorted(glob.glob(os.path.join(pic_dir,'*')))
    # print(pic_dir, len(frame_paths))
    frames = []
    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        width, height, channel = img.shape
        left = width / 4
        top = height / 2
        right = 3 * width / 4
        bottom = height
        img_croped = img[int(top):int(bottom), int(left):int(right), :]
        img_croped_resized = cv2.resize(img_croped, [96, 96])
        frames.append(torch.from_numpy(img_croped_resized).permute(2, 1, 0))
    
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    torch.save(torch.stack(frames, dim=0), os.path.join(store_dir, '{}.pt'.format(store_name)))
    return None


def main_crop(input_dir, output_dir):
    all_pic_dirs = os.listdir(input_dir)
    for pic_dir in tqdm.tqdm(all_pic_dirs):
        crop_one_sample(os.path.join(input_dir, pic_dir), store_name=pic_dir, store_dir=output_dir)
    return None


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run.')
    # parser.add_argument('--feature_dir', type=str, default='0', help='index of gpu')
    # parser.add_argument('--mix_num', type=int, default=1, help='index of gpu')
    # parser.add_argument('--mix_probability', type=float, default=0.5, help='index of gpu')
    # parser.add_argument('--alpha', type=float, default=0.2, help='index of gpu')
    # args = parser.parse_args()
    
    main_crop(input_dir='/disk3/htwang/MER2023-Baseline-master/dataset-process/features-test2/openface_face', 
              output_dir='/disk3/htwang/MER2023-Baseline-master/dataset-process/features-test2/crop-lip')
# python feature_extraction/generate_mix_file.py --feature_dir dataset-process/mixup_202307041119 --mix_num 1 --mix_probability 0.5 --alpha 0.5

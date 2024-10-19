import os
import cv2
import math
import time
import argparse
import numpy as np
from PIL import Image

import torch 
from torch.utils.data import Dataset, DataLoader

import timm # pip install timm==0.9.7
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor

# import config
import sys
sys.path.append('../../')
import config

##################### Pretrained models #####################
CLIP_VIT_BASE = 'clip-vit-base-patch32'  # https://huggingface.co/openai/clip-vit-base-patch32
CLIP_VIT_LARGE = 'clip-vit-large-patch14' # https://huggingface.co/openai/clip-vit-large-patch14
EVACLIP_VIT = 'eva02_base_patch14_224.mim_in22k' # https://huggingface.co/timm/eva02_base_patch14_224.mim_in22k
DATA2VEC_VISUAL = 'data2vec-vision-base-ft1k' # https://huggingface.co/facebook/data2vec-vision-base-ft1k
VIDEOMAE_BASE = 'videomae-base' # https://huggingface.co/MCG-NJU/videomae-base
VIDEOMAE_LARGE = 'videomae-large' # https://huggingface.co/MCG-NJU/videomae-large
DINO2_LARGE = 'dinov2-large' # https://huggingface.co/facebook/dinov2-large
DINO2_GIANT = 'dinov2-giant' # https://huggingface.co/facebook/dinov2-giant


def func_opencv_to_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def func_opencv_to_numpy(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def func_read_frames(face_dir, vid):
    npy_path  = os.path.join(face_dir, vid, f'{vid}.npy')
    assert os.path.exists(npy_path), f'Error: {vid} does not have frames.npy!'
    frames = np.load(npy_path)
    return frames

def func_read_frames_imgs(face_dir, vid):
    set_path  = os.path.join(face_dir, vid)
    assert os.path.exists(set_path), f'Error: {vid} does not have frames!'
    frames_path = []
    imgs = []
    for root, dirs, files in os.walk(set_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            frames_path.append(file_path)
            img = cv2.imread(file_path,0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
    return imgs

class ImgDataset(Dataset):
    """ 
        Data processing using albumentation same as torchvision transforms
    """
    def __init__(self, face_dir, vids):
        # here can control the dataset size percentage  
        self.root = face_dir
        self.names = vids
        
    def __getitem__(self, index):
        vid = self.names[index]
        set_path  = os.path.join(face_dir, vid)
        assert os.path.exists(set_path), f'Error: {vid} does not have frames!'
        imgs = []
        for root, dirs, files in os.walk(set_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                img = cv2.imread(file_path,0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
        # imgs = resample_frames(imgs, 5)
        return imgs, vid

    def __len__(self):
        return len(self.names)

# 策略3：相比于上面采样更加均匀 [将videomae替换并重新测试]
def resample_frames_uniform(frames, nframe=16):
    vlen = len(frames)
    start, end = 0, vlen
    
    n_frms_update = min(nframe, vlen) # for vlen < n_frms, only read vlen
    indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    
    # whether compress into 'n_frms'
    while len(indices) < nframe:
        indices.append(indices[-1])
    indices = indices[:nframe]
    assert len(indices) == nframe, f'{indices}, {vlen}, {nframe}'
    return frames[indices]

def resample_frames(frames, step=5):
    vlen = len(frames)
    start, end = 0, vlen
    
    n_frms_update = min(step, vlen) # for vlen < n_frms, only read vlen
    indices = np.arange(start, end, 5).astype(int).tolist()

    return [frames[i] for i in indices]
    
def split_into_batch(inputs, bsize=32):
    batches = []
    for ii in range(math.ceil(len(inputs)/bsize)):
        batch = inputs[ii*bsize:(ii+1)*bsize]
        batches.append(batch)
    return batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, default='MER2023', help='input dataset')
    parser.add_argument('--model_name', type=str, default=None, help='name of pretrained model')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', help='feature level [FRAME or UTTERANCE]')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    params = parser.parse_args()

    print(f'==> Extracting {params.model_name} embeddings...')
    model_name = params.model_name.split('.')[0]
    face_dir = config.PATH_TO_RAW_FACE[params.dataset]

    # gain save_dir
    save_dir = os.path.join(config.PATH_TO_FEATURES[params.dataset], f'{model_name}-{params.feature_level[:3]}')
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model
    if params.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE, DATA2VEC_VISUAL, VIDEOMAE_BASE, VIDEOMAE_LARGE]: # from huggingface
        model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{params.model_name}')
        model = AutoModel.from_pretrained(model_dir)
        processor  = AutoFeatureExtractor.from_pretrained(model_dir)
    elif params.model_name in [DINO2_LARGE, DINO2_GIANT]:
        model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{params.model_name}')
        model = AutoModel.from_pretrained(model_dir)
        processor  = AutoImageProcessor.from_pretrained(model_dir)
    elif params.model_name in [EVACLIP_VIT]: # from timm
        model_path = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'timm/{params.model_name}/model.safetensors')
        model = timm.create_model(params.model_name, pretrained=True, num_classes=0,pretrained_cfg_overlay=dict(file=model_path))
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    # 有 gpu 才会放在cuda上
    if params.gpu != -1:
        torch.cuda.set_device(params.gpu)
        model.cuda()
    model.eval()

    # extract embedding video by video
    vids = os.listdir(face_dir)
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    imgs_dataset = ImgDataset(face_dir, vids)
    dataloader = DataLoader(imgs_dataset,
                        batch_size=1,
                        num_workers=8,
                        pin_memory=True,
                        shuffle=False,
                        prefetch_factor=8)
    with torch.no_grad():
        for data, vid in dataloader:
            vid = vid[0]
            
            data = [item[0] for item in data]
            # a = time.time()
            inputs = processor(images=data, return_tensors="pt")['pixel_values']
            if params.gpu != -1: inputs = inputs.to("cuda")
            # b = time.time()
            # print(b-a)
            batches = split_into_batch(inputs, bsize=512)
            embeddings = []
            for batch in batches:
                embeddings.append(model.get_image_features(batch)) # [58, 768]
            embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]
            embeddings = embeddings.detach().squeeze().cpu().numpy()
            EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

            # save into npy
            save_file = os.path.join(save_dir, f'{vid}.npy')
            print(save_file)
            if params.feature_level == 'FRAME':
                embeddings = np.array(embeddings).squeeze()
                if len(embeddings) == 0:
                    embeddings = np.zeros((1, EMBEDDING_DIM))
                elif len(embeddings.shape) == 1:
                    embeddings = embeddings[np.newaxis, :]
                    print(embeddings.shape)
                np.save(save_file, embeddings)
            else:
                embeddings = np.array(embeddings).squeeze()
                if len(embeddings) == 0:
                    embeddings = np.zeros((EMBEDDING_DIM, ))
                elif len(embeddings.shape) == 2:
                    embeddings = np.mean(embeddings, axis=0)
                np.save(save_file, embeddings)


    # for i, vid in enumerate(vids, 1):
    #     print(f"Processing video '{vid}' ({i}/{len(vids)})...")
       
    #     # forward process [different model has its unique mode, it is hard to unify them as one process]
    #     # => split into batch to reduce memory usage
    #     with torch.no_grad():
    #         if params.model_name in [CLIP_VIT_BASE, CLIP_VIT_LARGE]:
    #             frames = func_read_frames_imgs(face_dir,vid)
    #             inputs = processor(images=frames, return_tensors="pt")['pixel_values']
    #             if params.gpu != -1: inputs = inputs.to("cuda")
    #             batches = split_into_batch(inputs, bsize=32)
    #             embeddings = []
    #             for batch in batches:
    #                 embeddings.append(model.get_image_features(batch)) # [58, 768]
    #             embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]

    #         elif params.model_name in [DATA2VEC_VISUAL]:
    #             frames = func_read_frames(face_dir, vid)
    #             frames = [func_opencv_to_image(frame) for frame in frames]
    #             inputs = processor(images=frames, return_tensors="pt")['pixel_values'] # [nframe, 3, 224, 224]
    #             if params.gpu != -1: inputs = inputs.to("cuda")
    #             batches = split_into_batch(inputs, bsize=32)
    #             embeddings = []
    #             for batch in batches: # [32, 3, 224, 224]
    #                 hidden_states = model(batch, output_hidden_states=True).hidden_states # [58, 196 patch + 1 cls, feat=768]
    #                 embeddings.append(torch.stack(hidden_states)[-1].sum(dim=1)) # [58, 768]
    #             embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]

    #         elif params.model_name in [DINO2_LARGE, DINO2_GIANT]:
    #             frames = func_read_frames(face_dir, vid)
    #             frames = resample_frames_uniform(frames, nframe=64) # 加速特征提起：这种方式更加均匀的采样64帧
    #             frames = [func_opencv_to_image(frame) for frame in frames]
    #             inputs = processor(images=frames, return_tensors="pt")['pixel_values'] # [nframe, 3, 224, 224]
    #             if params.gpu != -1: inputs = inputs.to("cuda")
    #             batches = split_into_batch(inputs, bsize=32)
    #             embeddings = []
    #             for batch in batches: # [32, 3, 224, 224]
    #                 hidden_states = model(batch, output_hidden_states=True).hidden_states # [58, 196 patch + 1 cls, feat=768]
    #                 embeddings.append(torch.stack(hidden_states)[-1].sum(dim=1)) # [58, 768]
    #             embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]

    #         elif params.model_name in [VIDEOMAE_BASE, VIDEOMAE_LARGE]:
    #             # videoVAE: only supports 16 frames inputs
    #             frames = func_read_frames(face_dir, vid)
    #             batches = [resample_frames_uniform(frames)] # convert to list of batches
    #             embeddings = []
    #             for batch in batches:
    #                 frames = [func_opencv_to_numpy(frame) for frame in batch] # 16 * [112, 112, 3]
    #                 inputs = processor(list(frames), return_tensors="pt")['pixel_values'] # [1, 16, 3, 224, 224]
    #                 if params.gpu != -1: inputs = inputs.to("cuda")
    #                 outputs = model(inputs).last_hidden_state # [1, 1586, 768]
    #                 num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2 # 14*14
    #                 outputs = outputs.view(16 // model.config.tubelet_size, num_patches_per_frame, -1) # [seg_number, patch, featdim]
    #                 embeddings.append(outputs.mean(dim=1)) # [seg_number, featdim]
    #             embeddings = torch.cat(embeddings, axis=0)

    #         elif params.model_name in [EVACLIP_VIT]:
    #             frames = func_read_frames(face_dir, vid)
    #             frames = [func_opencv_to_image(frame) for frame in frames]
    #             inputs = torch.stack([transforms(frame) for frame in frames]) # [117, 3, 224, 224]
    #             if params.gpu != -1: inputs = inputs.to("cuda")
    #             batches = split_into_batch(inputs, bsize=32)
    #             embeddings = []
    #             for batch in batches:
    #                 embeddings.append(model(batch)) # [58, 768]
    #             embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]
    
    #     embeddings = embeddings.detach().squeeze().cpu().numpy()
    #     EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

    #     # save into npy
    #     save_file = os.path.join(save_dir, f'{vid}.npy')
    #     if params.feature_level == 'FRAME':
    #         embeddings = np.array(embeddings).squeeze()
    #         if len(embeddings) == 0:
    #             embeddings = np.zeros((1, EMBEDDING_DIM))
    #         elif len(embeddings.shape) == 1:
    #             embeddings = embeddings[np.newaxis, :]
    #         np.save(save_file, embeddings)
    #     else:
    #         embeddings = np.array(embeddings).squeeze()
    #         if len(embeddings) == 0:
    #             embeddings = np.zeros((EMBEDDING_DIM, ))
    #         elif len(embeddings.shape) == 2:
    #             embeddings = np.mean(embeddings, axis=0)
    #         np.save(save_file, embeddings)
 
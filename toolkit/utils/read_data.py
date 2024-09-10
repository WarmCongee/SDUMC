import os
import glob
import tqdm
import math
import pickle
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from ..globals import *
from .functions import *
from .read_files import *

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

############################################################
# ------ for feat: feature_root+name -> (seqlen, featdim) ------
def func_read_one_feat(argv=None, feature_root=None, name=None, processor=None, model_name=None):
    feature_root, name, processor, model_name = argv

    # 路径可能的两个选项
    feature_path = os.path.join(feature_root, name+'.npy')
    feature_dir  = os.path.join(feature_root, name)

    feature = []
    if os.path.exists(feature_path): # audio/text => belong to speaker
        single_feature = np.load(feature_path)
        single_feature = single_feature.squeeze() # [Dim, ] or [Time, Dim]
        feature.append(single_feature)
    elif os.path.isdir(feature_dir):
        facenames = os.listdir(feature_dir) # 如果是文件夹，则依次读取文件夹内所有信息
        for facename in sorted(facenames):
            facefeat = np.load(os.path.join(feature_dir, facename))
            feature.append(facefeat)
    else:
        print(feature_path)
        raise Exception('feature path or dir do not exist!')

    # feature -> (seqlen, featdim)
    single_feature = np.array(feature).squeeze()
    if len(single_feature) == 0:
        print ('feature has errors!!')
    elif len(single_feature.shape) == 1:
        single_feature = single_feature[np.newaxis, :]
    return single_feature


def func_read_one_feat_lmdb(argv=None, feature_root=None, name=None, processor=None, model_name=None):
    feature_root, name, processor, model_name = argv

    feature = []
    with feature_root.begin(write=False) as txn:
        byteflow = txn.get(self.keys[name.encode('ascii')])
        feature.append(np.frombuffer(byteflow, dtype=np.float32))

    # feature -> (seqlen, featdim)
    single_feature = np.array(feature).squeeze()
    if len(single_feature) == 0:
        print ('feature has errors!!')
    elif len(single_feature.shape) == 1:
        single_feature = single_feature[np.newaxis, :]
    return single_feature

# model_name：表示用的哪个预训练模型
# read multiple data [different datasets need different processors]
def func_read_multiprocess(feature_root, names, processor=None, read_type='feat', model_name=None):
    ## names => features
    params = []
    for name in names:
        params.append((feature_root, name, processor, model_name))

    # ------ debug ------
    # func_read_one_feat(params[0])
    # func_read_one_e2e_video(params[0])
    # func_read_one_e2e_audio(params[0])

    features = []
    with multiprocessing.Pool(processes=12) as pool:
        if read_type == 'feat':
            features = list(tqdm.tqdm(pool.imap(func_read_one_feat, params), total=len(params)))

    ## save (names, features)
    feature_shape = np.array(features[0]).shape
    feature_name = os.path.basename(feature_root)
    print (f'Input feature {feature_name} ===> dim is {feature_shape}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    return features, feature_shape[-1]


def func_read_multiprocess_lmdb(feature_root, names, processor=None, read_type='feat', model_name=None):
    ## names => features
    params = []
    for name in names:
        params.append((feature_root, name, processor, model_name))

    # ------ debug ------
    # func_read_one_feat(params[0])
    # func_read_one_e2e_video(params[0])
    # func_read_one_e2e_audio(params[0])

    features = []
    with multiprocessing.Pool(processes=12) as pool:
        if read_type == 'feat':
            features = list(tqdm.tqdm(pool.imap(func_read_one_feat_lmdb, params), total=len(params)))

    ## save (names, features)
    feature_shape = np.array(features[0]).shape
    feature_name = os.path.basename(feature_root)
    print (f'Input feature {feature_name} ===> dim is {feature_shape}')
    assert len(names) == len(features), f'Error: len(names) != len(features)'
    return features, feature_shape[-1]


############################################################
# (seqlen, featdim) -> (dst_len, featdim)
def func_mapping_feature(feature, dst_len):
    featlen, featdim = feature.shape
    if featlen == dst_len:
        return feature
    elif featlen < dst_len:
        pad_feature = np.zeros((dst_len-featlen, featdim))
        feature = np.concatenate((feature, pad_feature), axis=0)
    else:
        if featlen // dst_len == featlen / dst_len:
            pad_len = 0
            pool_size = featlen // dst_len
        else:
            pad_len = dst_len - featlen % dst_len
            pool_size = featlen // dst_len + 1
        pad_feature = np.zeros((pad_len, featdim))
        feature = np.concatenate([pad_feature, feature]).reshape(dst_len, pool_size, featdim) # 相邻时刻特征取平均
        feature = np.mean(feature, axis=1)
    return feature

def func_mapping_feature_tensor(feature, dst_len, pad_place='right'):
    if len(feature.shape)>=2:
        featlen, featdim = feature.shape
    else:
        featlen = feature.shape[0]
    if featlen == dst_len:
        return feature
    elif featlen < dst_len:
        if pad_place=='right':
            pad_width = (0, dst_len - featlen)
        else:
            pad_width = (dst_len - featlen, 0)
        return torch.nn.functional.pad(feature, (0, 0, *pad_width), value=0)
    else:
        if featlen // dst_len == featlen / dst_len:
            pad_len = 0
            pool_size = featlen // dst_len
        else:
            pad_len = dst_len - featlen % dst_len
            pool_size = featlen // dst_len + 1
        pad_feature = torch.zeros((pad_len, featdim))
        feature = torch.cat([pad_feature, feature]).reshape(dst_len, pool_size, featdim) # 相邻时刻特征取平均
        feature = torch.mean(feature, axis=1)
        return feature

def func_mapping_feature_tensor_text_ids(feature, dst_len, pad_place='right'):

    featlen = feature.shape[0]
    if featlen == dst_len:
        return feature
    elif featlen < dst_len:
        if pad_place=='right':
            pad_width = (0, dst_len - featlen)
        else:
            pad_width = (dst_len - featlen, 0)
        return torch.nn.functional.pad(feature, pad_width, value=0)


# sample-level
def align_to_utt(audios, texts, videos):
    for ii in range(len(audios)):
        audios[ii] = np.mean(audios[ii], axis=0)
        texts[ii]  = np.mean(texts[ii],  axis=0)
        videos[ii] = np.mean(videos[ii], axis=0)
    return audios, texts, videos

# sample-level: 每个模态的特征长度压缩到原来的scale倍
def feature_scale_compress(audios, texts, videos, scale_factor=1):
    for ii in range(len(audios)):
        audios[ii] = func_mapping_feature(audios[ii], math.ceil(len(audios[ii]) / scale_factor))
        texts[ii]  = func_mapping_feature(texts[ii],  math.ceil(len(texts[ii])  / scale_factor))
        videos[ii] = func_mapping_feature(videos[ii], math.ceil(len(videos[ii]) / scale_factor))
    return audios, texts, videos

# sample-level: 三种模态压缩到文本长度
def align_to_text(audios, texts, videos):
    for ii in range(len(audios)):
        dst_len = len(texts[ii])
        audios[ii] = func_mapping_feature(audios[ii], dst_len)
        texts[ii]  = func_mapping_feature(texts[ii],  dst_len)
        videos[ii] = func_mapping_feature(videos[ii], dst_len)
    return audios, texts, videos

# batch-level: generate batch
def pad_to_maxlen_pre_modality(audios, texts, videos):
    audio_maxlen = max([len(feature) for feature in audios])
    text_maxlen  = max([len(feature) for feature in texts ])
    video_maxlen = max([len(feature) for feature in videos])
    for ii in range(len(audios)):
        audios[ii] = func_mapping_feature(audios[ii], audio_maxlen)
        texts[ii]  = func_mapping_feature(texts[ii],  text_maxlen)
        videos[ii] = func_mapping_feature(videos[ii], video_maxlen)
    return audios, texts, videos

def pad_to_maxlen_pre_modality_tensor(audios, texts, videos):
    audio_maxlen = max([len(feature) for feature in audios])
    text_maxlen  = max([len(feature) for feature in texts ])
    video_maxlen = max([len(feature) for feature in videos])
    for ii in range(len(audios)):
        audios[ii] = func_mapping_feature_tensor(audios[ii], audio_maxlen)
        texts[ii]  = func_mapping_feature_tensor(texts[ii],  text_maxlen)
        videos[ii] = func_mapping_feature_tensor(videos[ii], video_maxlen)
    return audios, texts, videos

def pad_to_maxlen_pre_modality_tensor_4(audios, texts, videos, feat4s):
    pad_lens = [[],[],[],[]]

    lens = [len(feature) for feature in audios]
    audio_maxlen = max(lens)
    pad_lens[0] = [audio_maxlen - x for x in lens]

    lens = [len(feature) for feature in texts]
    text_maxlen  = max(lens)
    pad_lens[1] = [text_maxlen - x for x in lens]

    lens = [len(feature) for feature in videos]
    video_maxlen = max(lens)
    pad_lens[2] = [video_maxlen - x for x in lens]

    lens = [len(feature) for feature in feat4s]
    feat4_maxlen = max(lens)
    pad_lens[3] = [feat4_maxlen - x for x in lens]


    for ii in range(len(audios)):
        audios[ii] = func_mapping_feature_tensor(audios[ii], audio_maxlen)
        texts[ii]  = func_mapping_feature_tensor(texts[ii],  text_maxlen)
        videos[ii] = func_mapping_feature_tensor(videos[ii], video_maxlen)
        feat4s[ii] = func_mapping_feature_tensor(feat4s[ii], feat4_maxlen)
    return audios, texts, videos, feat4s, pad_lens

# 三个特征的补长并且返回attention mask
def pad_to_maxlen_pre_modality_tensor_ReAMask(audios, texts, videos):
    audio_maxlen = max([len(feature) for feature in audios])
    text_maxlen  = max([len(feature) for feature in texts ])
    video_maxlen = max([len(feature) for feature in videos])
    audio_pad_masks = []
    text_pad_masks = []
    video_pad_masks = []
    
    for ii in range(len(audios)):
        audios[ii] = func_mapping_feature_tensor(audios[ii], audio_maxlen, 'left')
        mask = torch.ones(audio_maxlen)
        mask[:len(audios[ii])] = 0
        audio_pad_masks.append(mask)

        texts[ii]  = func_mapping_feature_tensor(texts[ii],  text_maxlen, 'left')
        mask = torch.ones(text_maxlen)
        mask[:len(texts[ii])] = 0
        text_pad_masks.append(mask)

        videos[ii] = func_mapping_feature_tensor(videos[ii], video_maxlen, 'left')
        mask = torch.ones(video_maxlen)
        mask[:len(videos[ii])] = 0
        video_pad_masks.append(mask)

    audio_pad_masks = torch.stack(audio_pad_masks)
    text_pad_masks = torch.stack(text_pad_masks)
    video_pad_masks = torch.stack(video_pad_masks)

    pad_masks = [audio_pad_masks, text_pad_masks, video_pad_masks]

    return audios, texts, videos, pad_masks

def pad_to_maxlen_llm_ids(text_ids):
    text_ids_maxlen = max([len(feature) for feature in text_ids])

    text_ids_pad_masks = []
    
    for ii in range(len(text_ids)):
        text_ids[ii] = func_mapping_feature_tensor_text_ids(text_ids[ii], text_ids_maxlen)
        mask = torch.ones(text_ids_maxlen)
        mask[len(text_ids[ii]):] = 0
        text_ids_pad_masks.append(mask)

    text_ids = torch.stack(text_ids)
    text_ids_pad_masks = torch.stack(text_ids_pad_masks)

    return text_ids, text_ids_pad_masks


# from USTC HaotianWang
class Collate_fn():
    def __init__(self, padding_value=0):
        self.padding_value = padding_value

    def __call__(self, batch):
        # print('*'*60)
        lengths = []
        feature_dim = []
        length_rule = 1024
        for i in range(3):
            length = []
            for tensor in batch:
                if(len(tensor[i].shape) == 2):
                    length.append(tensor[i].shape[0])
                else:
                    length.append(1)
            lengths.append(length)
            if(len(batch[0][i].shape) == 2):
                feature_dim.append(batch[0][i].shape[1])
            else:
                feature_dim.append(batch[0][i].shape[0])
               
        
        max_length = [max(length) for length in lengths]
        for i, item in enumerate(max_length):
            max_length[i] = length_rule
                
        collated_batch = []
        batch_size = len(batch)
        audio_collated_batch = torch.zeros(batch_size, max_length[0], feature_dim[0])
        text_collated_batch = torch.zeros(batch_size, max_length[1], feature_dim[1])
        video_collated_batch = torch.zeros(batch_size, max_length[2], feature_dim[2])
        # emo_collated_batch = torch.zeros(batch_size)
        val_collated_batch = torch.zeros(batch_size)
        name_collated_batch = []
        
        for i, tuple_pre in enumerate(batch):
            audio_length = tuple_pre[0].shape[0]
            audio_collated_tensor = torch.zeros(max_length[0], feature_dim[0])
            if audio_length <= length_rule:
                audio_collated_tensor[0:audio_length, ] = tuple_pre[0]
            else:
                audio_collated_tensor[0:length_rule, ] = tuple_pre[0][0:length_rule]
            audio_collated_batch[i,:,:] = audio_collated_tensor

            text_length = tuple_pre[1].shape[0]
            text_collated_tensor = torch.zeros(max_length[1], feature_dim[1])
            if text_length <= length_rule:
                text_collated_tensor[0:text_length, ] = tuple_pre[1]
            else:
                text_collated_tensor[0:text_length, ] = tuple_pre[1][0:length_rule]
            text_collated_batch[i,:,:] = text_collated_tensor

            video_length = tuple_pre[2].shape[0]
            video_collated_tensor = torch.zeros(max_length[2], feature_dim[2])
            if video_length <= length_rule:
                video_collated_tensor[0:video_length, ] = tuple_pre[2]
            else:
                video_collated_tensor[0:video_length, ] = tuple_pre[2][0:length_rule]
            video_collated_batch[i,:,:] = video_collated_tensor

            # emo_collated_batch[i] = tuple_pre[3]
            val_collated_batch[i] = tuple_pre[3]
            name_collated_batch.append(tuple_pre[4])
            
        collated_batch = (audio_collated_batch, text_collated_batch, video_collated_batch, val_collated_batch, name_collated_batch)
        return collated_batch

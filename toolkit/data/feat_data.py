import torch
import numpy as np
from torch.utils.data import Dataset
from toolkit.utils.read_data import *
import time

# MER2023的
class Data_Feat(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join(feat_root, args.audio_feature)
        text_root  = os.path.join(feat_root, args.text_feature )
        video_root = os.path.join(feat_root, args.video_feature)
        print (f'audio feature root: {audio_root}')

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # read datas (reduce __getitem__ durations)
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        # audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        # step2: align to batch
        # if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
        #     audios, texts, videos = align_to_utt(audios, texts, videos)
        # elif self.feat_type == 'frm_align':
        #     audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        # elif self.feat_type == 'frm_unalign':
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos = audios, texts, videos

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        instance = dict(
            audio = torch.FloatTensor(self.audios[index]),
            text  = torch.FloatTensor(self.texts[index]),
            video = torch.FloatTensor(self.videos[index]),
            emo   = self.labels[index]['emo'],
            val   = self.labels[index]['val'],
            name  = self.names[index],
        )
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
        audios, texts, videos = pad_to_maxlen_pre_modality_tensor(audios, texts, videos)

        batch = dict(
            audios = torch.stack(audios),
            texts  = torch.stack(texts),
            videos = torch.stack(videos),
        )
        
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim
    

# MOSEI加载emo和val，加载三个特征
class Data_Feat_MOSEI_EmoVal(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join(feat_root, args.audio_feature)
        text_root  = os.path.join(feat_root, args.text_feature )
        video_root = os.path.join(feat_root, args.video_feature)
        print (f'audio feature root: {audio_root}')

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # read datas (reduce __getitem__ durations)
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        
        # step2: align to batch
        # if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
        #     audios, texts, videos = align_to_utt(audios, texts, videos)
        # elif self.feat_type == 'frm_align':
        #     audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        # elif self.feat_type == 'frm_unalign':
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos = audios, texts, videos

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        instance = {
            'audio' : torch.FloatTensor(self.audios[index]),
            'text'  : torch.FloatTensor(self.texts[index]),
            'video' : torch.FloatTensor(self.videos[index]),
            'emo'   : self.labels[index]['emo'],
            'val'   : self.labels[index]['val'],
            'name'  : self.names[index],
        }
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
        
        audios, texts, videos = pad_to_maxlen_pre_modality_tensor(audios, texts, videos)


        batch = {
            'audios' : torch.stack(audios),
            'texts'  : torch.stack(texts),
            'videos' : torch.stack(videos),
        }
        emos  = torch.FloatTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim

## MOSEI加载emo和val标签，并且加载四个特征的
class Data_Feat_MOSEI_EmoVal_4F(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join(feat_root, args.audio_feature)
        text_root  = os.path.join(feat_root, args.text_feature )
        video_root = os.path.join(feat_root, args.video_feature)
        feat4_root = os.path.join(feat_root, args.feat4_feature)
        print (f'audio feature root: {audio_root}')

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # read datas (reduce __getitem__ durations)
        
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')
        feat4s, self.f4dim = func_read_multiprocess(feat4_root, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        # audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        
        # step2: align to batch
        # if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
        #     audios, texts, videos = align_to_utt(audios, texts, videos)
        # elif self.feat_type == 'frm_align':
        #     audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        # elif self.feat_type == 'frm_unalign':
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos, self.feat4s = audios, texts, videos, feat4s

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        instance = {
            'audio' : torch.FloatTensor(self.audios[index]),
            'text'  : torch.FloatTensor(self.texts[index]),
            'video' : torch.FloatTensor(self.videos[index]),
            'feat4' : torch.FloatTensor(self.feat4s[index]),
            'emo'   : self.labels[index]['emo'],
            'val'   : self.labels[index]['val'],
            'name'  : self.names[index],
        }
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
        feat4s = [instance['feat4'] for instance in instances]
        
        
        audios, texts, videos, feat4s, pads = pad_to_maxlen_pre_modality_tensor_4(audios, texts, videos, feat4s)


        batch = {
            'audios' : torch.stack(audios),
            'texts'  : torch.stack(texts),
            'videos' : torch.stack(videos),
            'feat4s' : torch.stack(feat4s),
        }
        
        emos  = torch.FloatTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, pads, emos, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}; feat4 dimension: {self.f4dim}')
        return self.adim, self.tdim, self.vdim, self.f4dim


## MOSEI加载emo和val标签，加载三个特征，一个文本，为LLM vicuna服务

class Data_Feat_Vicuna_MOSEI_EmoVal_4F(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join(feat_root, args.audio_feature)
        text_root  = os.path.join(feat_root, args.text_feature )
        video_root = os.path.join(feat_root, args.video_feature)
        feat4_root = os.path.join(feat_root, args.feat4_feature)
        # print (f'audio feature root: {audio_root}')

        

        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        self.audios, self.texts, self.videos = audios, texts, videos
        
        self.text_dict = {}
        import csv
        from transformers import AutoTokenizer
        with open(feat4_root, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = row['name']
                english = row['english']
                self.text_dict[name] = english
        self.text_tokenizer = AutoTokenizer.from_pretrained('/disk6/yzwen/SpeakerInvariantMER/tools/transformers/vicuna-7b-v1.5', use_fast=False, trust_remote_code=True)

        

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # read datas (reduce __getitem__ durations)
        
        self.adim = 1024
        self.tdim = 4096
        self.vdim = 1024
        self.f4dim = 4096
 
 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):

        name = self.names[index]

        text_raw = self.text_dict[name]

        instance = {
            'audio' : torch.FloatTensor(self.audios[index]),
            'text'  : torch.FloatTensor(self.texts[index]),
            'video' : torch.FloatTensor(self.videos[index]),
            'text_raw' : text_raw,
            'emo'   : self.labels[index]['emo'],
            'val'   : self.labels[index]['val'],
            'name'  : name,
        }
        return instance


    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
        text_ids = [instance['text_raw'] for instance in instances]
        
        encodings = self.text_tokenizer(text_ids, return_tensors="pt", padding=True)
        text_ids = encodings["input_ids"]
        attention_mask_text = encodings["attention_mask"]
        audios, texts, videos, attention_mask = pad_to_maxlen_pre_modality_tensor_ReAMask(audios, texts, videos)
        attention_mask.append(attention_mask_text)

        batch = {
            'audios' : torch.stack(audios),
            'texts'  : torch.stack(texts),
            'videos' : torch.stack(videos),
            'text_inputs' : text_ids,
            'attention_mask' : attention_mask,
        }
        
        emos  = torch.FloatTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, emos, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}; feat4 dimension: {self.f4dim}')
        return self.adim, self.tdim, self.vdim, self.f4dim



# Mosei原始的
class Data_Feat_MOSEI(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join(feat_root, args.audio_feature)
        text_root  = os.path.join(feat_root, args.text_feature )
        video_root = os.path.join(feat_root, args.video_feature)
        print (f'audio feature root: {audio_root}')

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        # read datas (reduce __getitem__ durations)
        audios, self.adim = func_read_multiprocess(audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(video_root, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        
        # step2: align to batch
        # if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
        #     audios, texts, videos = align_to_utt(audios, texts, videos)
        # elif self.feat_type == 'frm_align':
        #     audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        # elif self.feat_type == 'frm_unalign':
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        self.audios, self.texts, self.videos = audios, texts, videos

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        instance = {
            'audio' : torch.FloatTensor(self.audios[index]),
            'text'  : torch.FloatTensor(self.texts[index]),
            'video' : torch.FloatTensor(self.videos[index]),
            'val'   : self.labels[index]['val'],
            'name'  : self.names[index],
        }
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
        
        audios, texts, videos = pad_to_maxlen_pre_modality_tensor(audios, texts, videos)


        batch = {
            'audios' : torch.stack(audios),
            'texts'  : torch.stack(texts),
            'videos' : torch.stack(videos),
        }
        
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim


# Mosei的，使用了LMDB懒加载
class Data_Feat_MOSEI_LMDB(Dataset):
    def __init__(self, args, names, labels):

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = config.PATH_TO_FEATURES[args.dataset]
        audio_root = os.path.join(feat_root, args.audio_feature)
        text_root  = os.path.join(feat_root, args.text_feature )
        video_root = os.path.join(feat_root, args.video_feature)
        print (f'audio feature root: {audio_root}')
        
        self.env_audio = lmdb.open(audio_root+'.lmdb', subdir=os.path.isdir(audio_root+'.lmdb'),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env_text = lmdb.open(text_root+'.lmdb', subdir=os.path.isdir(text_root+'.lmdb'),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env_video = lmdb.open(video_root+'.lmdb', subdir=os.path.isdir(video_root+'.lmdb'),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        # --------------- temporal test ---------------
        # for name in names: assert os.path.exists(os.path.join(audio_root, name+'.npy'))

        # analyze params
        self.feat_type = args.feat_type
        self.feat_scale = args.feat_scale # 特征预压缩
        assert self.feat_scale >= 1
        assert self.feat_type in ['utt', 'frm_align', 'frm_unalign']

        self.adim = 1024
        self.tdim = 768
        self.vdim = 1024
        # read datas (reduce __getitem__ durations)
        # audios, self.adim = func_read_multiprocess_lmdb(self.env_audio, self.names, read_type='feat')
        # texts,  self.tdim = func_read_multiprocess_lmdb(self.env_text,  self.names, read_type='feat')
        # videos, self.vdim = func_read_multiprocess_lmdb(self.env_video, self.names, read_type='feat')

        ## read batch (reduce collater durations)
        # step1: pre-compress features
        # audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        
        # step2: align to batch
        # if self.feat_type == 'utt': # -> 每个样本每个模态的特征压缩到句子级别
        #     audios, texts, videos = align_to_utt(audios, texts, videos)
        # elif self.feat_type == 'frm_align':
        #     audios, texts, videos = align_to_text(audios, texts, videos) # 模态级别对齐
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        # elif self.feat_type == 'frm_unalign':
        #     audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos) # 样本级别对齐
        # self.audios, self.texts, self.videos = audios, texts, videos

 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):
        name = self.names[index]
        with self.env_audio.begin(write=False) as txn:
            byteflow = txn.get(name.encode('ascii'))
            audio_feat = np.frombuffer(byteflow, dtype=np.float32).reshape(-1,  self.adim)
        with self.env_text.begin(write=False) as txn:
            byteflow = txn.get(name.encode('ascii'))
            text_feat = np.frombuffer(byteflow, dtype=np.float32).reshape(-1,  self.tdim)
        with self.env_video.begin(write=False) as txn:
            byteflow = txn.get(name.encode('ascii'))
            vifeo_feat = np.frombuffer(byteflow, dtype=np.float32).reshape(-1,  self.vdim)

        instance = {
            'audio' : torch.FloatTensor(audio_feat.copy()),
            'text'  : torch.FloatTensor(text_feat.copy()),
            'video' : torch.FloatTensor(vifeo_feat.copy()),
            'val'   : self.labels[index]['val'],
            'name'  : name,
        }
        return instance
    

    def collater(self, instances):
        audios = [instance['audio'] for instance in instances]
        texts  = [instance['text']  for instance in instances]
        videos = [instance['video'] for instance in instances]
        
        audios, texts, videos = pad_to_maxlen_pre_modality_tensor(audios, texts, videos)


        batch = {
            'audios' : torch.stack(audios),
            'texts'  : torch.stack(texts),
            'videos' : torch.stack(videos),
        }
        
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        names = [instance['name'] for instance in instances]

        return batch, vals, names
    

    def get_featdim(self):
        print (f'audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}')
        return self.adim, self.tdim, self.vdim
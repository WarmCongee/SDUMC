import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as autocast
from transformers import AutoModel, BertTokenizer, AutoTokenizer # version: 4.5.1, pip install transformers
from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM

# local folder
import sys
import os
sys.path.append('../../')
import config
from toolkit.utils.read_data import *

VICUNA_7B_V15  = 'vicuna-7b-v1.5' 


################################################################
# 自动删除无意义token对应的特征
def find_start_end_pos(tokenizer):
    sentence = '今天天气真好' # 句子中没有空格
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
    start, end = None, None

    # find start, must in range [0, 1, 2]
    for start in range(0, 3, 1):
        # 因为decode有时会出现空格，因此我们显示的时候把这部分信息去掉看看
        outputs = tokenizer.decode(input_ids[start:]).replace(' ', '')
        if outputs == sentence:
            print (f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence):
            break
   
    # find end, must in range [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end]).replace(' ', '')
        if outputs == sentence:
            break
    
    assert tokenizer.decode(input_ids[start:end]).replace(' ', '') == sentence
    print (f'start: {start};  end: {end}')
    return start, end


# 找到 batch_pos and feature_dim
def find_batchpos_embdim(tokenizer, model, gpu):
    sentence = '今天天气真好'
    inputs = tokenizer(sentence, return_tensors='pt')
    if gpu != -1: inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
        outputs = torch.stack(outputs)[[-1]].sum(dim=0) # sum => [batch, T, D=768]
        outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1]
        feature_dim = outputs.shape[2]
    print (f'batch_pos:{batch_pos}, feature_dim:{feature_dim}')
    return batch_pos, feature_dim


# def func_read_multiprocess(feature_root, names, processor=None, read_type='feat', model_name=None):
#     ## names => features
#     params = []
#     for name in names:
#         params.append((feature_root, name, processor, model_name))

#     # ------ debug ------
#     # func_read_one_feat(params[0])
#     # func_read_one_e2e_video(params[0])
#     # func_read_one_e2e_audio(params[0])

#     features = []
#     with multiprocessing.Pool(processes=12) as pool:
#         if read_type == 'feat':
#             features = list(tqdm.tqdm(pool.imap(func_read_one_feat, params), total=len(params)))

#     ## save (names, features)
#     feature_shape = np.array(features[0]).shape
#     feature_name = os.path.basename(feature_root)
#     print (f'Input feature {feature_name} ===> dim is {feature_shape}')
#     assert len(names) == len(features), f'Error: len(names) != len(features)'
#     return features, feature_shape[-1]

class Data_WavLM_Text(Dataset):
    def __init__(self):
        debug = False
        names, labels = [], []
        label_path = '/disk6/yzwen/SpeakerInvariantMER/dataset/datasets_label/cmumosei-process/label_official.npz'
        corpus_train = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        corpus_val = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
        corpus_test = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        corpus = {**corpus_train, **corpus_val, **corpus_test}
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]

        # analyze path
        self.names = names
        self.labels = labels
        feat_root  = '/disk6/yzwen/SpeakerInvariantMER/dataset/features_mosei'
        wavlm_root  = os.path.join(feat_root, 'wavlm-large-FRA_-1')
        feat4_root = os.path.join(feat_root, 'mosei_text_mid_new.csv')
        # print (f'audio feature root: {audio_root}')

    
        wavlm_feats,  self.tdim = func_read_multiprocess(wavlm_root,  self.names, read_type='feat')

        self.wavlm_feats = wavlm_feats
        
        self.text_dict = {}
        import csv
        with open(feat4_root, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file) # , delimiter='\t'
            for row in reader:
                name = row['name']
                english = row['english']
                self.text_dict[name] = english

        self.wavlm_dim = 1024
        self.llm_dim = 4096
 
 
    def __len__(self):
        return len(self.names)


    def __getitem__(self, index):

        name = self.names[index]
        text_raw = self.text_dict[name]

        instance = {
            'wavlm_feats'  : torch.FloatTensor(self.wavlm_feats[index]),
            'text_raw' : text_raw,
            'emo'   : self.labels[index]['emo'],
            'val'   : self.labels[index]['val'],
            'name'  : name,
        }
        return instance

    def get_featdim(self):
        return self.wavlm_dim, self.llm_dim


class EncoderProjectorConcat(nn.Module):

    def __init__(self, encoder_projector_ds_rate, encoder_dim, llm_dim):
        super().__init__()
        self.k = encoder_projector_ds_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class WavLM2Vicuna(nn.Module):
    def __init__(self, output_dim1=1, output_dim2=1, layers='256,128', dropout=0.3):
        super(WavLM2Vicuna, self).__init__()

        self.encoder_projector = EncoderProjectorConcat(5, 1024, 4096)
        porjector_temp = torch.load('/disk6/yzwen/SpeakerInvariantMER/tools/transformers/WalmL2VicunaV1.5_model.pt')
        new_dict = {key[len('encoder_projector.'):]: value for key, value in porjector_temp.items()}
    
        print(porjector_temp.keys())
        self.encoder_projector.load_state_dict(new_dict)
        for param in self.encoder_projector.parameters():
            param.requires_grad = False

        self.layer_ids = [-3]
        self.vicuna_model = AutoModelForCausalLM.from_pretrained('/disk6/yzwen/SpeakerInvariantMER/tools/transformers/vicuna-7b-v1.5', low_cpu_mem_usage=True)
        self.vicuna_model = self.vicuna_model.half()
        for param in self.vicuna_model.parameters():
            param.requires_grad = False


    def forward(self, batch):
        text_feat, prompt_text_ids, groundtruth_text_ids = batch[0], batch[1], batch[2]

        # input_embeds_all = self.encoder_projector(text_feat)
        # wav_len = input_embeds_all.shape[1]
        # if len(groundtruth_text_ids) > 0:
        #     if hasattr(self.vicuna_model.model, "embed_tokens"):
        #         inputs_embeds_0 = self.vicuna_model.model.embed_tokens(prompt_text_ids)
        #         inputs_embeds_1 = self.vicuna_model.model.embed_tokens(groundtruth_text_ids)
        #     elif hasattr(self.vicuna_model.model.model, "embed_tokens"):
        #         inputs_embeds_0 = self.vicuna_model.model.embed_tokens(prompt_text_ids)
        #         inputs_embeds_1 = self.vicuna_model.model.embed_tokens(groundtruth_text_ids)
        #     else:
        #         inputs_embeds_0 = self.vicuna_model.model.model.model.embed_tokens(prompt_text_ids)
        #         inputs_embeds_1 = self.vicuna_model.model.model.model.embed_tokens(groundtruth_text_ids)
        #     pre_len = input_embeds_all.shape[1] + inputs_embeds_0.shape[1]
        #     input_embeds_all = torch.cat([input_embeds_all, inputs_embeds_0, inputs_embeds_1], dim=1)
        # elif len(prompt_text_ids) > 0:
        #     if hasattr(self.vicuna_model.model, "embed_tokens"):
        #         inputs_embeds_0 = self.vicuna_model.model.embed_tokens(prompt_text_ids)
        #     elif hasattr(self.vicuna_model.model.model, "embed_tokens"):
        #         inputs_embeds_0 = self.vicuna_model.model.embed_tokens(prompt_text_ids)
        #     else:
        #         inputs_embeds_0 = self.vicuna_model.model.model.model.embed_tokens(prompt_text_ids)
        #     pre_len = input_embeds_all.shape[1] + inputs_embeds_0.shape[1]
        #     input_embeds_all = torch.cat([input_embeds_all, inputs_embeds_0], dim=1)

        if hasattr(self.vicuna_model.model, "embed_tokens"):
            inputs_embeds = self.vicuna_model.model.embed_tokens(groundtruth_text_ids)
        elif hasattr(self.vicuna_model.model.model, "embed_tokens"):
            inputs_embeds = self.vicuna_model.model.embed_tokens(groundtruth_text_ids)
        else:
            inputs_embeds = self.vicuna_model.model.model.model.embed_tokens(groundtruth_text_ids)
        
        input_embeds_all = inputs_embeds

        # len_new = attention_mask_4asr.shape[1] // 5 * 5
        # attention_mask_4asr = attention_mask_4asr[:, :len_new]
        # attention_mask_4asr = attention_mask_4asr.view(attention_mask_4asr.shape[0], len_new // 5, 5)
        # attention_mask_reduced = attention_mask_4asr.mean(dim=2)
        # attention_mask_reduced = (attention_mask_reduced > 0.5).float()
        
        with autocast():
            llms_hidden_state = self.vicuna_model(inputs_embeds=input_embeds_all, output_hidden_states=True).hidden_states
            llms_hidden_state = torch.stack(llms_hidden_state)[self.layer_ids].sum(dim=0)
            print(llms_hidden_state.shape)
            # llms_hidden_state = torch.stack(llms_hidden_state)[self.layer_ids].sum(dim=0)
            # llms_hidden_state = llms_hidden_state[:,pre_len:,:]

            # 输出asr文本看看
            # print(input_embeds_all.shape)
            # text_outputs = self.vicuna_model.generate(
            #     inputs_embeds=input_embeds_all,
            #     # max_length=kwargs.get("max_length", 200),
            #     max_new_tokens=200,
            #     num_beams=4,
            #     do_sample=False,
            #     min_length=1,
            #     top_p=1.0,
            #     repetition_penalty=1.0,
            #     length_penalty=1.0,
            #     temperature=1.0,
            #     output_hidden_states=True, 
            #     return_dict_in_generate=True
            # )
            
            # llms_hidden_state = text_outputs.hidden_states
            # llms_hidden_state = llms_hidden_state[1:]
            # llms_hidden_state = [inner_tuple[-1] for inner_tuple in llms_hidden_state]
            # llms_hidden_state = torch.stack(llms_hidden_state)[:,0,:]
            # llms_hidden_state = llms_hidden_state.unsqueeze(0)
        #feat4rnc = self.orgin_linear_change(cross_fused_feat)
        text_outputs = []
        return llms_hidden_state, text_outputs


# main process
def extract_embedding(model_name, trans_dir, save_dir, feature_level, gpu=-1, punc_case=None, language='chinese', model_dir=None):

    print('='*30 + f' Extracting "{model_name}" ' + '='*30)
    start_time = time.time()

    # save last four layers
    layer_ids = [-3]
    save_dir = os.path.join(save_dir, f'{model_name}-{feature_level[:3]}-wavlm2vicuna-half-gt[-3]')


    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model and tokenizer: offline mode (load cached files) # 函数都一样，但是有些位置的参数就不好压缩
    # print('Loading pre-trained tokenizer and model...')
    # if model_dir is None:
    #     model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')

    # if model_name in [VICUNA_7B_V1.5]:
    #     tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained('/disk6/yzwen/SpeakerInvariantMER/tools/transformers/vicuna-7b-v1.5', use_fast=False, trust_remote_code=True)
    model = WavLM2Vicuna()
    
    # 有 gpu 并且是 LLM，才会增加 half process
    # if gpu != -1 and model_name in [LLAMA_7B, LLAMA_13B, LLAMA2_7B, LLAMA2_13B, VICUNA_7B, VICUNA_13B, ALPACE_13B, 
    #                                 OPT_13B, BLOOM_7B, CHATGLM2_6B, MOSS_7B, BAICHUAN_7B, FALCON_7B, BAICHUAN_13B, 
    #                                 STABLEML_7B, BAICHUAN2_7B, BAICHUAN2_13B, BAICHUAN2_13B_4bits]:
    #     model = model.half()

    # 有 gpu 才会放在cuda上
    # if gpu != -1:
    #     torch.cuda.set_device(gpu)
    #     model.cuda()


    if torch.cuda.device_count() > 1:
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map, get_balanced_memory
        device_map = infer_auto_device_map(model, max_memory=get_balanced_memory(model))

        model = dispatch_model(model, device_map)
        print('multi GPU predict => {}'.format(device_map))
    else:
        model = model.cuda()
        print("single GPU predict")

    model.eval()


    # print('Calculate embeddings...')
    # start, end = find_start_end_pos(tokenizer) # only preserve [start:end+1] tokens
    # batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model, gpu) # find batch pos

    dataset = Data_WavLM_Text()
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=4,
                            pin_memory=True,
                            # sampler=sampler,
                            prefetch_factor=8)

    for data in tqdm.tqdm(dataloader):
        wavlm_feat = data['wavlm_feats'].to('cuda')
        text = data['text_raw'][0]
        # prompt_template = "USER: Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. \n ASSISTANT:"
        # text = [prompt_template+text]
        name = data['name'][0]
        encodings = tokenizer(text, return_tensors="pt", padding=True)
        text_ids = encodings["input_ids"].to('cuda')
        attention_mask_text = encodings["attention_mask"]

        # prompt_encodings = tokenizer(prompt_template, return_tensors="pt", padding=True)
        # prompt_encodings_ids = prompt_encodings["input_ids"].to('cuda')
        prompt_encodings_ids = []
        csv_file = os.path.join(save_dir, f'{name}.npy')
        # print(csv_file)
        if os.path.exists(csv_file): #  or name=='dQ56b0bqmc8_0':# dQ56b0bqmc8_0 is too long and in the train set, meaningless
            continue
        with torch.no_grad():
            outputs, text_outputs = model([wavlm_feat, prompt_encodings_ids, text_ids])
            # text_outputs = tokenizer.batch_decode(text_outputs, add_special_tokens=False, skip_special_tokens=True)

            outputs = outputs.cpu().numpy()
        embeddings = outputs

        if feature_level == 'FRAME':
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((1, feature_dim))
            elif len(embeddings.shape) == 1:
                embeddings = embeddings[np.newaxis, :]
            print(embeddings.shape)
            np.save(csv_file, embeddings)
        else:
            embeddings = np.array(embeddings).squeeze()
            if len(embeddings) == 0:
                embeddings = np.zeros((feature_dim, ))
            elif len(embeddings.shape) == 2:
                embeddings = np.mean(embeddings, axis=0)
            np.save(csv_file, embeddings)

    # df = pd.read_csv(trans_dir, sep = '\t')
    # for idx, row in df.iterrows():
    #     name = row['name']
    #     # --------------------------------------------------
    #     if language == 'chinese':
    #         sentence = row['chinese'] # process on Chinese
    #     elif language == 'english':
    #         sentence = row['english']
    #     # --------------------------------------------------
    #     print(f'Processing {name} ({idx}/{len(df)})...')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--dataset', type=str, help='input dataset')
    parser.add_argument('--gpu', type=int, default='2', help='gpu id')
    parser.add_argument('--model_name', type=str, help='name of pretrained model')
    parser.add_argument('--feature_level', type=str, default='UTTERANCE', choices=['UTTERANCE', 'FRAME'], help='output types')
    # ------ 临时测试外部接受的 model_dir [for gu hao] ------
    parser.add_argument('--model_dir', type=str, default=None, help='used user-defined model_dir')
    args = parser.parse_args()

    # (trans_dir, save_dir)
    # if args.punc_case is None:
    #     trans_dir = config.PATH_TO_TRANSCRIPTIONS[args.dataset]
    # else:
    #     assert args.punc_case in ['case1', 'case2', 'case3']
    #     trans_dir = config.PATH_TO_TRANSCRIPTIONS[args.dataset][:-4] + f'-{args.punc_case}.csv'
    #     assert os.path.exists(trans_dir)
    trans_dir = '/disk6/yzwen/SpeakerInvariantMER/dataset/mosei_text_asr_whisper-base.en_vad.csv'
    save_dir = config.PATH_TO_FEATURES[args.dataset]

    extract_embedding(model_name=args.model_name, 
                      trans_dir=trans_dir, 
                      save_dir=save_dir,
                      feature_level=args.feature_level,
                      gpu=args.gpu,
                      model_dir=args.model_dir)
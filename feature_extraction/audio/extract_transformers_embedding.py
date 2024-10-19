import os
import time
import glob
import torch
import argparse
import numpy as np
import soundfile as sf

from transformers import Wav2Vec2FeatureExtractor # pip install transformers==4.16.2

# import config
import sys
sys.path.append('../../')
import config as config

# supported models
################## CHINESE ######################
HUBERT_BASE_CHINESE = 'chinese-hubert-base' # https://huggingface.co/TencentGameMate/chinese-hubert-base
HUBERT_LARGE_CHINESE = 'chinese-hubert-large' # https://huggingface.co/TencentGameMate/chinese-hubert-large
WAV2VEC2_BASE_CHINESE = 'chinese-wav2vec2-base' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-base
WAV2VEC2_LARGE_CHINESE = 'chinese-wav2vec2-large' # https://huggingface.co/TencentGameMate/chinese-wav2vec2-large
HUBERT_LARGE = 'hubert-large-ls960-ft' 
################## ENGLISH ######################
WAV2VEC2_BASE = 'wav2vec2-base-960h' # https://huggingface.co/facebook/wav2vec2-base-960h
WAV2VEC2_LARGE = 'wav2vec2-large-960h' # https://huggingface.co/facebook/wav2vec2-large-960h

WAVLM_LARGE = 'wavlm-large'

def extract(model_name, audio_files, save_dir, feature_level, layer_ids=None, gpu=None):

    start_time = time.time()

    # load model
    model_file = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{model_name}')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_file)
    if model_name.find('hubert') != -1 and model_name.find('chinese') != -1:
        from transformers import HubertModel
        model = HubertModel.from_pretrained(model_file)
    elif model_name.find('hubert') != -1:
        from transformers import AutoProcessor, HubertModel
        processor = AutoProcessor.from_pretrained(model_file)
        model = HubertModel.from_pretrained(model_file)
        print(model)
        print("hubert-large-ls960-ft")
        print("======================")
    elif model_name.find('wav2vec') != -1:
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(model_file)
    elif model_name.find('wavlm') != -1:
        from transformers import Wav2Vec2Processor, WavLMForCTC, WavLMModel
        from transformers import AutoProcessor, AutoModel
        print("wavlm-large")
        # processor = Wav2Vec2Processor.from_pretrained(model_file)
        model = WavLMModel.from_pretrained(model_file)
        print(model)
        print("wavlm-large")
        print("======================")

    if gpu != -2:
        device = torch.device(f'cuda:{gpu}')
        model.to(device)
    model.eval()

    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file).split('.')[0]
        vid = file_name
        print(f'Processing "{file_name}" ({idx}/{len(audio_files)})...')

        ## process for too short ones
        samples, sr = sf.read(audio_file)
        print(samples.shape)
        ## print(sr)
        if model_name.find('hubert') != -1 and model_name.find('chinese') == -1:
            input_values = processor(samples, sampling_rate=sr, return_tensors="pt", padding='longest').input_values
            print("processor")
        elif model_name.find('wavlm') != -1:
            input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt", padding='longest').input_values
            ## print("processor_wavlm")
        else: 
            input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt", padding='longest').input_values
        # Batch size 1
        # print(input_values.shape)
        # print(sr)
        if gpu != -1: input_values = input_values.to(device)

        with torch.no_grad():
            # model inference
            # print(model(input_values, output_hidden_states=True).hidden_states[-2].shape)
            hidden_states = model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
            ## print(torch.stack(hidden_states).shape)
            # print(torch.stack(hidden_states)[layer_ids].sum(dim=0).shape)
            feature = torch.stack(hidden_states)[layer_ids].sum(dim=0)  # [1, T, D]
            # print(feature.shape)
            assert feature.shape[0] == 1
            feature = feature[0].detach().squeeze().cpu().numpy() # [T, D]
            # print(feature.shape) 

        csv_file = os.path.join(save_dir, f'{vid}.npy')
        if feature_level == 'UTTERANCE':
            feature = np.array(feature).squeeze() # [T, D]
            # print(feature.shape)
            if len(feature.shape) != 1: # change [T, C] => [C,]
                feature = np.mean(feature, axis=0)
            np.save(csv_file, feature)
        else:
            ## print(feature.shape) 
            np.save(csv_file, feature)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--gpu', type=int, default=1, help='index of gpu')
    parser.add_argument('--model_name', type=str, default='opensmile', help='name of feature extractor')
    parser.add_argument('--feature_level', type=str, default='FRAME', help='name of feature level, FRAME or UTTERANCE')
    parser.add_argument('--overwrite', action='store_true', default=True, help='whether overwrite existed feature folder.')
    parser.add_argument('--dataset', type=str, default='BoxOfLies', help='input dataset')
    args = parser.parse_args()

    # analyze input
    layer_ids = [-5]
    # audio_dir = config.PATH_TO_RAW_AUDIO[args.dataset]
    # save_dir = config.PATH_TO_FEATURES[args.dataset]
    audio_dir = '/disk6/yzwen/SpeakerInvariantMER/dataset/MOSEI_audio'
    save_dir = '/disk6/yzwen/SpeakerInvariantMER/dataset/features_mosei'

    # => audio_files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    print(audio_files[0])
    print(f'Find total "{len(audio_files)}" audio files.')

    # => save_dir
    dir_name = args.model_name if len(layer_ids) == 1 else f'{args.model_name}-{len(layer_ids)}'
    dir_name = f'{dir_name}-{args.feature_level[:3]}_{layer_ids[0]}'
    save_dir = os.path.join(save_dir, dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif args.overwrite or len(os.listdir(save_dir)) == 0:
        print(f'==> Warning: overwrite save_dir "{save_dir}"!')
    else:
        raise Exception(f'==> Error: save_dir "{save_dir}" already exists, set overwrite=TRUE if needed!')
    
    # extract features
    extract(args.model_name, audio_files, save_dir, args.feature_level, layer_ids=layer_ids, gpu=args.gpu)
    
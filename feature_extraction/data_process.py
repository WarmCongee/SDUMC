import os
import sys
import glob
import tqdm
import shutil
import pandas as pd
import numpy as np
sys.path.append("..")
import config 

def normalize_dataset_format(data_root, save_root):
    ## input path
    train_label = os.path.join(data_root, 'extract_val.csv')

    ## output path
    save_label = os.path.join(save_root, 'label_22856.npz')
    if not os.path.exists(save_root): os.makedirs(save_root)

    ## generate label path
    train_corpus = {}
    df_label = pd.read_csv(train_label)
    for _, row in df_label.iterrows(): ## read for each row
        name = row['name']
        val  = row['valence']
        train_corpus[name] = {'val': val}

    np.savez_compressed(save_label,
                        train_corpus=train_corpus)
    
def divide_train_test_valid(data_root, save_root):
    train_label = os.path.join(data_root, 'label.csv')
    label_speaker = os.path.join(data_root, 'speaker.csv')

    ## output path
    save_csv = os.path.join(save_root, 'label_speaker_mode.csv')
    save_label_train = os.path.join(save_root, 'label_train.npz')
    save_label_test = os.path.join(save_root, 'label_test.npz')
    save_label_valid = os.path.join(save_root, 'label_valid.npz')

    if not os.path.exists(save_root): os.makedirs(save_root)

    
    name_list = []
    speaker_list = []
    valence_list = []
    
    df_label_speaker = pd.read_csv(label_speaker)
    train_corpus = {}
    for _, row in df_label_speaker.iterrows(): ## read for each row
        name = row['name']
        speaker  = row['speaker']
        valence = row['valence']

        name_list.append(name)
        speaker_list.append(speaker)
        valence_list.append(valence)
        
        # train_corpus[name] = {'val': val}

    ## generate label path
    name2mode = {}
    df_label = pd.read_csv(train_label)
    for _, row in df_label.iterrows(): ## read for each row
        video_id = row['video_id']
        clip_id  = row['clip_id']
        mode = row['mode']
        name2mode[video_id+'_'+str(clip_id)] = mode

    mode_list = []
    for item in name_list:
        mode_list.append(name2mode[item])

    columns = ['name', 'speaker', 'valence', 'mode']
    data = np.column_stack([name_list, speaker_list, valence_list, mode_list])
    df = pd.DataFrame(data=data, columns=columns)
    df[columns] = df[columns].astype(str)
    df.to_csv(save_csv, index=False)
    
    
def normalize_dataset_format_divide(data_root, save_root):
    ## input path
    train_label = os.path.join(data_root, 'label_speaker_mode.csv')

    ## output path
    save_label_train = os.path.join(save_root, 'label_train.npz')
    save_label_valid = os.path.join(save_root, 'label_valid.npz')
    save_label_test = os.path.join(save_root, 'label_test.npz')
    if not os.path.exists(save_root): os.makedirs(save_root)

    ## generate label path
    train_corpus = {}
    valid_corpus = {}
    test_corpus = {}
    df_label = pd.read_csv(train_label)
    for _, row in df_label.iterrows(): ## read for each row
        name = row['name']
        val  = row['valence']
        mode = row['mode']
        if mode=='train':
            train_corpus[name] = {'val': val}
        elif mode=='valid':
            valid_corpus[name] = {'val': val}
        elif mode=='test':
            test_corpus[name] = {'val': val}

    np.savez_compressed(save_label_train,
                        train_corpus=train_corpus)
    
    np.savez_compressed(save_label_valid,
                        train_corpus=valid_corpus)
    
    np.savez_compressed(save_label_test,
                        train_corpus=test_corpus)
    
# 
def normalize_dataset_format_newlabel(data_root, save_root):
    ## input path
    train_label = os.path.join(data_root, 'extract_val_emo_speaker_mode.csv')

    ## output path
    save_label_train = os.path.join(save_root, 'label_new_val_emo_22856.npz')
    if not os.path.exists(save_root): os.makedirs(save_root)

    ## generate label path
    train_corpus = {}
    valid_corpus = {}
    test_corpus = {}
    df_label = pd.read_csv(train_label)
    for _, row in df_label.iterrows(): ## read for each row
        name = row['name']
        val  = row['valence']
        mode = row['mode']
        if mode=='train':
            train_corpus[name] = {'val': val, 'happy': row['happy'], 'sad': row['sad'], 'anger': row['anger'], 'surprise': row['surprise'], 'disgust': row['disgust'], 'fear': row['fear']}
        elif mode=='valid':
            valid_corpus[name] = {'val': val, 'happy': row['happy'], 'sad': row['sad'], 'anger': row['anger'], 'surprise': row['surprise'], 'disgust': row['disgust'], 'fear': row['fear']}
        elif mode=='test':
            test_corpus[name] = {'val': val, 'happy': row['happy'], 'sad': row['sad'], 'anger': row['anger'], 'surprise': row['surprise'], 'disgust': row['disgust'], 'fear': row['fear']}

    np.savez_compressed(save_label_train,
                        train_corpus=train_corpus,
                        valid_corpus=valid_corpus,
                        test_corpus=test_corpus)


def split_and_copy_folders(src_dir, dest_base_dir, num_folders):
    # 获取目录下的所有第一级子文件夹
    subfolders = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]

    # 计算每个目标文件夹应包含的子文件夹数量
    folders_per_dest = len(subfolders) // num_folders
    remainder = len(subfolders) % num_folders

    # 创建目标文件夹
    for i in range(num_folders):
        dest_dir = os.path.join(dest_base_dir, f'folder_{i + 1}')
        os.makedirs(dest_dir, exist_ok=True)

        # 计算当前目标文件夹应包含的子文件夹数量
        if i==num_folders-1:
            current_folders = folders_per_dest + remainder
        else:
            current_folders = folders_per_dest

        # 复制子文件夹到目标文件夹
        for j in range(current_folders):
            src_subfolder = os.path.join(src_dir, subfolders[i * folders_per_dest + j])
            dest_subfolder = os.path.join(dest_dir, subfolders[i * folders_per_dest + j])
            shutil.copytree(src_subfolder, dest_subfolder)

# split audios from videos
def split_audio_from_video_16k(video_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for video_path in tqdm.tqdm(glob.glob(video_root+'/*')):
        videoname = os.path.basename(video_path)[:-4]
        audio_path = os.path.join(save_root, videoname + '.wav')
        if os.path.exists(audio_path): continue
        cmd = "%s -loglevel quiet -y -i %s -ar 16000 -ac 1 %s" %(config.PATH_TO_FFMPEG, video_path, audio_path)
        os.system(cmd)

if __name__=='__main__':
    # normalize_dataset_format_divide(config.DATA_DIR['CMU-MOSEI'], config.DATA_DIR['CMU-MOSEI'])
    # normalize_dataset_format_newlabel('/disk1/yzwen/SpeakerInvariantMER/dataset', '/disk1/yzwen/SpeakerInvariantMER/dataset')
    split_audio_from_video_16k('/disk6/yzwen/SpeakerInvariantMER/dataset/CMU-MOSI/video', '/disk6/yzwen/SpeakerInvariantMER/dataset/CMU-MOSI/audio')
# *_*coding:utf-8 *_*
import os
import sys
import socket


############ For LINUX ##############
DATA_DIR = {
	'MER2023': '/disk6/yzwen/SpeakerInvariantMER/dataset',
	'MER2023-EMU': '/disk3/htwang/MER2023-Baseline-master/dataset-emulation',
    'CHVAD': '/disk4/htwang/CHVAD/',
	'CMU-MOSEI': '/disk6/yzwen/SpeakerInvariantMER/dataset',
}
PATH_TO_RAW_AUDIO = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'MER2023/audio/test1'),
	'CHVAD': os.path.join(DATA_DIR['CHVAD'], 'audio-train'),
	'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'MOSEI_audio'),
}
PATH_TO_RAW_FACE = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features-train/openface_face_MER2023test2'),
	'MER2023-EMU': os.path.join(DATA_DIR['MER2023-EMU'], 'features-musan-total/openface_face_blur1'),
    'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'features-train/openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'mer2023_test3_text.csv'),
    'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'mosei_text_mid_new.csv'),
}
PATH_TO_FEATURES = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'features_mer2023'),
 	'CHVAD': os.path.join(DATA_DIR['CHVAD'], 'features-train'),
  	'eval': os.path.join(DATA_DIR['CHVAD'], 'features-eval'),
    'test1': os.path.join(DATA_DIR['MER2023'], 'features-test1'),
    'test2': os.path.join(DATA_DIR['MER2023'], 'features-test2'),
    'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'features_mosei'),
}

PATH_TO_LABEL = {
	'MER2023': os.path.join(DATA_DIR['MER2023'], 'datasets_label/mer2023-dataset-process/label-6way.npz'),
	# label_new_val_emo_22856.npz  datasets_label/mer2023-dataset-process/label-6way.npz
 	'CHVAD': os.path.join(DATA_DIR['CHVAD'], 'train_label.npz'),
    'eval': os.path.join(DATA_DIR['CHVAD'], 'eval_label.npz'),
 	'test1': os.path.join(DATA_DIR['MER2023'], 'label-test1.npz'),
	'test2': os.path.join(DATA_DIR['MER2023'], 'label-test2.npz'),
    'CMU-MOSEI': os.path.join(DATA_DIR['CMU-MOSEI'], 'datasets_label/cmumosei-process/label_official.npz'),
    'CMU-MOSEI_valid': os.path.join(DATA_DIR['CMU-MOSEI'], 'label_valid.npz'),
    'CMU-MOSEI_test': os.path.join(DATA_DIR['CMU-MOSEI'], 'label_test.npz'),
    'CMU-MOSEI_new': os.path.join(DATA_DIR['CMU-MOSEI'], 'label_new_val_emo_22856.npz'),
}

PATH_TO_TOOL = {
	'wenet': '/disk6/yzwen/SpeakerInvariantMER/tools/wenet/20220506_u2pp_conformer_libtorch/',
	'whisper': '/disk6/yzwen/SpeakerInvariantMER/tools/whisper/'
}

PATH_TO_PRETRAINED_MODELS = '/disk6/yzwen/SpeakerInvariantMER/tools'
PATH_TO_OPENSMILE = '/disk6/yzwen/SpeakerInvariantMER/tools/opensmile-2.3.0/'
PATH_TO_FFMPEG = '/disk6/yzwen/SpeakerInvariantMER/tools/ffmpeg-4.4.1-i686-static/ffmpeg'
PATH_TO_NOISE = '/disk6/yzwen/SpeakerInvariantMER/tools/musan/speech'

SAVED_ROOT = os.path.join('./saved')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')


############ For Windows (openface-win) ##############
DATA_DIR_Win = {
	'MER2023': 'H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\dataset-process',
}

PATH_TO_RAW_FACE_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'video'),
}

PATH_TO_FEATURES_Win = {
	'MER2023':   os.path.join(DATA_DIR_Win['MER2023'],   'features'),
}

PATH_TO_OPENFACE_Win = "H:\\desktop\\Multimedia-Transformer\\MER2023-Baseline-master\\tools\\openface_win_x64"
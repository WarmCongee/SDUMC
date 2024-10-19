export CUDA_VISIBLE_DEVICES=2

python ./extract_wavlm_vicuna.py --dataset='CMU-MOSEI' --gpu=2 --model_name='vicuna-7b-v1.5' --feature_level='FRAME' 
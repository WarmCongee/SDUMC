export CUDA_VISIBLE_DEVICES=0

python ./extract_text_embedding_huggingface.py --dataset='CMU-MOSEI' --gpu=0 --model_name='VICUNA_7B' --feature_level='FRAME' --language='english'

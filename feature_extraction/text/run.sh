export CUDA_VISIBLE_DEVICES=3

/disk6/yzwen/miniconda3/envs/troch2.0/bin/python ./extract_text_embedding_huggingface.py --dataset='CMU-MOSEI' --gpu=3 --model_name='Baichuan2-7B-Base' --feature_level='FRAME' --language='english'

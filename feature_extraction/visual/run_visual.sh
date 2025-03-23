# python ./extract_manet_embedding.py --dataset='MER2023' --feature_level='UTTERANCE' --gpu='2'
# python ./extract_manet_embedding.py --dataset='MER2023' --feature_level='FRAME' --gpu='3'

# python ./extract_ferplus_embedding.py --dataset='MER2023' --feature_level='UTTERANCE' --gpu='1' --model_name='resnet50_ferplus_dag'
# python ./extract_ferplus_embedding.py --dataset='MER2023' --feature_level='FRAME' --gpu='0' --model_name='resnet50_ferplus_dag'

# python ./extract_vision_huggingface.py \
# --dataset='MER2023' --gpu=3 --model_name='clip-vit-large-patch14' --feature_level='FRAME' 


# Recommended to configure and operate in Windows
python ./extract_openface.py --overwrite=True  --dataset='CMU-MOSEI' --type=videoOne


# Operation in Linux
python ./extract_manet_embedding.py \
--dataset='CMU-MOSEI' --gpu=3 --feature_level='FRAME' 

# nohup /disk6/yzwen/miniconda3/envs/torch1.10/bin/python /disk6/yzwen/SpeakerInvariantMER/feature_extraction/visual/extract_manet_embedding.py --dataset='MER2023' --feature_level='UTTERANCE' --gpu='2' >utt_manet_log.out 2>&1 &
# nohup /disk6/yzwen/miniconda3/envs/torch1.10/bin/python /disk6/yzwen/SpeakerInvariantMER/feature_extraction/visual/extract_manet_embedding.py --dataset='MER2023' --feature_level='FRAME' --gpu='3' >fra_manet_log.out 2>&1 &

# nohup /disk6/yzwen/miniconda3/envs/torch1.10/bin/python /disk6/yzwen/SpeakerInvariantMER/feature_extraction/visual/extract_ferplus_embedding.py --dataset='MER2023' --feature_level='UTTERANCE' --gpu='1' --model_name='resnet50_ferplus_dag' >utt_resnet_log.out 2>&1 &
# nohup /disk6/yzwen/miniconda3/envs/torch1.10/bin/python /disk6/yzwen/SpeakerInvariantMER/feature_extraction/visual/extract_ferplus_embedding.py --dataset='MER2023' --feature_level='FRAME' --gpu='0' --model_name='resnet50_ferplus_dag' >fra_resnet_log.out 2>&1 &

/disk6/yzwen/miniconda3/envs/torch1.13/bin/python ./extract_vision_huggingface.py --dataset='MER2023' --gpu=3 --model_name='clip-vit-large-patch14' --feature_level='FRAME' 

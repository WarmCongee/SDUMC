
python -u ./main_frame_val_text_missing_inference.py \
--dataset='CMU-MOSEI' --valid_dataset='CMU-MOSEI_valid' \
--test_dataset='CMU-MOSEI_test' \
--model='wengnet_mosei_mult_views_text_missing' \
--test_sets='test3'  \
--num_workers=4 \
--audio_feature='wavlm-large-FRA_-5' \
--text_feature='vicuna-7b-v1.5-FRA-wavlm2vicuna-half-gt' \
--video_feature='manet_FRA' \
--feat4_feature='vicuna-7b-v1.5-FRA-wavlm2vicuna-half-wav+prompt[take_generate_wordembed_-4]' \
--checkpoint_path='checkpoints/mosei_mult-view_kd_full_0.5060_0.5503.pt' \
--batch_size=128 \
--lr=1e-4 \
--epochs=1 \
--gpu=3 \
--text_feat_loss_w=0.1 \
--text_query_feat_loss_w=0.7 \
--features_loss_w=0.13 \
--rnc_loss_w=0.5


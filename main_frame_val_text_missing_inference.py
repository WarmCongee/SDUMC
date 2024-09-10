
import os
import sys
import time
import copy
import tqdm
import glob
import json
import math
import scipy
import shutil
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing
import copy

import sklearn
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from metric import *
from toolkit.dataloader import get_dataloaders
from toolkit.models import get_models
from toolkit.utils.loss import *
import config
from scipy.ndimage import gaussian_filter1d


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))


class Proj(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(Proj, self).__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)

    def forward(self, input_tensor):
        output_tensor = self.input_proj(input_tensor)
        output_tensor = F.normalize(output_tensor, p=2, dim=1)
        return output_tensor

########################################################
########### main training/testing function #############
########################################################
def train_or_eval_model(args, model, cl_proj, losses, dataloader, optimizer=None, train=False, task='all'):
    
    vidnames = []
    val_preds_full, val_preds_missing, val_labels = [], [], []
    emo_probs, emo_labels = [], []
    embeddings_full = []
    embeddings_missing = []
    embeddings_full_rnc = []
    embeddings_missing_rnc = []

    
    text_rep_query_full = []
    text_rep_query_missing = []

    text_rep_full = []
    text_rep_missing = []

    attention_masks= []
    p = 0.2

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        ## analyze dataloader
        audio_feat, text_feat, visual_feat, feat4_feat = data[0]['audios'], data[0]['texts'], data[0]['videos'], data[0]['feat4s']
        if task=='val':
            vals = data[-2].float()
            vals = vals.cuda()
        else:
            emos, vals = data[-3], data[-2].float()
            emos = emos.cuda()
            vals = vals.cuda()
        vidnames += data[-1]
        # multi_feat = torch.cat([audio_feat, text_feat, visual_feat], dim=1)

        ## add cuda
        audio_feat  = audio_feat.cuda()
        text_feat   = text_feat.cuda()
        visual_feat = visual_feat.cuda()
        feat4_feat = feat4_feat.cuda()
        # multi_feat  = multi_feat.cuda()

        ## feed-forward process
        # features, vals_out, attention_mask = model([audio_feat, text_feat, visual_feat])
        # features, vals_out, _= model([audio_feat, text_feat, visual_feat])
        #emo_probs.append(emos_out.data.cpu().numpy())

        ## optimize params
        if train:
            vals_out_0, embeddings_0= model([audio_feat, text_feat, visual_feat, False])
            features_0, rnc_feat_0, text_feat_0, text_query_feat_0 = embeddings_0
            

            mask = torch.rand((audio_feat.size(0), audio_feat.size(1), 1)) > p  # 生成随机掩码
            mask = mask.cuda()
            dropped_audio_feat = audio_feat * mask.expand_as(audio_feat)  # 应用掩码并调整未被丢弃的权重

            mask = torch.rand((visual_feat.size(0), visual_feat.size(1), 1)) > p  # 生成随机掩码
            mask = mask.cuda()
            dropped_visual_feat = visual_feat * mask.expand_as(visual_feat)  # 应用掩码并调整未被丢弃的权重

            vals_out_1, embeddings_1 = model([audio_feat, feat4_feat, visual_feat, True])
            features_1, rnc_feat_1, text_feat_1, text_query_feat_1 = embeddings_1

            n_views_feature = torch.stack((rnc_feat_0, rnc_feat_1), dim=1)
            # loss1 = cls_loss(emos_out, emos)

            MSEloss_0 = losses['reg_loss'](vals_out_0, vals)
            MSEloss_1 = losses['reg_loss'](vals_out_1, vals)

            rnc_loss = losses['rnc_loss'](n_views_feature, vals.unsqueeze(1))

            # loss = 0.5*(MSEloss_0 + MSEloss_1) + 5*losses[4](features_0, features_1) + 0.8*rnc_loss # 
            loss = 0.5*(MSEloss_0 + MSEloss_1) + args.text_feat_loss_w*losses['rmse_loss'](text_feat_1, text_feat_0.detach()) + args.text_query_feat_loss_w*losses['rmse_loss'](text_query_feat_1, text_query_feat_0.detach()) + args.features_loss_w*losses['rmse_loss'](features_1, features_0) + args.rnc_loss_w*rnc_loss #    + 0.1*losses[4](text_feat_1, text_feat_0.detach()) + 0.7*losses[4](text_query_feat_1, text_query_feat_0.detach()) + 0.1*losses[4](features_1, features_0)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                vals_out_0, embeddings_0= model([audio_feat, text_feat, visual_feat, False])
                vals_out_1, embeddings_1= model([audio_feat, feat4_feat, visual_feat, True])

        val_preds_full.append(vals_out_0.detach().cpu().numpy())
        val_preds_missing.append(vals_out_1.detach().cpu().numpy())
        val_labels.append(vals.detach().cpu().numpy())
        embeddings_full.append(embeddings_0[0].detach().cpu().numpy())
        embeddings_missing.append(embeddings_1[0].detach().cpu().numpy())
        embeddings_full_rnc.append(embeddings_0[1].detach().cpu().numpy())
        embeddings_missing_rnc.append(embeddings_1[1].detach().cpu().numpy())

        text_rep_query_full.append(embeddings_0[2].detach().cpu().numpy())
        text_rep_query_missing.append(embeddings_1[2].detach().cpu().numpy())

        text_rep_full.append(embeddings_0[3].detach().cpu().numpy())
        text_rep_missing.append(embeddings_1[3].detach().cpu().numpy())
        # for idx, _ in enumerate(attention_mask[0].detach().cpu().numpy()):
        #     import matplotlib.pyplot as plt
        #     _ = np.squeeze(_, axis=1)
        #     _ = gaussian_filter1d(_, 4)
        #     plt.bar(range(len(_)),_)
        #     plt.savefig('attention_images/'+ str(idx) + ".png")
        #     plt.cla()

    ## evaluate on dimensional labels
    val_preds_full = np.concatenate(val_preds_full, axis=0)
    val_preds_missing = np.concatenate(val_preds_missing, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_mse = mean_squared_error(val_labels, val_preds_full)
    embeddings_full = np.concatenate(embeddings_full, axis=0)
    embeddings_missing = np.concatenate(embeddings_missing, axis=0)
    embeddings_full_rnc = np.concatenate(embeddings_full_rnc, axis=0)
    embeddings_missing_rnc = np.concatenate(embeddings_missing_rnc, axis=0)

    text_rep_query_full = np.concatenate(text_rep_query_full, axis=0)
    text_rep_query_missing = np.concatenate(text_rep_query_missing, axis=0)
    text_rep_full = np.concatenate(text_rep_full, axis=0)
    text_rep_missing = np.concatenate(text_rep_missing, axis=0)

    save_results = {}
    # item1: statistic results
    save_results['val_mse'] = val_mse
    save_results['val_preds_full'] = val_preds_full
    save_results['val_preds_missing'] = val_preds_missing
    save_results['val_labels'] = val_labels
    save_results['names'] = vidnames
    save_results['full_rep'] = embeddings_full
    save_results['missing_rep'] = embeddings_missing
    save_results['full_rnc'] = embeddings_full_rnc
    save_results['missing_rnc'] = embeddings_missing_rnc

    save_results['text_rep_query_full'] = text_rep_query_full
    save_results['text_rep_query_missing'] = text_rep_query_missing
    save_results['text_rep_full'] = text_rep_full
    save_results['text_rep_missing'] = text_rep_missing
    # if args.savewhole: save_results['embeddings'] = embeddings
    return save_results


########################################################
############# metric and save results ##################
########################################################
def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score


def gain_name2feat(folder_save, testname):
    name2feat = {}
    assert len(folder_save) >= 1
    names      = folder_save[0][f'{testname}_names']
    embeddings = folder_save[0][f'{testname}_embeddings']
    for jj in range(len(names)):
        name = names[jj]
        embedding = embeddings[jj]
        name2feat[name] = embedding
    return name2feat


def record_exp_result(cv_fscore, cv_valmse, cv_metric, args_saved_path):
    save_path = config.PATH_TO_RESULT['RESULT_CSV']
    result_text = "fscore: {:.4f}, valmse: {:.4f}, metric: {:.4f}, train_args_path: {}".format(cv_fscore, cv_valmse, cv_metric, args_saved_path)
    # t = ",".join([str(item) for item in t])
    f = open(save_path, "a")
    f.write(result_text + '\n')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--train_dataset', type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--valid_dataset',  type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--test_dataset',  type=str, default=None, help='dataset') # for cross-dataset evaluation
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--feat4_feature', type=str, default=None, help='4th feature name')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--test_sets', type=str, default='test1,test2', help='process on which test sets, [test1, test2, test3]')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether save latent embeddings')
    parser.add_argument('--feat_type',  type=str, default='frm_unalign', help='feature type [utt, frm_align, frm_unalign]')
    parser.add_argument('--feat_scale', type=int, default=1, help='pre-compress input from [seqlen, dim] -> [seqlen/scale, dim]')

    ## Params for model
    parser.add_argument('--model', type=str, default='wengnet', help='model name for training [wengnet]')
    parser.add_argument('--layers', type=str, default='256,128', help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=-1, help='number of classes [defined by args.label_path]')
    parser.add_argument('--num_folder', type=int, default=-1, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type for training [mlp or attention]')
    parser.add_argument('--text_feat_loss_w', type=float, default=0.1, help='model type for training [mlp or attention]')
    parser.add_argument('--text_query_feat_loss_w', type=float, default=0.7, help='model type for training [mlp or attention]')
    parser.add_argument('--features_loss_w', type=float, default=0.1, help='model type for training [mlp or attention]')
    parser.add_argument('--rnc_loss_w', type=float, default=0.8, help='model type for training [mlp or attention]')
    
    ## Params for training
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')

    ## Params for Distribution
    parser.add_argument('--local_rank', default=0, type=int, help='Process rank')
    args = parser.parse_args()

    args.n_classes = 6
    args.num_folder = 5
    args.test_sets = args.test_sets.split(',')

    # if args.dataset is not None:
    #     args.train_dataset = args.dataset
    if args.test_dataset is None:
        args.test_dataset  = args.dataset
    if args.valid_dataset is None:
        args.valid_dataset  = args.dataset
    assert args.valid_dataset is not None
    assert args.test_dataset  is not None

    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    if len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(args)

    
    print(f'====== Reading Data =======')
    get_dataloaders = get_dataloaders(args)
    train_loaders, eval_loaders, test_loaders, input_dims = get_dataloaders.get_loaders()         
    # assert len(train_loaders) == args.num_folder, f'Error: folder number'
    # assert len(eval_loaders)  == args.num_folder, f'Error: folder number'
    
    args.input_dims = input_dims
    
    print (f'====== Training and Evaluation =======')
    folder_save = []
    folder_evalres = []
    best_epoch_valid = {'mae': 1.0, 'f1': 0}
    best_epoch_test_full = {'mae': 1.0, 'f1': 0}
    best_epoch_test_missing = {'mae': 1.0, 'f1': 0}
    for ii in range(len(train_loaders)):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()
        name_time  = time.time()

        print (f'Step1: build model (each folder has its own model)')
        model = get_models(args)
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('/disk6/yzwen/SpeakerInvariantMER/shell/mosei_mult-view_kd_full_0.5088659491481743_17.pt')['state_dict'].items()}, strict=False)
        model = model.cuda()
        model.eval()
        getModelSize(model)
        cl_proj = Proj().cuda()
        # torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        # model = torch.nn.parallel.DataParallel(model)
        reg_loss = MSELoss().cuda()
        rmse_loss = RMSELoss().cuda()
        coss_loss = CosineSimilarityLoss4Seq().cuda()
        cls_loss = CELoss().cuda()
        kl_loss = KLLoss().cuda()
        rnc_loss = RnCLoss().cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        gamma = 0.9; stepsize = 10; warm_up_epochs=5
        warm_up_with_step_lr = lambda epoch: (epoch+1) / warm_up_epochs if epoch < warm_up_epochs \
            else gamma**( (epoch+1 - warm_up_epochs)//stepsize )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_step_lr)

        print (f'Step2: training (multiple epoches)')
        eval_metrics = []
        eval_fscores = []
        eval_valmses = []
        test_save = []
        last_sava_path = ''

        losses = {'reg_loss': reg_loss, 'cls_loss': cls_loss, 'kl_loss': kl_loss, 'rnc_loss': rnc_loss,'rmse_loss': rmse_loss,'coss_loss': coss_loss}

        for epoch in range(args.epochs):

            store_values = {}
            ## training and validation
            train_start = time.time()
            train_results = train_or_eval_model(args, model, cl_proj, losses, train_loader, optimizer=optimizer, train=False)
            train_end = time.time()
            print('used: {} s'.format(train_end-train_start))
            eval_results  = train_or_eval_model(args, model, cl_proj, losses, eval_loader,  optimizer=None,      train=False)
            scheduler.step()
            eval_valmses.append(eval_results['val_mse'])
        
            # store_values['eval_valpreds'] = eval_results['val_preds']
            # store_values['eval_names']    = eval_results['names']
            print ('epoch:%d; train_val_mse:%.4f' %(epoch+1, train_results['val_mse']))

            ## testing and saving： test in all trained dataset
            # for jj, test_loader in enumerate(test_loaders):
            test_set = args.test_sets[ii]
            test_results = train_or_eval_model(args, model, cl_proj, losses, test_loader, optimizer=None, train=False)


            # store_values[f'{test_set}_valpreds']   = test_results['val_preds']
            # store_values[f'{test_set}_vallabels']   = test_results['val_labels']
            # if args.savewhole: store_values[f'{test_set}_embeddings'] = test_results['embeddings']
            # test_save.append(store_values)


            # valid_result = eval_mosei_metric(eval_results['val_preds'], eval_results['val_labels'], eval_results['names'])
            # if valid_result['mae'] <= best_epoch_valid['mae']: # and valid_result['f1'] >= best_epoch_valid['f1']:
            #     best_epoch_valid = valid_result
            #     best_epoch_valid['epoch'] = epoch
                
             
            test_result_full = eval_mosei_metric(test_results['val_preds_full'], test_results['val_labels'], test_results['names'])
            test_result_missing = eval_mosei_metric(test_results['val_preds_missing'], test_results['val_labels'], test_results['names'])
            
            print(test_result_full)
            print(test_result_missing)


            print("-" * 50)
        
        end_time = time.time()
        print (f'>>>>> Finish: training on the {ii+1} data, duration: {end_time - start_time} >>>>>')


    print (f'====== Gain predition on test data =======')
    save_modelroot = os.path.join(args.save_root, 'model')
    save_predroot  = os.path.join(args.save_root, 'prediction')
    if not os.path.exists(save_predroot): os.makedirs(save_predroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    feature_name = f'{args.audio_feature}+{args.text_feature}+{args.video_feature}'

    # np.save(f'/disk6/yzwen/SpeakerInvariantMER/shell/wo_MIA_reps_{best_epoch_test["mae"]}.npy', test_results)

    # print("best_valid:")
    # print(best_epoch_valid)
    # print("best_test:")
    # print(best_epoch_test)
    # with open('paper_exp.txt', mode='a') as filename:
    #     filename.write(f'--text_feat_loss_w={args.text_feat_loss_w} --text_query_feat_loss_w={args.text_query_feat_loss_w} --features_loss_w={args.features_loss_w} --rnc_loss_w={args.rnc_loss_w}\n')
    #     filename.write(str(best_epoch_test))
    #     filename.write('\n') # 换行
    # print(feature_name)



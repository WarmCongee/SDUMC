import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from ..globals import *
from toolkit.data import get_datasets
from toolkit.utils.read_data import DataLoaderX

too_long_data = [
    '125344_0',
    'SqAiJrvHXNA_0',
    '30162_9',
    '96361_16',
    '6-0bcijTR8k_0',
    '254427_0',
    'PEBwwe0PLZ8_0',
    'JGEEA_JVriE_0',
    'skRqBxLLJkE_0',
    'd-Uw_uZyUys_1',
    'jjbOD6u7V34_16',
    'IRSxo_XXArg_11',
    'aNOuoSVlunM_5',
    'veHYwR7ge6Y_0',
    '9K5mYSaoBL4_2',
    'd-Uw_uZyUys_2',
    '139006_5',
    '245243_1',
    '4Vl6AeEkAg4_1',
    'mHEtr7PHxoA_0',
    '70710_2',
    'fsBzpr4k3rY_0',
    'wI7DDCRh4Nw_1',
    '69707_3',
    '4oeKDFIaL7o_4',
    'y3r2kk8zvl0_3',
    '6UV6ktwbLoo_0',
    'HR18U0yAlTc_5',
    'MFrwi-RibUk_3',
    '83310_2',
    '69707_4',
    '112425_10',
    'vttEPA6Xffk_1',
    'gLTxaEcx41E_3',
    '8XODJwsvBa0_2',
    '130149_5',
    'NuRvTWhELqs_5',
    'GSnt_fW8qjI_4',
    'dQ56b0bqmc8_3',
    'xkEK17UUyi4_0',
    '193291_0',
    'slLRsFFiiRc_5',
    'dQ56b0bqmc8_0',
    '275248_0',
    'cX8FScpsfLE_0',
    '243646_10',
    'SqAiJrvHXNA_6',
    '264418_7',
    'AggyS1coOb8_1',
    'XXvSLz8QmGk_11',
    'XaVYxIW0FDg_2',
]


# CMU 数据集测试的时候，是包括 [train, val, test]
class CMUMOSEI:
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_path = config.PATH_TO_LABEL[args.dataset]
        # self.valid_label_path = config.PATH_TO_LABEL[args.valid_dataset]
        # self.test_label_path = config.PATH_TO_LABEL[args.test_dataset]

        self.dataset = args.dataset
        assert self.dataset in ['CMU-MOSEI']
        
        # update args
        args.output_dim1 = 0
        args.output_dim2 = 1
        args.metric_name = 'emo'

    def get_loaders(self):
        dataloaders = []
        input_dims = []
        for data_type in ['train', 'val', 'test']:
            names, labels = self.read_names_labels(self.label_path, data_type, debug=self.debug)
            if data_type == 'train':
                for item in too_long_data:
                    if item in names:
                        idx_temp = names.index(item)
                        names.pop(idx_temp)
                        labels.pop(idx_temp)
            print (f'{data_type}: sample number {len(names)}')

            if data_type in ['train', 'val']:
                dataset = get_datasets(self.args, names, labels)
            else:
                dataset = get_datasets(self.args, names, labels)
            # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

            if data_type in ['train']:
                dataloader = DataLoaderX(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        pin_memory=True,
                                        # sampler=sampler,
                                        collate_fn=dataset.collater,
                                        prefetch_factor=8)
            else:
                input_dims = dataset.get_featdim()
                                        
                dataloader = DataLoaderX(dataset,
                                        batch_size=self.batch_size,
                                        num_workers=self.num_workers,
                                        pin_memory=True,
                                        #sampler=sampler,
                                        collate_fn=dataset.collater,
                                        shuffle=False,
                                        prefetch_factor=8)
            
            
            dataloaders.append(dataloader)
        train_loaders = [dataloaders[0]]
        eval_loaders  = [dataloaders[1]]
        test_loaders  = [dataloaders[2]]

                
        return train_loaders, eval_loaders, test_loaders, input_dims
    

    def read_names_labels(self, label_path, data_type, debug=False):
        names, labels = [], []
        if data_type == 'train': corpus = np.load(label_path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'val':   corpus = np.load(label_path, allow_pickle=True)['val_corpus'].tolist()
        if data_type == 'test':  corpus = np.load(label_path, allow_pickle=True)['test_corpus'].tolist()
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels


    # CMU 采用的指标，是将val转成2分类计算 ACC, WAF
    def calculate_results(self, emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):
        
        non_zeros = np.array([i for i, e in enumerate(val_labels) if e != 0]) # remove 0, and remove mask
        emo_accuracy = accuracy_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0))
        emo_fscore = f1_score((val_labels[non_zeros] > 0), (val_preds[non_zeros] > 0), average='weighted')

        results = { 
                    'valpreds':  val_preds,
                    'vallabels': val_labels,
                    'emoacc':    emo_accuracy,
                    'emofscore': emo_fscore
                    }
        outputs = f'f1:{emo_fscore:.4f}_acc:{emo_accuracy:.4f}'

        return results, outputs
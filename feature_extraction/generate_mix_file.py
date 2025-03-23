#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os, argparse, random, json, codecs
import numpy as np


def main_generate(sample_names, mix_num, mix_probability, alpha, json_path):
    name2sample_weight = {}
    for current_i in range(len(sample_names)):
        current_ies = [current_i]
        current_names = [sample_names[current_i]]
        current_weights = np.array([1])
        
        for j in range(mix_num):
            if random.uniform(0, 1) < mix_probability:
                mixed_i = random.randint(0, len(sample_names) - 1)
                if mixed_i in current_ies:
                    pass
                else:
                    current_ies.append(mixed_i)
                    current_names.append(sample_names[mixed_i])
                    lam = np.random.beta(alpha, alpha)
                    current_weights = np.append((1-lam)*current_weights, lam)
        name2sample_weight[sample_names[current_i]] = {'name': current_names, 'weight': current_weights.tolist()}
    
    json_dir = os.path.split(json_path)[0]
    if not os.path.exists(json_dir):
        os.makedirs(json_dir, exist_ok=True)
    
    with codecs.open(json_path, 'w') as handle:
        json.dump(name2sample_weight, handle, indent=4, ensure_ascii=False)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run.')
    parser.add_argument('--feature_dir', type=str, default='0', help='index of gpu')
    parser.add_argument('--mix_num', type=int, default=1, help='index of gpu')
    parser.add_argument('--mix_probability', type=float, default=0.5, help='index of gpu')
    parser.add_argument('--alpha', type=float, default=0.2, help='index of gpu')
    args = parser.parse_args()
    
    all_train_sample_names = list(np.load('/disk3/htwang/MER2023-Baseline-master/dataset-process/label-6way.npz', allow_pickle=True)['train_corpus'].tolist().keys())
    main_generate(sample_names=all_train_sample_names, mix_num=args.mix_num, mix_probability=args.mix_probability, alpha=args.alpha, json_path=os.path.join(args.feature_dir, 'name2sample_weight.json'))
    
# python feature_extraction/generate_mix_file.py --feature_dir dataset-process/mixup_202307041119 --mix_num 1 --mix_probability 0.5 --alpha 0.5

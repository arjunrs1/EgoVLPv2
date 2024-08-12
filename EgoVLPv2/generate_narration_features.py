# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tqdm
import argparse
import numpy as np
import transformers
import torch
import torch.nn as nn
import model.metric as module_metric
import data_loader.data_loader as module_data
from utils import state_dict_data_parallel_fix
from parse_config import ConfigParser
import pdb
from model.model import FrozenInTime
from transformers import RobertaTokenizer
import model.model as module_arch


def run():
    # setup data_loader instances
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = 1
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride

    data_loader = config.initialize('data_loader', module_data)

    model = config.initialize('arch', module_arch)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['arch']['args']['text_params']['model'],
                                                               TOKENIZERS_PARALLELISM=False)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    device = torch.device(args.cuda_base if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = nn.DataParallel(model, device_ids = args.device_ids)
    model.eval()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # extract narration features
    dim = config.config['arch']['args']['projection_dim']
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if os.path.exists(os.path.join(args.save_dir, data['meta']['video_uid'][0], data['meta']['narration_uid'][0]+'.pt')):
                print(f"{data['meta']['narration_uid']} is already.")
                continue
            try:
                data['text'] = tokenizer(data['text'], return_tensors='pt', padding='max_length', max_length=30, truncation=True)
            except:
                print(data['text'])
                continue
            data['text'] = {key: val.cuda() for key, val in data['text'].items()}
            ret = model.module.infer(data, return_embeds=True, task_names="Feature_Extraction_Text", ret={}) #TODO: Look at task names; dual implies reliance on vid features which we dont want
            text_embed = ret['text_embeds']
            if not os.path.exists(os.path.join(args.save_dir, data['meta']['video_uid'][0])):
                os.makedirs(os.path.join(args.save_dir, data['meta']['video_uid'][0]))
            torch.save(text_embed, os.path.join(args.save_dir, data['meta']['video_uid'][0], data['meta']['narration_uid'][0]+'.pt'))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('--save_dir',
                      help='path to store text & video feats, this is for saving embeddings if you want to do offline retrieval.')
    args.add_argument('--cuda_base', help="in form cuda:x")
    args.add_argument('--device_ids', help='delimited list input', type=lambda s: [int(item) for item in s.split(',')])

    config = ConfigParser(args, test=True, eval_mode='epic_nfg')
    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride

    run()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch
import argparse

from utils.options import args_parser
from utils.train_utils import get_model
from models.Update import LocalUpdateHydro
from models.test import test_img, test_img_local, test_img_local_all
import os
import pdb
from torch.utils.data import TensorDataset


if __name__ == '__main__':
    args = args_parser() 
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    '''args.model = 'bilstm'
    args.input_size = 5 
    args.hidden_size = 64 
    args.num_layers = 2  
    args.output_size = 1  
    args.regression_output_size = 1 '''

    net_glob = get_model(args)
    net_glob.train()

    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    
    for iter in range(1):
        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        task = iter//10
        # Local Updates
        print(idxs_users)
        for idx in idxs_users:
            local = LocalUpdateHydro(args=args, idxs=idx, global_model=net_glob, task = task)
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        
        # Broadcast
        update_keys = list(w_glob.keys())
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        net_glob.load_state_dict(w_glob, strict=False)
        net_best = copy.deepcopy(net_glob)
    best_save_path = '/root/full_model_new.pt'
    torch.save(net_best.state_dict(), best_save_path)
                

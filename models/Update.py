#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer
from utils.data_utils import load_train_data, load_test_data
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        
        if local_eps is None:
            local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep
        
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                labels = labels.view(-1, 1)
                loss = self.loss_func(logits, labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print(f"Epoch {iter+1}, Loss: {epoch_loss[-1]}")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdateHydro(object):
    def __init__(self, args, idxs=None, global_model=None, task=None, pretrain=False):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.task = task
        self.total_rounds = 50

    def train(self, net, lr):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        
        optimizer = torch.optim.SGD(body_params, lr=lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.local_ep, eta_min=0)
        epoch_loss = []
        
        local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            # Check if the region is well-trained
            if iter > 0 and abs(epoch_loss[-1] - epoch_loss[-2]) < 0.01:
                self.args.local_bs //= 2  # Reduce batch size by half for well-trained regions
                self.ldr_train = load_train_data(idxs, task, batch_size=self.args.local_bs)
            elif iter > 0 and abs(epoch_loss[-1] - epoch_loss[-2]) >= 0.01:
                self.args.local_bs = self.args.local_bs * 2  # Increase batch size for less-trained regions
                self.ldr_train = load_train_data(idxs, task, batch_size=self.args.local_bs)

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.float().unsqueeze(1)  # Ensure labels are Float type and size (batch_size, 1)
                logits = net(images)
                
                mse_loss = self.loss_func(logits, labels)
                mae_loss = nn.L1Loss()(logits, labels)
                gamma = 0.5 + (0.5 * iter) / 50 

                total_loss = gamma * mse_loss + (1-gamma) * mae_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_loss.append(total_loss.item())
            scheduler.step()  
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(f"Epoch {iter + 1}/{local_eps}, Loss: {epoch_loss[-1]}")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    '''def calculate_nse(self, predictions, targets):
        # 确保 predictions 和 targets 是相同的形状
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")
        numerator = torch.mean((predictions - targets) ** 2)
        mean_targets = torch.mean(targets)
        denominator = torch.mean((targets - mean_targets) ** 2)
        if denominator == 0:
            print('the value is 0')
            return 1.0
        nse = 1 - (numerator / denominator)
        if torch.isnan(nse):
            raise ValueError("NSE calculation resulted in NaN.")
        
        return nse'''


    

    

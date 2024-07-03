import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

from .hydro import *

def read_data(kind='train'):
    hydro_dataset = HydroDataSetUnit17()

    if kind == 'train':
        train_x, train_y, _, _ = hydro_dataset.load()
        return {'x': train_x, 'y': train_y}

    else:
        _, _, test_x, test_y = hydro_dataset.load()
        return {'x': test_x, 'y': test_y}


def read_client_data(idx, task, is_train=True):
    data = read_data(kind='train' if is_train else 'test')
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return {'x': X, 'y': y}

def load_train_data(idx, task, batch_size=16):
    train_data = read_client_data(idx, task, is_train=True)
    train_dataset = TensorDataset(train_data['x'], train_data['y'])
    return DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

def load_test_data(idx, batch_size=16):
    test_data = read_client_data(idx, task=0, is_train=False)
    test_dataset = TensorDataset(test_data['x'], test_data['y'])
    return DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)


if __name__ == '__main__':
    dataset_path = './dataset/series_data/'
    train_data = read_client_data(dataset_path, idx=0, task=0, is_train=True)
    test_data = read_client_data(dataset_path, idx=0, task=0, is_train=False)

    print(train_data['x'].shape)
    print(train_data['y'].shape)
    print(test_data['x'].shape)
    print(test_data['y'].shape)

import pickle
from typing import Dict

import numpy as np
import os
import hashlib
from abc import ABCMeta,abstractmethod

from .interfaces import AbsDataset


class IFedDataset(AbsDataset):

    @abstractmethod
    def set_node_id(self, node_id: int) -> None:
        pass


class Femnist(IFedDataset):

    def __init__(self, check_sum=None):
        self.path = './dataset/femnist_data/'
        if check_sum is None:
            self.num = 0
        else:
            self.num = int(check_sum.split(",")[0])
        self.file_name: str = ""
        super().__init__(check_sum)

    def set_node_id(self, node_id):
        self.num = node_id

    def __repr__(self):
        return '<Femnist classification dataset, only have 8 nodes 01.>'

    def load(self) -> tuple:
        train_data = np.load(self.path+str(self.num)+".pkl", allow_pickle=True)
        test_data = np.load(self.path+"8.pkl", allow_pickle=True)
        # 01: Dict[str, np.ndarray] = np.load(self.path + "cifar10", allow_pickle=True)[()]
        return np.asarray(train_data[0]), np.asarray(train_data[1]), np.asarray(test_data[0]), np.asarray(test_data[1])

    def check_sum(self) -> str:
        if not os.path.exists(self.path+str(self.num)+".pkl"):
            return ''
        sum = hashlib.md5()  # 获取一个md5加密算法对象
        with open(self.path+str(self.num)+".pkl", 'rb') as f:
            sum.update(f.read())   # 制定需要加密的字符串
        return str(self.num)+","+sum.hexdigest()  # 获取加密后的16进制字符串

    def extract_files(self) -> list:
        files = [self.path+str(self.num)+".pkl",self.path+"8.pkl"]
        return files

    def estimate_size(self) -> int:
        return 756274995   #??MB


if __name__ == '__main__':
    fed = Femnist()
    fed.set_node_id(1)
    x, y, x_t, y_t = fed.load()

    print(x[0])
import numpy as np
import os
import hashlib
from .femnist import IFedDataset


class HydroDataSetUnit17(IFedDataset):

    def set_node_id(self, node_id: int) -> None:
        pass

    def __init__(self, check_sum=None):

        # 路径写成这样是因为是在主程序调用当前类
        self.path = '/root/pythonProject/dataset'
        # self.path = path
        super().__init__(check_sum)

    def __repr__(self):
        return '<TEST prediction dataset.>'

    # 获取数据文件校验和
    def check_sum(self) -> str:

        # os.path.join() 路径拼接
        if not (os.path.exists(os.path.join(self.path, 'train_unit18_8basins_t30_x.txt')) and
                os.path.exists(os.path.join(self.path, 'train_unit18_8basins_t30_y.txt')) and
                os.path.exists(os.path.join(self.path, 'test_unit18_8basins_t30_x.txt')) and
                os.path.exists(os.path.join(self.path, 'test_unit18_8basins_t30_y.txt'))):
            return ''

        sum = hashlib.md5()  # 获取一个md5加密算法对象
        with open(os.path.join(self.path, 'train_unit18_8basins_t30_x.txt'), 'rb') as file:
            sum.update(file.read())  # 制定需要加密的字符串
        with open(os.path.join(self.path, 'train_unit18_8basins_t30_y.txt'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, 'test_unit18_8basins_t30_x.txt'), 'rb') as file:
            sum.update(file.read())
        with open(os.path.join(self.path, 'test_unit18_8basins_t30_y.txt'), 'rb') as file:
            sum.update(file.read())
        return sum.hexdigest()  # 获取加密后的16进制字符串

    def extract_files(self) -> list:
        return [
            os.path.join(self.path, 'train_unit18_8basins_t30_x.txt'),
            os.path.join(self.path, 'train_unit18_8basins_t30_y.txt'),
            os.path.join(self.path, 'test_unit18_8basins_t30_x.txt'),
            os.path.join(self.path, 'test_unit18_8basins_t30_y.txt')
        ]

    def estimate_size(self) -> int:
        return 31508340 + 94527825  # 约120M

    def __load_core(self, kind='train'):
        """Load hydro 01 from `path`"""
        x_path = os.path.join(self.path, '%s_unit18_8basins_t30_x.txt' % kind)
        y_path = os.path.join(self.path, '%s_unit18_8basins_t30_y.txt' % kind)
        # 注意, time_step和generate_data中的对应起来
        time_step = 30
        x = np.loadtxt(x_path, delimiter=' ').reshape((-1, time_step, 5))
        labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
        return x, labels

    def load(self):
        train_x, train_y = self.__load_core(kind='train')
        test_x, test_y = self.__load_core(kind='test')
        return train_x, np.reshape(train_y, (-1)), test_x, np.reshape(test_y, (-1))


if __name__ == '__main__':

    hydroDataSet = HydroDataSetUnit17()
    print(hydroDataSet.check_sum())
    print(hydroDataSet.extract_files())
    train_x, train_y, test_x, test_y = hydroDataSet.load()
    print(train_x)
    print(train_y)

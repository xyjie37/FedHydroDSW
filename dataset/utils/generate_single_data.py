import numpy as np
from dataset.series_data.utils.camels_operate import CamelsOperate
import sys as sys
import pandas as pd
import os

class GenerateData:
    """产生多个流域数据合并版本数据集"""
    def __init__(self, batch_size, basin_id, data_range, time_step, input_dim):
        # self.path = './dataset/series_data/'
        self.path = 'D:/河海大学/研究课题/水文预报/课题实验/Parallel-SGD/dataset/series_data/'
        self.__batch_size = batch_size
        self.__basin_id = basin_id
        self.__data_range = data_range
        self.__time_step = time_step
        self.__input_dim = input_dim

    def generate(self):
        """Load hydro 01 from path, and merge 01"""
        time_step = self.__time_step
        basin_id = self.__basin_id

        # 训练数据
        train_start_date = self.__data_range['train_date']['start_date']
        train_end_date = self.__data_range['train_date']['end_date']
        ds_train = CamelsOperate(file_path=self.path, basin=basin_id, seq_length=time_step, period="train",
                                 dates=[train_start_date, train_end_date])
        train_basin_x = ds_train.x
        train_basin_y = ds_train.y

        # 测试数据
        test_start_date = self.__data_range['test_date']['start_date']
        test_end_date = self.__data_range['test_date']['end_date']
        ds_test = CamelsOperate(file_path=self.path, basin=basin_id, seq_length=time_step, period="test",
                                dates=[test_start_date, test_end_date], means=ds_train.get_means(),
                                stds=ds_train.get_means())
        test_basin_x = ds_test.x
        test_basin_y = ds_test.y

        return train_basin_x, train_basin_y, test_basin_x, test_basin_y


if __name__ == '__main__':

    def generate1(basin_id):
        """此处生成的数据配置中时间步修改，相应的HydroDataSet1处也要修改
            此处的batch_size, block_size修改了，相应的提交者处的也要修改
        """
        # basin_ids = ['01030500', '01013500', '01022500']
        # basin_ids = ['01030500', '01013500', '01031500']
        # basin_id = '01030500'
        basin_id = basin_id
        date_range = {
            'train_date': {
                'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
                'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")
            },

            'test_date': {
                'start_date': pd.to_datetime("1995-10-01", format="%Y-%m-%d"),
                'end_date': pd.to_datetime("2000-09-30", format="%Y-%m-%d")
            }
        }
        # 此处的batch_size, block_size和数据划分有关
        time_step = 30
        batch_size = 256
        # block_size = 256
        hydroSingleDataSet = GenerateData(batch_size=batch_size, basin_id=basin_id, data_range=date_range, time_step=time_step, input_dim=5)
        train_x, train_y, test_x, test_y = hydroSingleDataSet.generate()
        path = 'D:\\河海大学\\研究课题\\水文预报\\课题实验\\Parallel-SGD\\dataset\\series_data\\01'
        train_path_x = os.path.join(path, str(basin_id)+'\\train_'+str(basin_id)+'_x.txt')
        train_path_y = os.path.join(path, str(basin_id)+'\\train_'+str(basin_id)+'_y.txt')
        test_path_x = os.path.join(path, str(basin_id)+'\\test_'+str(basin_id)+'_x.txt')
        test_path_y = os.path.join(path, str(basin_id)+'\\test_'+str(basin_id)+'_y.txt')
        with open(train_path_x, 'w') as outfile:
            for slice in train_x:
                np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

        with open(train_path_y, 'w') as outfile:
            for slice in train_y:
                np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

        with open(test_path_x, 'w') as outfile:
            for slice in test_x:
                np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

        with open(test_path_y, 'w') as outfile:
            for slice in test_y:
                np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    # generate1('01013500')
    generate1('01031500')
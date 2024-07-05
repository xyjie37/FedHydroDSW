import numpy as np
import pandas as pd
from utils.camels_operate import CamelsOperate
from utils.constants import path


class GetHydroDataWithDate:
    def __init__(self, basin_id, sequence_length, date_range):
        self.__basin_id = basin_id
        self.__date_range = date_range
        self.__sequence_length = sequence_length
        self.__path = path
        self.__ds_train = None
        self.__ds_val = None
        self.__ds_test = None
        self.__x = None
        self.__y = None

    def get_data(self):
        file_path = self.__path
        basin = self.__basin_id
        sequence_length = self.__sequence_length
        train_start_date = self.__date_range['train_date']['start_date']
        train_end_date = self.__date_range['train_date']['end_date']
        ds_train = CamelsOperate(file_path=file_path, basin=basin, seq_length=sequence_length, period="train",
                                 dates=[train_start_date, train_end_date])
        self.__ds_train = ds_train
        self.__x, self.__y = ds_train.get_discharge()
        print("数据长度：", ds_train.__len__())
        print("ds_train[0]，长度", ds_train[0], len(ds_train[0][0]))
        train_x = np.asarray(ds_train.x)
        train_y = np.asarray(ds_train.y)

        means = ds_train.get_means()
        stds = ds_train.get_stds()
        val_start_date = self.__date_range['val_date']['start_date']
        val_end_date = self.__date_range['val_date']['end_date']
        ds_val = CamelsOperate(file_path=file_path, basin=basin, seq_length=sequence_length, period="eval",
                                dates=[val_start_date, val_end_date],
                                means=means, stds=stds)

        self.__ds_val = ds_val
        val_x = np.asarray(ds_val.x)
        val_y = np.asarray(ds_val.y)

        means = ds_train.get_means()
        stds = ds_train.get_stds()
        test_start_date = self.__date_range['test_date']['start_date']
        test_end_date = self.__date_range['test_date']['end_date']
        ds_test = CamelsOperate(file_path=file_path, basin=basin, seq_length=sequence_length, period="eval",
                               dates=[test_start_date, test_end_date],
                               means=means, stds=stds)

        self.__ds_test = ds_test
        test_x = np.asarray(ds_test.x)
        test_y = np.asarray(ds_test.y)

        return train_x, train_y, val_x, val_y, test_x, test_y

    def get_ds_train(self):
        return self.__ds_train

    def get_ds_val(self):
        return self.__ds_val

    def get_ds_test(self):
        return self.__ds_test

    def ds_test_renormalize_discharge(self, Qobs):
        return self.__ds_test.reshape_discharge(Qobs)

    def get_discharge_data(self):
        return self.__x, self.__y


if __name__=='__main__':
    
    basin_id = '02479155'
    the_date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("1982-09-30", format="%Y-%m-%d")
        },
        'val_date': {
            'start_date' : pd.to_datetime("2008-10-01", format="%Y-%m-%d"),
            'end_date' : pd.to_datetime("2009-09-30", format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2004-10-01", format="%Y-%m-%d"),# 没用
            'end_date': pd.to_datetime("2006-09-30", format="%Y-%m-%d")
        },
    }
    sequence_length = 30
    getdata = GetHydroDataWithDate(basin_id, sequence_length, the_date_range)
    print(getdata.get_data())
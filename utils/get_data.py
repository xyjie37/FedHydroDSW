import sys
sys.path.append('/root/pythonProject')

import pandas as pd
import numpy as np
from models.lstm import LSTM, BiLSTM
import torch
from utils.calc_utils import cal_nse_rmse_mae_rae
from utils.get_hydro_data_with_date_range import GetHydroDataWithDate


def train_test_data(basin, the_date_range):
    train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_sparse_data(basin, the_date_range)
    single_basin_data = []
    single_basin_data.append(train_x)
    single_basin_data.append(train_y)
    return single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test


def get_sparse_data(basin_id, date_range):
    train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test = get_limited_data(basin_id, date_range)
    return train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test


def get_limited_data(basin_id, date_range):
    # getHydroData = GetHydroData(basin_id, sequence_length)
    sequence_length = 30
    getHydroData = GetHydroDataWithDate(basin_id, sequence_length, date_range)
    train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()
    ds_test = getHydroData.get_ds_test()
    ds_val = getHydroData.get_ds_val()
    return train_x, train_y, val_x, val_y, ds_val, test_x, test_y, ds_test


if __name__ == '__main__':
    # 02046000 02051500 02053200 02053800 02231000 02212600 -------------------03
    # 12414500 12447390 14137000 14154500  14185900 ---------------------------17
    # 11478500 11480390 11481200 11482500 11522500 11381500 -------------------18
    # 01055000 ----------------------------------------------------------------01
    # 
    basin03 = '02479155'
    basin01 = '01169000' # '01055000'
    basin18 = '11162500'
    basin17 = '13331500'

    the_date_range = {
        'train_date': {
            'start_date': pd.to_datetime("2003-06-15", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2003-07-14", format="%Y-%m-%d")
        },
        'val_date': {
            'start_date' : pd.to_datetime("2003-12-15", format="%Y-%m-%d"),# end - start + 2
            'end_date' : pd.to_datetime("2004-01-14", format="%Y-%m-%d")
        },
        'test_date': {
            'start_date': pd.to_datetime("2004-10-01", format="%Y-%m-%d"),# 没用
            'end_date': pd.to_datetime("2004-10-02", format="%Y-%m-%d")
        },
    }
    
    
    basin_id = basin01
    model_path='/root/best_model_01.pt'
    model =LSTM(input_size = 5,hidden_size = 64,num_layers = 2,output_size = 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    single_basin_data, val_x, val_y, ds_val, test_x, test_y, ds_test = train_test_data(basin_id, the_date_range)
    input_val_x = torch.Tensor(val_x)
    with torch.no_grad():
        model.eval()  # 设置模型为评估模式
        predict = model(input_val_x)
    
    predict_y = predict.numpy()
    predict_y = ds_val.local_rescale(predict_y, variable='output')
    print(predict_y)
    print(val_y)
    nse, rmse, mae, rae = cal_nse_rmse_mae_rae(val_y, predict_y)
    print('basin:', basin_id)
    print('nse:', nse)
    print('rmse:', rmse)
    print('mae:', mae)
    print('rae:', rae)
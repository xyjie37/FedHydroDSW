import numpy as np
from dataset.series_data.utils.camels_operate import CamelsOperate
import sys as sys
import pandas as pd
from utils.constants import path
from pathlib import Path


class GenerateData:
    """产生多个流域数据合并版本数据集"""
    def __init__(self, batch_size, block_size, basin_ids, data_range, time_step, input_dim):
        # self.path = './dataset/series_data/'
        self.path = path
        self.__batch_size = batch_size
        self.__block_size = block_size
        self.__basin_ids = basin_ids
        self.__data_range = data_range
        self.__time_step = time_step
        self.__input_dim = input_dim

    def generate(self):
        """Load hydro 01 from path, and merge 01"""
        time_step = self.__time_step
        train_x_list = []
        train_y_list = []
        test_x_list = []
        test_y_list = []

        for basin_id in self.__basin_ids:
            # 训练数据
            train_start_date = self.__data_range['train_date']['start_date']
            train_end_date = self.__data_range['train_date']['end_date']
            ds_train = CamelsOperate(file_path=self.path, basin=basin_id, seq_length=time_step, period="train",
                                     dates=[train_start_date, train_end_date])
            train_basin_x = ds_train.x
            train_basin_y = ds_train.y

            train_x_list.append(train_basin_x)
            train_y_list.append(train_basin_y)

            # 测试数据
            test_start_date = self.__data_range['test_date']['start_date']
            test_end_date = self.__data_range['test_date']['end_date']
            ds_test = CamelsOperate(file_path=self.path, basin=basin_id, seq_length=time_step, period="eval",
                                    dates=[test_start_date, test_end_date], means=ds_train.get_means(),
                                    stds=ds_train.get_stds())
            test_basin_x = ds_test.x
            test_basin_y = ds_test.y
            test_x_list.append(test_basin_x)
            test_y_list.append(test_basin_y)

        train_len = sys.maxsize
        test_len = sys.maxsize
        for x in train_x_list:
            if len(x) < train_len:
                train_len = len(x)
        for x in test_x_list:
            if len(x) < test_len:
                test_len = len(x)

        train_len = train_len - train_len % self.__block_size  # 去除多余数据
        test_len = test_len - test_len % self.__block_size
        margin1 = train_len // self.__block_size  # //取整
        margin2 = test_len // self.__block_size

        # 去除多余数据
        for i in range(len(train_x_list)):
            train_x_list[i] = train_x_list[i][:train_len]
            train_y_list[i] = train_y_list[i][:train_len]
            test_x_list[i] = test_x_list[i][:test_len]
            test_y_list[i] = test_y_list[i][:test_len]

        # 合并x
        def merge_x(data_list, block_size, margin, data_len, input_dim):
            pos = 0
            result = np.zeros(shape=(data_len * len(data_list), self.__time_step, input_dim))
            basin_num = len(data_list)
            for num in range(1, margin+1):
                cur_pos = pos * basin_num
                for data in data_list:
                    result[cur_pos: cur_pos + block_size, :, :] = data[pos: pos + block_size, :, :]
                    cur_pos = cur_pos + block_size
                pos = block_size * num  # cur_pos是相当于合并前的单个数据集而言，所以是乘block_size
            return result

        # 合并标签
        def merge_y(data_list, block_size, margin, data_len):
            pos = 0
            result = np.zeros(shape=(data_len * len(data_list), 1))
            basin_num = len(data_list)
            for num in range(1, margin+1):
                cur_pos = pos * basin_num
                for data in data_list:
                    result[cur_pos: cur_pos + block_size, :] = data[pos: pos + block_size, :]
                    cur_pos = cur_pos + block_size
                pos = block_size * num  # cur_pos是相当于合并前的单个数据集而言，所以是乘block_size
            return result

        train_x = merge_x(train_x_list, self.__block_size, margin1, train_len, self.__input_dim)
        train_y = merge_y(train_y_list, self.__block_size, margin1, train_len)
        test_x = merge_x(test_x_list, self.__block_size, margin2, test_len, self.__input_dim)
        test_y = merge_y(test_y_list, self.__block_size, margin2, test_len)

        return train_x, train_y, test_x, test_y


def get_basin_ids(unit_code):
    # 获取流域id 列表
    file_path = path
    forcing_path = file_path + '\\forcing_data\\' + str(unit_code)
    forcing_path = Path(forcing_path)
    print("forcing_path:", forcing_path)
    # get path of forcing file
    files = list(forcing_path.glob("**/*_forcing_leap.txt"))  # 这里是在特定的水文单元文件夹下查找
    basin_ids_list = []
    for f in files:
        basin_ids_list.append(f.name[:8])  # 截取文件名的前8位，为流域id，和basin比较

    return basin_ids_list


def run_generate_data(unit_code, the_date_range, basin_ids, node_count, time_step, batch_size,  block_size):
    """此处生成的数据配置中时间步修改，相应的HydroDataSet1处也要修改
          此处的batch_size, block_size修改了，相应的提交者处的也要修改
      """

    # 此处的batch_size, block_size和数据划分有关
    node_count = node_count
    time_step = time_step
    batch_size = batch_size
    block_size = block_size
    hydroDataSet = GenerateData(batch_size=batch_size, block_size=block_size, basin_ids=basin_ids,
                                data_range=the_date_range, time_step=time_step, input_dim=5)
    train_x, train_y, test_x, test_y = hydroDataSet.generate()
    from utils.constants import path
    with open(path + '\\train_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_x.txt', 'w') as outfile:
        for slice in train_x:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\train_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_y.txt', 'w') as outfile:
        for slice in train_y:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\test_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_x.txt', 'w') as outfile:
        for slice in test_x:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\test_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_y.txt', 'w') as outfile:
        for slice in test_y:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')


def run_generate_data2(unit_code, the_date_range, basin_ids, node_count, time_step, batch_size,  block_size):
    """此处生成的数据配置中时间步修改，相应的HydroDataSet1处也要修改
          此处的batch_size, block_size修改了，相应的提交者处的也要修改
      """

    # 此处的batch_size, block_size和数据划分有关
    node_count = node_count
    time_step = time_step
    batch_size = batch_size
    block_size = block_size
    hydroDataSet = GenerateData(batch_size=batch_size, block_size=block_size, basin_ids=basin_ids,
                                data_range=the_date_range, time_step=time_step, input_dim=5)
    train_x, train_y, test_x, test_y = hydroDataSet.generate()
    from utils.constants import path
    with open(path + '\\train_'+str(basin_ids[-1])+'_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_x.txt', 'w') as outfile:
        for slice in train_x:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\train_'+str(basin_ids[-1])+'_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_y.txt', 'w') as outfile:
        for slice in train_y:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\test_'+str(basin_ids[-1])+'_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_x.txt', 'w') as outfile:
        for slice in test_x:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\test_'+str(basin_ids[-1])+'_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_y.txt', 'w') as outfile:
        for slice in test_y:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')


def run_generate_fed_meta_data(unit_code, basin_ids, node_count, time_step, batch_size,  block_size):
    """此处生成的数据配置中时间步修改，相应的HydroDataSet1处也要修改
          此处的batch_size, block_size修改了，相应的提交者处的也要修改
      """

    date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1985-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2000-09-30", format="%Y-%m-%d")
        },

        'test_date': {
            'start_date': pd.to_datetime("2000-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2005-09-30", format="%Y-%m-%d")
        }
    }
    # 此处的batch_size, block_size和数据划分有关
    node_count = node_count
    time_step = time_step
    batch_size = batch_size
    block_size = block_size
    hydroDataSet = GenerateData(batch_size=batch_size, block_size=block_size, basin_ids=basin_ids,
                                data_range=date_range, time_step=time_step, input_dim=5)
    train_x, train_y, test_x, test_y = hydroDataSet.generate()
    from utils.constants import path
    with open(path + '\\meta_data\\meta_train_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_x.txt', 'w') as outfile:
        for slice in train_x:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\meta_data\\meta_train_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_y.txt', 'w') as outfile:
        for slice in train_y:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\meta_data\\meta_test_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_x.txt', 'w') as outfile:
        for slice in test_x:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')

    with open(path + '\\meta_data\\meta_test_unit'+str(unit_code)+'_'+str(node_count)+'basins_t30_y.txt', 'w') as outfile:
        for slice in test_y:
            np.savetxt(outfile, slice, fmt='%f', delimiter=' ')


if __name__ == '__main__':
    # unit_code = 11
    # basin_ids = get_basin_ids(unit_code)
    # node_count = 12
    # time_step = 30
    # batch_size = 256 * node_count
    # block_size = 256
    # basin_ids = basin_ids[:node_count]
    # run_generate_data(unit_code, basin_ids, node_count, time_step, batch_size, block_size)

    """
    水文单元：01
    data for fedsrp
    """
    unit_code = "01"
    # basin_ids = get_basin_ids(unit_code)
    # "01047000"为数据稀缺流域
    # 这里先生成具有相同年份的数据记录
    # 后3个在3个场景中分别作为数据稀缺流域
    # basin_ids = ['01013500', '01030500', '01031500', '01052500', '01057000', '01055000', '01054200', '01047000']
    # basin_ids = ['01013500', '01030500', '01031500', '01052500', '01057000', '01055000', '01047000', '01054200']
    basin_ids = ['01013500', '01030500', '01031500', '01052500', '01057000', '01054200', '01047000', '01055000']
    # basin_ids = basin_ids[:3]
    node_count = len(basin_ids)
    time_step = 30
    batch_size = 256 * node_count
    block_size = 256
    date_range = {
        'train_date': {
            'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2000-09-30", format="%Y-%m-%d")
        },

        'test_date': {
            'start_date': pd.to_datetime("2000-10-01", format="%Y-%m-%d"),
            'end_date': pd.to_datetime("2002-09-30", format="%Y-%m-%d")  # 这里对应的是验证集，
        }
    }
    # # # basin_ids = ['01030500', '01013500', '01031500']
    run_generate_data2(unit_code, date_range, basin_ids, node_count, time_step, batch_size, block_size)

    # """
    #    水文单元：01
    #    """
    # unit_code = "01"
    # # basin_ids = get_basin_ids(unit_code)
    # node_count = 6
    # time_step = 30
    # batch_size = 256 * node_count
    # block_size = 256
    # basin_ids = ['01013500', '01030500', '01031500', '01052500', '01055000', '01057000']
    # date_range = {
    #     'train_date': {
    #         'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    #     },
    #
    #     'test_date': {
    #         'start_date': pd.to_datetime("2005-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("2007-09-30", format="%Y-%m-%d")
    #     }
    # }
    # # # # basin_ids = ['01030500', '01013500', '01031500']
    # run_generate_data(unit_code, date_range, basin_ids, node_count, time_step, batch_size, block_size)

    """
      水文单元：03
      """
    # unit_code = "01"
    # # unit_code = "03"
    # # basin_ids = get_basin_ids(unit_code)
    # node_count = 8
    # time_step = 30
    # batch_size = 256 * node_count
    # block_size = 256
    # basin_ids = get_basin_ids(unit_code)
    # print("basin_ids:", basin_ids[:12])
    # basin_ids = basin_ids[:node_count]
    # print(basin_ids)
    # date_range = {
    #     'train_date': {
    #         'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    #     },
    #
    #     'test_date': {
    #         'start_date': pd.to_datetime("1995-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    #     }
    # }
    # run_generate_data(unit_code, date_range, basin_ids, node_count, time_step, batch_size, block_size)

    # [07056000, 07057500, 07060710, 07066000, 07067000, 07068000]
    """
        水文单元：11
    """
    # unit_code = "11"
    # basin_ids = get_basin_ids(unit_code)
    # node_count = 9
    # time_step = 30
    # batch_size = 256 * node_count
    # block_size = 256
    # basin_ids = basin_ids[:node_count]
    # run_generate_fed_meta_data(unit_code, basin_ids, node_count, time_step, batch_size, block_size)

    """
         水文单元：17
         """
    # unit_code = "17"
    # # basin_ids = get_basin_ids(unit_code)
    # node_count = 12
    # time_step = 30
    # batch_size = 256 * node_count
    # block_size = 256
    # basin_ids = get_basin_ids(unit_code)
    # basin_ids = basin_ids[:node_count]
    # print(basin_ids)
    # run_generate_data(unit_code, basin_ids, node_count, time_step, batch_size, block_size)

    # basin_ids = ['01030500', '01013500', '01022500']
    # basin_ids = ['01030500', '01013500', '01031500']

    # basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000',
    # '01057000', '01073000', '01078000']
    # date_range = {
    #     'train_date': {
    #         'start_date': pd.to_datetime("1980-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("1995-09-30", format="%Y-%m-%d")
    #     },
    #
    #     'test_date': {
    #         'start_date': pd.to_datetime("1995-10-01", format="%Y-%m-%d"),
    #         'end_date': pd.to_datetime("2000-09-30", format="%Y-%m-%d")
    #     }
    # }
    # # 此处的batch_size, block_size和数据划分有关
    # node_count = 12
    # time_step = 30
    # batch_size = 256 * node_count
    # block_size = 256
    # hydroDataSet = GenerateData(batch_size=batch_size, block_size=block_size, basin_ids=basin_ids,
    #                             data_range=date_range, time_step=time_step, input_dim=5)
    # train_x, train_y, test_x, test_y = hydroDataSet.generate()
    # from utils.constants import path
    #
    # with open(path + '\\train_12basins_t30_x.txt', 'w') as outfile:
    #     for slice in train_x:
    #         np.savetxt(outfile, slice, fmt='%f', delimiter=' ')
    #
    # with open(path + '\\train_12basins_t30_y.txt', 'w') as outfile:
    #     for slice in train_y:
    #         np.savetxt(outfile, slice, fmt='%f', delimiter=' ')
    #
    # with open(path + '\\test_12basins_t30_x.txt', 'w') as outfile:
    #     for slice in test_x:
    #         np.savetxt(outfile, slice, fmt='%f', delimiter=' ')
    #
    # with open(path + '\\test_12basins_t30_y.txt', 'w') as outfile:
    #     for slice in test_y:
    #         np.savetxt(outfile, slice, fmt='%f', delimiter=' ')
    # basin_ids = ['01030500', '01013500', '01031500', '01052500', '01054200', '01055000',
    #              '01057000', '01073000', '01078000', '01118300', '01121000', '01123000']

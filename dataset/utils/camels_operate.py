#Imports
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# 现在我们将所有内容包装到PyTorch数据集中。PyTorchs DataLoader类可以使用这些特定类来生成小批量（并在多个线程中并行执行）。
# 这样的数据集类必须从PyTorch数据集类继承，并且必须实现三个函数
# init（）：对象初始化函数。
#
# len（）：此函数必须返回数据集中的样本数。
#
# getitem（i）：返回数据集样本i（样本+目标值）的函数。


class CamelsOperate:
    """Torch Dataset for basic use of 01 from the CAMELS 01 set.
    This 01 set provides meteorological observations and discharge of a given
    basin from the CAMELS 01 set.
    """
    def __init__(self, file_path, basin: str, seq_length: int = 365, period: str = None,
                 dates: List = None, means: pd.Series = None, stds: pd.Series = None):
        """
        数据集类
            Initialize Dataset containing the 01 of a single basin.
        :param basin: 8-digit code of basin as string.

        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.

        :param period: (optional) One of ['train', 'eval']. None loads the
            entire time series.

        :param dates: (optional) List of pd.DateTimes of the start and end date
            of the discharge period that is used.

        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the 01 set.

        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the 01 set.
        """
        self.basin = basin
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds
        self.file_path = Path(file_path)

        # load 01 into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    # 此函数必须返回数据集中的样本数
    def __len__(self):
        return self.num_samples

    # 返回数据集样本i（样本 + 目标值）的函数
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output 01 from text files."""
        df, area = self.load_forcing(self.basin)
        self.area = area
        df['QObs(mm/d)'] = self.load_discharge(self.basin, area)
        print(df)
        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup period.
            print(self.dates[0] - pd.DateOffset(days=self.seq_length))
            # 当前日期减去365，得到开始日期，如果比最早日期早，则开始日期更新为：self.dates[0] - pd.DateOffset(days=self.seq_length)
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            df = df[start_date:self.dates[1]]

        # if training period store means and stds
        # df.mean()
        # 等价于df.mean(0)。把轴向数据求平均，得到每列数据的平均值。
        # df.mean(1)
        # 按照另外一个axis的方向来求平均，得到每行数据的平均值。
        # 如果不是训练阶段，那就用训练阶段的means和stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()

        # extract input and output features from DataFrame
        # 得到(5479, 5)的矩阵，即取了其中的5列关键特征
        x = np.array([df['prcp(mm/day)'].values,
                      df['srad(W/m2)'].values,
                      df['tmax(C)'].values,
                      df['tmin(C)'].values,
                      df['vp(Pa)'].values]).T
        y = np.array([df['QObs(mm/d)'].values]).T

        # normalize 01, reshape for LSTM training and remove invalid samples
        # 执行(x - mean)/std的操作
        x = self._local_normalization(x, variable='inputs')
        x, y = self.reshape_data(x, y, self.seq_length)

        if self.period == "train":
            # Delete all samples, where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print(f"Deleted some records because of NaNs {self.basin}")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # Deletes all records, where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # normalize discharge
            y = self._local_normalization(y, variable='output')
        return x, y

    def get_discharge(self):
        """
            获取流量数据，以及其他类型数据
        """
        df, area = self.load_forcing(self.basin)
        df['QObs(mm/d)'] = self.load_discharge(self.basin, area)
        x = np.array([df['prcp(mm/day)'].values,
                      df['srad(W/m2)'].values,
                      df['tmax(C)'].values,
                      df['tmin(C)'].values,
                      df['vp(Pa)'].values]).T
        y = np.array([df['QObs(mm/d)'].values]).T
        if np.sum(np.isnan(y)) > 0:
            print(f"Deleted some records because of NaNs {self.basin}")
            x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
            y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

        # Deletes all records, where no discharge was measured (-999)
        x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
        y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)
        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])
            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            # 会自动进行广播
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["QObs(mm/d)"]) /
                       self.stds["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> \
            np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['prcp(mm/day)'],
                              self.means['srad(W/m2)'],
                              self.means['tmax(C)'],
                              self.means['tmin(C)'],
                              self.means['vp(Pa)']])

            stds = np.array([self.stds['prcp(mm/day)'],
                             self.stds['srad(W/m2)'],
                             self.stds['tmax(C)'],
                             self.stds['tmin(C)'],
                             self.stds['vp(Pa)']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["QObs(mm/d)"] +
                       self.means["QObs(mm/d)"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def load_forcing(self, basin: str) -> Tuple[pd.DataFrame, int]:
        """
        # 箭头后面表示返回值类型，“：”表示参数数据类型
        Load the meteorological forcing 01 of a specific basin.
        :param basin: 8-digit code of basin as string.
        :return: pd.DataFrame containing the meteorological forcing 01 and the
            area of the basin as integer.
        """
        # directory of meteorological forcings
        forcing_path = self.file_path / 'forcing_data'
        print("forcing_path:", forcing_path)
        # get path of forcing file
        files = list(forcing_path.glob("**/*_forcing_leap.txt"))
        # print(files)
        files = [f for f in files if basin == f.name[:8]]  # 截取文件名的前8位，为流域id，和basin比较
        if len(files) == 0:
            raise RuntimeError(f'No forcing file file found for Basin {basin}')
        else:
            forcing_file_path = files[0]
        # read-in 01 and convert date to datetime index
        with forcing_file_path.open('r') as fp:
            df = pd.read_csv(fp, sep='\s+', header=3)
        dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
                 + df.Day.map(str))
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # load area from header
        with forcing_file_path.open('r') as fp:
            content = fp.readlines()
            area = int(content[2])
        print("返回结果：")
        print("df", df, "area", area)
        return df, area

    def load_discharge(self, basin: str, area: int) -> pd.Series:
        """Load the discharge time series for a specific basin.
        :param basin: 8-digit code of basin as string.
        :param area: int, area of the catchment in square meters
        :return: A pd.Series containng the catchment normalized discharge.
        """
        # directory of the streamflow 01
        discharge_path = self.file_path / 'discharge_data'

        # get path of streamflow file file
        files = list(discharge_path.glob("**/*_streamflow_qc.txt"))
        files = [f for f in files if basin in f.name]
        if len(files) == 0:
            raise RuntimeError(f'No discharge file found for Basin {basin}')
        else:
            discharge_file_path = files[0]

        # read-in 01 and convert date to datetime index
        col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
        with discharge_file_path.open('r') as fp:
            df = pd.read_csv(fp, sep='\s+', header=None, names=col_names)
        dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/"
                 + df.Day.map(str))
        df.index = pd.to_datetime(dates, format="%Y/%m/%d")

        # 将流量的单位转换成 mm/每天
        # normalize discharge from cubic feed per second to mm per day, 86400=3600s*24
        df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10 ** 6)
        print("流量：", df.QObs)
        return df.QObs

    def reshape_discharge(self, Qbs):
        return Qbs * (self.area * 10 ** 6)/(28316846.592 * 86400)

    def reshape_data(self, x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        #    接下来，需要另一个实用函数来将数据重塑为适当的格式，以用于训练LSTM。这些递归神经网络期望形状的顺序输入（序列长度、特征数量）。
        #     我们训练我们的网络，从n天之前的气象观测中预测一天的流量。例如，假设n=365，则单个训练样本应为形状（365，特征数），并且由于
        #     我们使用5个输入特征，因此形状为（365，5）。
        #     但是，从文件加载时，整个数据存储在矩阵中，其中行数对应于训练集中的总天数和要素的列数。因此，我们需要在这个矩阵上滑动，并切出
        #     适合我们的LSTM设置的小样本。为了加快速度，我们在这里使用了非常棒的Numba库（小小的@njit decorator JIT编译了这个函数并显著提高了速度）。

        Reshape matrix 01 into sample shape for LSTM training.

        :param x: Matrix containing input features column wise and time steps row wise
        :param y: Matrix containing the output feature.
        :param seq_length: Length of look back days for one day of prediction

        :return: Two np.ndarrays, the first of shape (samples, length of sequence,
            number of features), containing the input 01 for the LSTM. The second
            of shape (samples, 1) containing the expected output for each input
            sample.
        """
        num_samples, num_features = x.shape
        # x_new.shape:(5114, 365, 5)
        x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))  # 返回指定形状和类型的用0填充的数组
        y_new = np.zeros((num_samples - seq_length + 1, 1))

        for i in range(0, x_new.shape[0]):
            a = x[i:i + seq_length, :]
            b = y[i + seq_length - 1, 0]
            x_new[i, :, :num_features] = x[i:i + seq_length, :]
            y_new[i, :] = y[i + seq_length - 1, 0]

        return x_new, y_new

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds
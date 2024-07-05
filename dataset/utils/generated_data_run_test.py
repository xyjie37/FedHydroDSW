
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset.code_test.utils_test.get_hydro_data import GetHydroData
from utils.constants import path
import nn


def load_data(path, kind):
    """Load hydro 01 from `path`"""
    x_path = os.path.join(path, '%s_12basins_t30_x.txt' % kind)
    y_path = os.path.join(path, '%s_12basins_t30_y.txt' % kind)
    # 注意
    time_step = 30
    x = np.loadtxt(x_path, delimiter=' ').reshape((-1, time_step, 5))
    labels = np.loadtxt(y_path, delimiter=' ').reshape((-1, 1))
    return x, labels


path = path
# Training 01
basin = '01030500'  # can be changed to any 8-digit basin id contained in the CAMELS 01 set
hidden_size = 20  # Number of LSTM cells
dropout_rate = 0  # Dropout rate of the final fully connected Layer [0.0, 1.0]
learning_rate = 0.001  # Learning rate used to update the weights
sequence_length = 30  # Length of the meteorological record provided to the network

getHydroData = GetHydroData(basin, sequence_length)
# train_x, train_y, val_x, val_y, test_x, test_y = getHydroData.get_data()

train_x, train_y = load_data(path, 'train')
test_x, test_y = load_data(path, 'test')

model_path = 'D:\\河海大学\\研究课题\\研究课题\\实验相关\\PSGD\\Parallel-SGD' \
             '\\dataset\\code_test\\MODEL-fed_hydro_3basins-N(0).model'
model = nn.model.SequentialModel.load(model_path)
loss_history = model.fit(train_x, train_y, epoch=10,  batch_size=256)


# 这里的val_x, val_y对应txt中的test_x , test_y
ds_val = getHydroData.get_ds_val()
predict_y = model.predict(test_x)
predict_y = ds_val.local_rescale(predict_y, variable='output')
# predict_y = ds_test.local_rescale(predict_y, variable='output')
test_y = test_y[:500]
predict_y = predict_y[:500]
print(loss_history.history)


epoch_loss = []
batch_num_per_epoch = loss_history.history[0][3]
i = 0
per_epoch_loss = 0
for batch_loss in loss_history.history:
    per_epoch_loss += batch_loss[-1]
    if i != 0 and i % batch_num_per_epoch == 0:
        epoch_loss.append(per_epoch_loss)
        per_epoch_loss = 0
    i = i + 1
print("epoch loss:", epoch_loss)
plt.subplot(2, 1, 1)  # 2行1列，第一张图
plt.plot(list(range(len(predict_y))), predict_y, color='r', label='prediction')
plt.plot(list(range(len(test_y))), test_y, color='b', label='origin')
plt.xlabel('day number')
plt.ylabel('discharge')
plt.subplot(2, 1, 2)
plt.plot(list(range(len(epoch_loss))), epoch_loss, color='orange')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()





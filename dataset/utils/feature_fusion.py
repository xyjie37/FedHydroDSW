import numpy as np


def feature_fusion(x, weights_list):
    """(x, 30, 5)----> (x, 30, 1)"""
    d0, d1, d2 = x.shape
    weights_list = np.reshape(np.asarray(weights_list), newshape=(5, 1))
    res = np.zeros(shape=(d0, d1, 1))
    i = 0
    for sample in x:
        sample = np.reshape(sample, newshape=(d1, d2))
        time_array = np.zeros(shape=(d1, 1))
        j = 0
        for pre_time in sample:
            """(30, 5)  ---> (30, 1)"""
            # a = pre_time[0]*weights_list[0]
            # b = pre_time[1]*weights_list[1]
            # c = pre_time[2]*weights_list[2]
            # d = pre_time[3]*weights_list[3]
            # e = pre_time[4]*weights_list[4]
            new_val = np.dot(pre_time, weights_list)
            time_array[j] = new_val
            j += 1
        res[i, :, :] = time_array
        i += 1

    return res



import numpy as np
from typing import List
from sklearn.preprocessing import OneHotEncoder



def cal_quota(real_y, pred_y, negative, neutral, positive, log=False):
    """
    需要将 real_y 和 pred_y 数据先转换为向量，然后指定其 negative,neutral,positive 所对应的值
    :return:
    """
    # negative[TP, FP, TN, FN, P, R, F], neutral[...], positive[...], overall[...]
    quotas = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]
    for pred, real in zip(pred_y, real_y):
        for i, val in enumerate((negative, neutral, positive)):
            if pred == val and real == val:
                quotas[i][0] += 1
                quotas[3][0] += 1
            if pred == val and real != val:
                quotas[i][1] += 1
                quotas[3][1] += 1
            if pred != val and real != val:
                quotas[i][2] += 1
                quotas[3][2] += 1
            if pred != val and real == val:
                quotas[i][3] += 1
                quotas[3][3] += 1
    for i in range(4):
        try:
            quotas[i][4] = p = quotas[i][0] / (quotas[i][0] + quotas[i][1])
            quotas[i][5] = r = quotas[i][0] / (quotas[i][0] + quotas[i][3])
            quotas[i][6] = (2 * p * r) / (p + r)
        except ZeroDivisionError:
            pass
    if log:
        print('\t\t\tP\t\tR\t\tF')
        print(f'overall:\t{quotas[3][4]:.5f}\t{quotas[3][5]:.5f}\t{quotas[3][6]:.5f}')
        print(f'positive:\t{quotas[2][4]:.5f}\t{quotas[2][5]:.5f}\t{quotas[2][6]:.5f}')
        print(f'neutral:\t{quotas[1][4]:.5f}\t{quotas[1][5]:.5f}\t{quotas[1][6]:.5f}')
        print(f'negative:\t{quotas[0][4]:.5f}\t{quotas[0][5]:.5f}\t{quotas[0][6]:.5f}')
    return quotas


def pad_sequences(seq: List[np.ndarray], padding=0, padding_method='same'):
    """
    :param seq: 数据
    :param padding: 是否将 x 数据填充至指定长度，0表示不填充
    :param padding_method: 填充方式，有 same, zero, one 三种方式，填充是指在数据的开始和结尾填充指定数据至指定长度
    """
    padded_seq = []
    for index in range(len(seq)):
        t = seq[index]
        pad_l = padding - t.shape[0]
        if pad_l > 0:
            pad_t = np.array([t[-1]])
            if padding_method == 'one':
                pad_t = np.ones(pad_t.shape, pad_t.dtype)
            elif padding_method == 'zero':
                pad_t = np.zeros(pad_t.shape, pad_t.dtype)
            pad_t = np.concatenate([pad_t for _ in range(pad_l)])
            t = np.concatenate((t, pad_t))
        padded_seq.append(t)
    return padded_seq


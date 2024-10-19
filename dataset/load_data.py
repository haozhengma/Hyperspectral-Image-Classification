#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split


SAMPLE_RADIUS = 10


def load_data_split01(data_dir):
    _HSI_DATA = 'data_hsi_01'
    _SAR_DATA = 'data_msi_01'
    _TRAIN_LABELS = 'train_label_01'

    data_hsi = scio.loadmat(os.path.join(data_dir, _HSI_DATA + '.mat'))['data']
    data_msi = scio.loadmat(os.path.join(data_dir, _SAR_DATA + '.mat'))['data']
    y_train = scio.loadmat(os.path.join(data_dir, _TRAIN_LABELS + '.mat'))['train_label']

    assert data_hsi.shape[0] == data_msi.shape[0] == y_train.shape[0], 'Dimension of data arrays does not match'
    assert data_hsi.shape[1] == data_msi.shape[1] == y_train.shape[1], 'Dimension of data arrays does not match'

    rows, cols = y_train.shape
    X_data_hsi = np.zeros((rows + SAMPLE_RADIUS * 2, cols + SAMPLE_RADIUS * 2, data_hsi.shape[2]))  # zero padding
    X_data_msi = np.zeros((rows + SAMPLE_RADIUS * 2, cols + SAMPLE_RADIUS * 2, data_msi.shape[2]))

    X_data_hsi[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = data_hsi
    X_data_msi[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = data_msi

    X_data_hsi = np.transpose(X_data_hsi, (2, 0, 1))
    X_data_msi = np.transpose(X_data_msi, (2, 0, 1))

    # channel-wise normalization
    # for b in range(X_data_hsi.shape[0]):
    #     band = X_data_hsi[b, ...]
    #     X_data_hsi[b, ...] = (band-np.min(band)) / (np.max(band)-np.min(band))
    # for b in range(X_data_sar.shape[0]):
    #     band = X_data_sar[b, ...]
    #     X_data_sar[b, ...] = (band-np.min(band)) / (np.max(band)-np.min(band))

    # 训练集
    X_hsi = []
    X_msi = []
    y = []

    # # 验证集
    # X_hsi_te = []
    # X_msi_te = []
    # y_te = []

    for r in range(SAMPLE_RADIUS, rows + SAMPLE_RADIUS):
        for c in range(SAMPLE_RADIUS, cols + SAMPLE_RADIUS):
            sample_hsi = X_data_hsi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            sample_sar = X_data_msi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            label_tr = y_train[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            if label_tr > 0:
                X_hsi.append(sample_hsi)
                X_msi.append(sample_sar)
                y.append(label_tr)


    # 使用train_test_split根据c来拆分a和b
    X_hsi_tr, X_hsi_te, X_msi_tr, X_msi_te, y_tr, y_te = train_test_split(
        X_hsi, X_msi, y, test_size=0.25, train_size=0.75, stratify=y, random_state=42)

    # # 打印结果
    # print("X_hsi_tr:", X_hsi_tr)
    # print("X_hsi_te:", X_hsi_te)
    # print("X_msi_tr:", X_msi_tr)
    # print("X_msi_te:", X_msi_te)
    # print("y_tr:", y_tr)
    # print("y_te:", y_te)

    return np.array(X_hsi_tr), np.array(X_msi_tr), np.array(y_tr) - 1, \
           np.array(X_hsi_te), np.array(X_msi_te), np.array(y_te) - 1


def load_data_split02(data_dir):
    _HSI_DATA = 'data_hsi_02'
    _SAR_DATA = 'data_msi_02'
    _TRAIN_LABELS = 'train_label_02'

    data_hsi = scio.loadmat(os.path.join(data_dir, _HSI_DATA + '.mat'))['data']
    data_msi = scio.loadmat(os.path.join(data_dir, _SAR_DATA + '.mat'))['data']
    data_hsi = np.transpose(data_hsi, (1, 2, 0))
    data_msi = np.transpose(data_msi, (1, 2, 0))
    y_train = scio.loadmat(os.path.join(data_dir, _TRAIN_LABELS + '.mat'))['train_label']

    assert data_hsi.shape[0] == data_msi.shape[0] == y_train.shape[0], 'Dimension of data arrays does not match'
    assert data_hsi.shape[1] == data_msi.shape[1] == y_train.shape[1], 'Dimension of data arrays does not match'

    rows, cols = y_train.shape
    X_data_hsi = np.zeros((rows + SAMPLE_RADIUS * 2, cols + SAMPLE_RADIUS * 2, data_hsi.shape[2]))  # zero padding
    X_data_msi = np.zeros((rows + SAMPLE_RADIUS * 2, cols + SAMPLE_RADIUS * 2, data_msi.shape[2]))

    X_data_hsi[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = data_hsi
    X_data_msi[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = data_msi

    X_data_hsi = np.transpose(X_data_hsi, (2, 0, 1))
    X_data_msi = np.transpose(X_data_msi, (2, 0, 1))

    # channel-wise normalization
    # for b in range(X_data_hsi.shape[0]):
    #     band = X_data_hsi[b, ...]
    #     X_data_hsi[b, ...] = (band-np.min(band)) / (np.max(band)-np.min(band))
    # for b in range(X_data_sar.shape[0]):
    #     band = X_data_sar[b, ...]
    #     X_data_sar[b, ...] = (band-np.min(band)) / (np.max(band)-np.min(band))

    # 训练集
    X_hsi = []
    X_msi = []
    y = []

    # # 验证集
    # X_hsi_te = []
    # X_msi_te = []
    # y_te = []

    for r in range(SAMPLE_RADIUS, rows + SAMPLE_RADIUS):
        for c in range(SAMPLE_RADIUS, cols + SAMPLE_RADIUS):
            sample_hsi = X_data_hsi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            sample_sar = X_data_msi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            label_tr = y_train[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            if label_tr > 0:
                X_hsi.append(sample_hsi)
                X_msi.append(sample_sar)
                y.append(label_tr)


    # 使用train_test_split根据c来拆分a和b
    X_hsi_tr, X_hsi_te, X_msi_tr, X_msi_te, y_tr, y_te = train_test_split(
        X_hsi, X_msi, y, test_size=0.25, train_size=0.75, stratify=y, random_state=42)

    # # 打印结果
    # print("X_hsi_tr:", X_hsi_tr)
    # print("X_hsi_te:", X_hsi_te)
    # print("X_msi_tr:", X_msi_tr)
    # print("X_msi_te:", X_msi_te)
    # print("y_tr:", y_tr)
    # print("y_te:", y_te)

    return np.array(X_hsi_tr), np.array(X_msi_tr), np.array(y_tr) - 1, \
           np.array(X_hsi_te), np.array(X_msi_te), np.array(y_te) - 1


def load_data(data_dir):
    _HSI_DATA = 'data_hsi'
    _SAR_DATA = 'data_msi'
    _TRAIN_LABELS = 'train_label'
    _TEST_LABELS = 'test_label'

    data_hsi = scio.loadmat(os.path.join(data_dir, _HSI_DATA + '.mat'))['data']
    data_msi = scio.loadmat(os.path.join(data_dir, _SAR_DATA + '.mat'))['data']
    y_train = scio.loadmat(os.path.join(data_dir, _TRAIN_LABELS + '.mat'))[_TRAIN_LABELS]
    y_test = scio.loadmat(os.path.join(data_dir, _TEST_LABELS + '.mat'))[_TEST_LABELS]
    assert data_hsi.shape[0] == data_msi.shape[0] == y_train.shape[0] == y_test.shape[
        0], 'Dimension of data arrays does not match'
    assert data_hsi.shape[1] == data_msi.shape[1] == y_train.shape[1] == y_test.shape[
        1], 'Dimension of data arrays does not match'

    rows, cols = y_train.shape
    X_data_hsi = np.zeros((rows + SAMPLE_RADIUS * 2, cols + SAMPLE_RADIUS * 2, data_hsi.shape[2]))  # zero padding
    X_data_msi = np.zeros((rows + SAMPLE_RADIUS * 2, cols + SAMPLE_RADIUS * 2, data_msi.shape[2]))

    X_data_hsi[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = data_hsi
    X_data_msi[SAMPLE_RADIUS:-SAMPLE_RADIUS, SAMPLE_RADIUS:-SAMPLE_RADIUS, :] = data_msi

    X_data_hsi = np.transpose(X_data_hsi, (2, 0, 1))
    X_data_msi = np.transpose(X_data_msi, (2, 0, 1))

    # channel-wise normalization
    # for b in range(X_data_hsi.shape[0]):
    #     band = X_data_hsi[b, ...]
    #     X_data_hsi[b, ...] = (band-np.min(band)) / (np.max(band)-np.min(band))
    # for b in range(X_data_sar.shape[0]):
    #     band = X_data_sar[b, ...]
    #     X_data_sar[b, ...] = (band-np.min(band)) / (np.max(band)-np.min(band))

    # 训练集
    X_hsi_tr = []
    X_msi_tr = []
    y_tr = []

    # 验证集
    X_hsi_te = []
    X_msi_te = []
    y_te = []

    for r in range(SAMPLE_RADIUS, rows + SAMPLE_RADIUS):
        for c in range(SAMPLE_RADIUS, cols + SAMPLE_RADIUS):
            sample_hsi = X_data_hsi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            sample_sar = X_data_msi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            label_tr = y_train[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            label_te = y_test[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            if label_tr > 0:
                X_hsi_tr.append(sample_hsi)
                X_msi_tr.append(sample_sar)
                y_tr.append(label_tr)
            elif label_te > 0:
                X_hsi_te.append(sample_hsi)
                X_msi_te.append(sample_sar)
                y_te.append(label_te)

    return np.array(X_hsi_tr), np.array(X_msi_tr), np.array(y_tr) - 1, \
           np.array(X_hsi_te), np.array(X_msi_te), np.array(y_te) - 1


if __name__ == "__main__":
    a1, a2, a3, a4, a5, a6 = load_data_split02(r'I:\demo\pytorch\01-classification\yancheng\test02')
    print(a1.shape)
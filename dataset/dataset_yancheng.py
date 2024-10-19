#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import scipy.io as scio


SAMPLE_RADIUS = 5


def load_dataset(data_dir):
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
    X_sar_tr = []
    y_tr = []

    # # 验证集
    # X_hsi_te = []
    # X_sar_te = []
    # y_te = []

    for r in range(SAMPLE_RADIUS, rows + SAMPLE_RADIUS):
        for c in range(SAMPLE_RADIUS, cols + SAMPLE_RADIUS):
            sample_hsi = X_data_hsi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            sample_sar = X_data_msi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            label_tr = y_train[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            # label_te = y_test[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            if label_tr > 0:
                X_hsi_tr.append(sample_hsi)
                X_sar_tr.append(sample_sar)
                y_tr.append(label_tr)

    return np.array(X_hsi_tr), np.array(X_sar_tr), np.array(y_tr) - 1, X_data_hsi, X_data_msi, y_test


def load_test01(data_dir):
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
    X_hsi_tr = []
    X_sar_tr = []
    y_tr = []

    # # 验证集
    # X_hsi_te = []
    # X_sar_te = []
    # y_te = []

    for r in range(SAMPLE_RADIUS, rows + SAMPLE_RADIUS):
        for c in range(SAMPLE_RADIUS, cols + SAMPLE_RADIUS):
            sample_hsi = X_data_hsi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            sample_sar = X_data_msi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            label_tr = y_train[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            # label_te = y_test[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            if label_tr > 0:
                X_hsi_tr.append(sample_hsi)
                X_sar_tr.append(sample_sar)
                y_tr.append(label_tr)

    return np.array(X_hsi_tr), np.array(X_sar_tr), np.array(y_tr) - 1, X_data_hsi, X_data_msi


def load_test02(data_dir):
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
    X_hsi_tr = []
    X_sar_tr = []
    y_tr = []

    # # 验证集
    # X_hsi_te = []
    # X_sar_te = []
    # y_te = []

    for r in range(SAMPLE_RADIUS, rows + SAMPLE_RADIUS):
        for c in range(SAMPLE_RADIUS, cols + SAMPLE_RADIUS):
            sample_hsi = X_data_hsi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            sample_sar = X_data_msi[:, r - SAMPLE_RADIUS:r + SAMPLE_RADIUS + 1, c - SAMPLE_RADIUS:c + SAMPLE_RADIUS + 1]
            label_tr = y_train[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            # label_te = y_test[r - SAMPLE_RADIUS, c - SAMPLE_RADIUS]
            if label_tr > 0:
                X_hsi_tr.append(sample_hsi)
                X_sar_tr.append(sample_sar)
                y_tr.append(label_tr)

    return np.array(X_hsi_tr), np.array(X_sar_tr), np.array(y_tr) - 1, X_data_hsi, X_data_msi


if __name__ == "__main__":
    a1, a2, a3, a4, a5, a6 = load_dataset(r'dataset/data_msi.mat')
    print(a1.shape)
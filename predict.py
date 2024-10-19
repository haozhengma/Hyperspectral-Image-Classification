import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
from torch import nn, optim
from torch.utils import data
import argparse
from osgeo import gdal

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset.dataset_yancheng import load_dataset, load_test01


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     # np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(10)  # 100


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    _, _, _, X_data_HSI, X_data_SAR = load_test01(r'I:\demo\pytorch\01-classification\yancheng\test01')

    model = torch.load(r'I:\demo\pytorch\01-classification\yancheng\test01\CASA_seed0_r10_4v1.pth')

    print(model)
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)

    data_rows = X_data_HSI.shape[1]
    data_cols = X_data_HSI.shape[2]
    model.eval()
    pred_all = []
    x_batch_hsi = []
    x_batch_sar = []
    batch_count = 0
    sample_radius = 10

    # 遍历整张图
    for r in range(sample_radius, data_rows - sample_radius):
        for c in range(sample_radius, data_cols - sample_radius):
            x_hsi_test = X_data_HSI[:, r-sample_radius:r+sample_radius+1, c-sample_radius:c+sample_radius+1]
            x_sar_test = X_data_SAR[:, r-sample_radius:r+sample_radius+1, c-sample_radius:c+sample_radius+1]
            x_batch_hsi.append(x_hsi_test)
            x_batch_sar.append(x_sar_test)
            batch_count += 1
            if batch_count == args.batch_size:
                x_batch_hsi = torch.Tensor(np.array(x_batch_hsi))
                x_batch_hsi = x_batch_hsi.to(device)
                x_batch_sar = torch.Tensor(np.array(x_batch_sar))
                x_batch_sar = x_batch_sar.to(device)
                outputs = model(x_batch_hsi, x_batch_sar)
                _, pred = torch.max(outputs[1].data, 1)
                pred += 1
                pred_all.append(pred.cpu().numpy())
                batch_count = 0
                x_batch_hsi = []
                x_batch_sar = []

    if batch_count > 0:
        x_batch_hsi = torch.Tensor(np.array(x_batch_hsi))
        x_batch_sar = torch.Tensor(np.array(x_batch_sar))
        x_batch_hsi = x_batch_hsi.to(device)
        x_batch_sar = x_batch_sar.to(device)
        outputs = model(x_batch_hsi, x_batch_sar)
        _, pred = torch.max(outputs[1].data, 1)
        pred += 1
        pred_all.append(pred.cpu().numpy())

    pred_all_fla = np.hstack(np.array(pred_all, dtype=object).flatten())
    pred_map = pred_all_fla.reshape(X_data_HSI.shape[1] - 2 * sample_radius, X_data_HSI.shape[2] - 2 * sample_radius)
    print(pred_map)

    driver = gdal.GetDriverByName("GTiff")
    output_map = driver.Create(r'I:\demo\pytorch\01-classification\yancheng\test01\CASA_seed0_r10_4v1.tif', 1342, 1185, 1, gdal.GDT_Byte)

    band = output_map.GetRasterBand(1)
    band.WriteArray(pred_map)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=18, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--model_path", default='weights', type=str)
    args = parser.parse_args()
    main(args)

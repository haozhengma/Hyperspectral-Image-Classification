import numpy as np
import rasterio
from scipy.io import loadmat
from sklearn.metrics import recall_score, precision_score, accuracy_score, cohen_kappa_score


# 读取TIF文件
with rasterio.open('maps/S2FL.tif') as src:
    tif_data = src.read(1)  # 读取第一个波段

# 读取MAT文件
mat_data = loadmat('dataset/test_label.mat')
mat_data = mat_data['test_label']  # 假设MAT文件中的数据存储在名为 'data' 的字段里

# 确保两个图像尺寸一致
assert tif_data.shape == mat_data.shape, "TIF和MAT数据尺寸不一致"

# 只对mat文件中大于0的像素进行过滤
valid_mask = mat_data > 0
tif_valid = tif_data[valid_mask]
mat_valid = mat_data[valid_mask]

# 召回率（按类别计算）
recall = recall_score(mat_valid, tif_valid, labels=np.arange(1, 19), average=None)

# 平均精度（按类别计算）
average_accuracy = sum(recall) / len(recall)

# 整体准确率（OA）
overall_accuracy = accuracy_score(mat_valid, tif_valid)

# Kappa系数
kappa = cohen_kappa_score(mat_valid, tif_valid, labels=np.arange(1, 19))

# 输出结果
print(f"召回率: {recall}")
print(f"平均准确率 AA: {average_accuracy}")
print(f"整体准确率 OA: {overall_accuracy}")
print(f"Kappa系数: {kappa}")
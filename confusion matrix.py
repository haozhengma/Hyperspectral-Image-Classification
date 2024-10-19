import os
import json

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dataset.load_data import load_data


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list, title: str):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.title = title

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    # 根据混淆矩阵进行计算
    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        po_numerator = list()
        pe_numerator = list()
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "Accuracy"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            Accuracy = round((TP + TN) / (TP + FP + TN + FN), 3) if TP + FP + TN + FN != 0 else 0.

            po_numerator.append(TP)
            pe_numerator.append((np.sum(self.matrix[i, :]) * np.sum(self.matrix[:, i])))

            table.add_row([self.labels[i], Precision, Recall, Specificity, Accuracy])

        # 计算Kappa的公式：kappa = (po-pe)/(1-pe)
        po = sum(po_numerator) / np.sum(self.matrix)
        pe = sum(pe_numerator) / ((np.sum(self.matrix)) * (np.sum(self.matrix)))
        kappa = (po - pe) / (1 - pe)

        print("the model kappa is ", kappa)
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix {}'.format(self.title))

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    TrainPatch_HSI, TrainPatch_SAR, Y_train, TestPatch_HSI, TestPatch_SAR, Y_test = load_data(r'dataset')
    # 训练集
    TrainPatch_HSI = torch.from_numpy(TrainPatch_HSI).float()
    TrainPatch_SAR = torch.from_numpy(TrainPatch_SAR).float()
    Y_train = torch.from_numpy(Y_train).long()

    TestPatch_HSI = torch.from_numpy(TestPatch_HSI).float()
    TestPatch_SAR = torch.from_numpy(TestPatch_SAR).float()
    Y_test = torch.from_numpy(Y_test).long()

    batch_size = 32
    train_dataset = TensorDataset(TrainPatch_HSI, TrainPatch_SAR, Y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(TestPatch_HSI, TestPatch_SAR, Y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    net = torch.load(r"I:\demo\pytorch\01-classification\yancheng\weights\CASANet_0.pth")
    net.to(device)

    # read class_indict
    json_label_path = './class_indices_num.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=18, labels=labels, title='')
    net.eval()
    with torch.no_grad():

        for hsi, msi, labels in test_loader:
            hsi = hsi.to(device)
            msi = msi.to(device)
            labels = labels.to(device)
            outputs = net(hsi, msi)

            outputs = torch.softmax(outputs[1], dim=1)
            outputs = torch.argmax(outputs, dim=1)

            confusion.update(outputs.to("cpu").numpy(), labels.to("cpu").numpy())

    confusion.plot()
    confusion.summary()

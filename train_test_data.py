import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
from torch import nn, optim
from torch.utils import data
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset

from models2.CASANet_light import CASANet_test01
from models.MFT import MFT
from models.AsyFFNet import Net, Bottleneck
from models.TwoBranchCNN import TwoBranchCNN
from models.DFINet import Net

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset.load_data import load_data_split01, load_data_split02
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     # np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(0)  # 100


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def main(args):
    # Create model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    yancheng_path = r'test01'
    TrainPatch_HSI, TrainPatch_MSI, Y_train, TestPatch_HSI, TestPatch_MSI, Y_test = load_data_split01(yancheng_path)
    # 训练集
    TrainPatch_HSI=torch.from_numpy(TrainPatch_HSI).float()
    TrainPatch_MSI=torch.from_numpy(TrainPatch_MSI).float()
    Y_train = torch.from_numpy(Y_train).long()

    TestPatch_HSI=torch.from_numpy(TestPatch_HSI).float()
    TestPatch_MSI=torch.from_numpy(TestPatch_MSI).float()
    Y_test = torch.from_numpy(Y_test).long()

    train_dataset = TensorDataset(TrainPatch_HSI, TrainPatch_MSI, Y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = TensorDataset(TestPatch_HSI, TestPatch_MSI, Y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    model = CASANet_test01(args.num_class)  # CASA
    # model = Net(253, 4, 128, Bottleneck, 2, num_classes=args.num_class)
    # model = MFT(16, 253, 4, args.num_class, False)  # MFT
    # model = TwoBranchCNN(num_classes=args.num_class)
    # model = Net(253, 4, args.num_class)

    # print(model)
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)
    cost = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    oa = 0
    best_acc = 0.
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        # start time
        start = time.time()
        index = 0

        for hsi, msi, labels in train_loader:
            hsi = hsi.to(device)
            msi = msi.to(device)
            labels = labels.to(device).squeeze()

            # Forward pass
            img_feature, outputs = model(hsi, msi)
            loss = cost(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            index += 1

        if epoch % 1 == 0:
            end = time.time()
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss.item(), (end-start) * 2))
            test_start = time.time()
            model.eval()
            correct_prediction = 0.
            pred_all = []
            true_all = []
            with torch.no_grad():
                for hsi, msi, labels in test_loader:
                    hsi = hsi.to(device)
                    msi = msi.to(device)
                    labels = labels.to(device).squeeze()

                    outputs = model(hsi, msi)
                    loss_test = cost(outputs[1], labels)
                    _, pred = torch.max(outputs[1].data, 1)
                    correct_prediction += (pred == labels).sum().item()
                    true_all.extend(labels.cpu().numpy())
                    pred_all.extend(pred.cpu().numpy())

            test_end = time.time()
            print("Epoch [%d/%d], Loss: %.8f, Time: %.1fsec!" % (epoch, args.epochs, loss_test.item(), (test_end - test_start) * 2))

            cm = confusion_matrix(true_all, pred_all)
            accuracy_for_each_class = recall_score(true_all, pred_all, average=None)  # User Accuracy
            oa = np.sum(cm*np.eye(args.num_class, args.num_class)) / np.sum(cm)
            aa = recall_score(true_all, pred_all, average='macro')
            kappa = cohen_kappa_score(true_all, pred_all)
            print('correct_prediction', correct_prediction)
            print("oa is ", oa, 'the former best oa is ', best_acc)
            print("aa is ", aa)

            print("accuracy for each class:", accuracy_for_each_class)
            print("kappa is ", kappa)

            print('----------------------------------------------------')

        if oa >= best_acc:
            print('save new best oa', oa)
            torch.save(model, os.path.join(args.model_path, '{}_seed0_r10_4v1.pth'.format(args.model_name)))
            best_acc = oa
            best_epoch = epoch

    print('The best oa', best_acc, best_epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=18, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--turn", default=0, type=int)
    parser.add_argument("--model_name", default='CASA', type=str)
    parser.add_argument("--model_path", default='test01', type=str)
    parser.add_argument("--pretrained", default=False, type=bool)
    parser.add_argument("--pretrained_model", default='', type=str)
    args = parser.parse_args()

    main(args)

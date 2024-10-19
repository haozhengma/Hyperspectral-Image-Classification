import torch
from torch import nn
# from models.MLPMixer_CASANet import CASA_Block
# from models.resnext import resnext50_253, resnext50_4,
from models2.MLPMixer_CASANet_light import CASA_Block
from models2.resnext import resnext50_253, resnext50_4, resnext50_285, resnext50_330


class CASANet(nn.Module):
    def __init__(self, n_class):
        super(CASANet, self).__init__()
        self.n_class = n_class

        self.resnext50_253 = resnext50_253()

        self.resnext50_4 = resnext50_4()

        self.mixer = CASA_Block(num_classes=18,
                                image_size=11,
                                channels=64,  # 128-->64
                                patch_size=3,
                                tokens_hidden_dim=32,  # 64-->32
                                channels_hidden_dim=256)  # 512-->256

    def forward(self, hsi, msi):
        hsi = self.resnext50_253(hsi)
        msi = self.resnext50_4(msi)
        patch_mixer = self.mixer(hsi, msi)
        return patch_mixer, patch_mixer


class CASANet_test01(nn.Module):
    def __init__(self, n_class):
        super(CASANet_test01, self).__init__()
        self.n_class = n_class

        self.resnext50_285 = resnext50_285()

        self.resnext50_4 = resnext50_4()

        self.mixer = CASA_Block(num_classes=18,
                                image_size=21,  # 11, 21
                                channels=64,  # 128-->64
                                patch_size=7,  # 3, 7
                                tokens_hidden_dim=32,  # 64-->32
                                channels_hidden_dim=256)  # 512-->256

    def forward(self, hsi, msi):
        hsi = self.resnext50_285(hsi)
        msi = self.resnext50_4(msi)
        patch_mixer = self.mixer(hsi, msi)
        return patch_mixer, patch_mixer


class CASANet_test02(nn.Module):
    def __init__(self, n_class):
        super(CASANet_test02, self).__init__()
        self.n_class = n_class

        self.resnext50_330 = resnext50_330()

        self.resnext50_4 = resnext50_4()

        self.mixer = CASA_Block(num_classes=18,
                                image_size=11,  # 11 or 21
                                channels=64,  # 128-->64
                                patch_size=3,  # 3 or 7
                                tokens_hidden_dim=32,  # 64-->32
                                channels_hidden_dim=256)  # 512-->256

    def forward(self, hsi, msi):
        hsi = self.resnext50_330(hsi)
        msi = self.resnext50_4(msi)
        patch_mixer = self.mixer(hsi, msi)
        return patch_mixer, patch_mixer


if __name__ == '__main__':

    x1 = torch.randn(32, 253, 11, 11)
    x2 = torch.randn(32, 4, 11, 11)

    model = CASANet(18)
    # print(model)
    print('model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    output = model(x1, x2)
    # print(output)


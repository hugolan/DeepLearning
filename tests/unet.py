import torch
import torch.nn as nn


class Conv2dBatch(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop_conv2d=0.005):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop_conv2d)
        )

    def forward(self, x):
        x = self.convolution(x)
        return x


class Encode(nn.Module):
    # 1 contraction step in the network
    def __init__(self, in_ch, out_ch, p_drop_conv2d=0.005):
        super().__init__()
        self.contraction = nn.MaxPool2d(2, stride=2)
        self.convolution = Conv2dBatch(in_ch, out_ch, p_drop_conv2d)

    def forward(self, x):
        x = self.contraction(x)
        x = self.convolution(x)
        return x


class Decode(nn.Module):
    # 1 expansion step in the network
    def __init__(self, in_ch, out_ch, p_drop_decode=0.5, p_drop_conv2d=0.005):
        super().__init__()
        self.up_convolution = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), # Notice a kernel size of 3x3 instead of 2x2 as specified in the paper. This is to promote implementation  simplicity
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_drop_decode)
        )
        self.convolution = Conv2dBatch(in_ch, out_ch, p_drop_conv2d=p_drop_conv2d)

    def forward(self, x, xi):
        x = self.up_convolution(x)
        x = torch.cat((xi, x), dim=1)
        x = self.convolution(x)
        return x


# From https://arxiv.org/abs/1505.04597
class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop_decode=0.5, p_drop_conv2d=0.005):
        super(UNet, self).__init__()

        self.InConvBatch = Conv2dBatch(in_ch, 64, p_drop_conv2d=p_drop_conv2d)

        self.Contract1 = Encode(64, 128, p_drop_conv2d=p_drop_conv2d)
        self.Contract2 = Encode(128, 256, p_drop_conv2d=p_drop_conv2d)
        self.Contract3 = Encode(256, 512, p_drop_conv2d=p_drop_conv2d)
        self.Contract4 = Encode(512, 1024, p_drop_conv2d=p_drop_conv2d)
        self.Contract5 = Encode(1024, 2048, p_drop_conv2d=p_drop_conv2d)

        self.Expand5 = Decode(2048, 1024, p_drop_decode=p_drop_decode, p_drop_conv2d=p_drop_conv2d)
        self.Expand4 = Decode(1024, 512, p_drop_decode=p_drop_decode, p_drop_conv2d=p_drop_conv2d)
        self.Expand3 = Decode(512, 256, p_drop_decode=p_drop_decode, p_drop_conv2d=p_drop_conv2d)
        self.Expand2 = Decode(256, 128, p_drop_decode=p_drop_decode, p_drop_conv2d=p_drop_conv2d)
        self.Expand1 = Decode(128, 64, p_drop_decode=p_drop_decode, p_drop_conv2d=p_drop_conv2d)

        self.OutConv = nn.Sequential(
            nn.Conv2d(64, out_ch, 1),  # 1x1 convolution to transfer back to out channel
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.InConvBatch(x)

        x2 = self.Contract1(x1)
        x3 = self.Contract2(x2)
        x4 = self.Contract3(x3)
        x5 = self.Contract4(x4)
        x = self.Contract5(x5)

        x = self.Expand5(x, x5)
        x = self.Expand4(x, x4)
        x = self.Expand3(x, x3)
        x = self.Expand2(x, x2)
        x = self.Expand1(x, x1)

        x = self.OutConv(x)
        return x

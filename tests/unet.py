import torch
import torch.nn as nn

class Conv2dBatch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )

    def forward(self, x):
        x = self.convolution(x)
        return x

class Contract(nn.Module):
    # 1 contraction step in the network
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.contraction = nn.MaxPool2d(2, stride=2)
        self.convolution = Conv2dBatch(in_ch, out_ch)

    def forward(self, x):
        x = self.contraction(x)
        x = self.convolution(x)
        return x

    
class Expand(nn.Module):
    # 1 expansion step in the network
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up_convolution = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, padding=1), # Notice a kernel size of 3x3 instead of 2x2 as specified in the paper. This is to promote implementation  simplicity
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2)
        )
        self.convolution = Conv2dBatch(in_ch, out_ch)

    def forward(self, x, xi):
        x = self.up_convolution(x)
        x = torch.cat((xi, x), dim=1)
        x = self.convolution(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        
        self.InConvBatch = Conv2dBatch(in_ch, 64)

        self.Contract1 = Contract(64, 128)
        self.Contract2 = Contract(128, 256)
        self.Contract3 = Contract(256, 512)
        self.Contract4 = Contract(512, 1024)
        self.Contract5 = Contract(1024, 2048)
        #self.Contract6 = Contract(2048, 4096)
            
        #self.Expand6 = Expand(4096, 2048)
        self.Expand5 = Expand(2048, 1024)
        self.Expand4 = Expand(1024, 512)
        self.Expand3 = Expand(512, 256)
        self.Expand2 = Expand(256, 128)
        self.Expand1 = Expand(128, 64)

        self.OutConv = nn.Conv2d(64, out_ch, 1) # 1x1 convolution to transfer back to 1 channel

    def forward(self, x):
        x1 = self.InConvBatch(x)

        x2 = self.Contract1(x1)
        x3 = self.Contract2(x2)
        x4 = self.Contract3(x3)
        x5 = self.Contract4(x4)
        x = self.Contract5(x5)
        #x = self.Contract6(x6)
        
        #x = self.Expand6(x, x6)
        x = self.Expand5(x, x5)
        x = self.Expand4(x, x4)
        x = self.Expand3(x, x3)
        x = self.Expand2(x, x2)
        x = self.Expand1(x, x1)

        x = self.OutConv(x)
        return x
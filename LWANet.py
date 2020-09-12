import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import models


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def conv_up(inp, oup):
    return nn.Sequential(
        conv_dw(inp, oup, 1),
        nn.ConvTranspose2d(oup, oup, kernel_size=4,
                           stride=2, padding=1, output_padding=0),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class LWANet(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super(LWANet, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.layer1=mobilenet.features[0]
        self.layer2=mobilenet.features[1]
        self.layer3 = nn.Sequential(
            mobilenet.features[2],
            mobilenet.features[3],)
        self.layer4 = nn.Sequential(
            mobilenet.features[4],
            mobilenet.features[5],
            mobilenet.features[6],)
        self.layer5 = nn.Sequential(
            mobilenet.features[7],
            mobilenet.features[8],
            mobilenet.features[9],
            mobilenet.features[10],)
        self.layer6 = nn.Sequential(
            mobilenet.features[11],
            mobilenet.features[12],
            mobilenet.features[13],)
        self.layer7 = nn.Sequential(
            mobilenet.features[14],
            mobilenet.features[15],
            mobilenet.features[16], )
        self.layer8 = nn.Sequential(
            mobilenet.features[17],
            )

        self.up3 = conv_up(320, 96)
        self.up2 = conv_up(96, 32)
        self.up1 = conv_up(32, 24)

        self.afb3 = AFB(96, 96)
        self.afb2 = AFB(32, 32)
        self.afb1 = AFB(24, 24)
        self.final=nn.Sequential(
            conv_dw(24, 24, 1),
            conv_dw(24, num_classes, 1),)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)  # x / 2
        x1 = self.layer3(x)  # 24, x / 4
        x2 = self.layer4(x1)  # 32, x / 8
        l5 = self.layer5(x2)  # 64, x / 16
        x3 = self.layer6(l5)  # 96, x / 16
        l7 = self.layer7(x3)  # 160, x / 32
        x4 = self.layer8(l7)  # 320, x / 32

        f3 = self.afb3(self.up3(x4), x3)
        f2 = self.afb2(self.up2(f3), x2)
        f1 = self.afb1(self.up1(f2), x1)

        x= self.final(f1)
        x_out = F.log_softmax(x, dim=1)
        return x_out


class AFB(nn.Module):
    def __init__(self, mid_ch,out_ch):
        super(AFB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid())



    def forward(self, input_high, input_low):
        mid_high=self.global_pooling(input_high)
        weight_high=self.conv1(mid_high)
        mid_low = self.global_pooling(input_low)
        weight_low = self.conv2(mid_low)

        return input_high.mul(weight_high)+input_low.mul(weight_low)


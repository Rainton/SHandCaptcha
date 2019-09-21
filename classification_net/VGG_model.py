# encoding:utf-8
import torch
from torch.autograd import Variable
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import time


class ConvBlock(nn.Module):
    def __init__(self, bottom, nout, kernel_size=3, stride=1, padding=1, bias=True, groups=1, has_relu=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(bottom, nout, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(nout)
        self.has_relu = has_relu

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.has_relu:
            return F.relu(x)
        else:
            return x


class DwConvBlock(nn.Module):
    def __init__(self, bottom, nout, kernel_size=3, stride=1, padding=1, bias=True, groups=1):
        super(DwConvBlock, self).__init__()
        self.conv = nn.Conv2d(bottom, nout, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(nout)
        self.conv1x1 = nn.Conv2d(nout, nout, kernel_size=1, stride=1,
                                 padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(nout)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = F.relu(self.bn2(self.conv1x1(x)))
        return x


class Mobilev2Block(nn.Module):
    def __init__(self, bottom, nout, nexpansion, dw_stride=1, has_sum=False):
        super(Mobilev2Block, self).__init__()
        self.conv = ConvBlock(bottom, nexpansion, kernel_size=1, stride=1, padding=0)
        self.dwconv = DwConvBlock(nexpansion, nexpansion, kernel_size=3, stride=dw_stride, groups=nexpansion)
        self.linear = ConvBlock(nexpansion, nout, kernel_size=1, stride=1, padding=0, has_relu=False)
        self.has_sum = has_sum

    def forward(self, x):
        y = self.linear(self.dwconv(self.conv(x)))
        if self.has_sum:
            return x + y
        else:
            return y


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.__init_weight()
        scale = 1
        self.bn1 = nn.BatchNorm2d(3)
        numout = 20 * scale
        self.conv1_1 = ConvBlock(3, numout)
        self.conv1_2 = ConvBlock(numout, numout)
        numout = 32 * scale
        self.conv2_1 = ConvBlock(20 * scale, numout)
        self.conv2_2 = ConvBlock(numout, numout)
        numout = 48 * scale
        self.conv3_1 = ConvBlock(32 * scale, numout)
        self.conv3_2 = ConvBlock(numout, numout)
        self.conv3_3 = ConvBlock(numout, numout)
        numout = 60 * scale
        self.conv4_1 = ConvBlock(48 * scale, numout)
        self.conv4_2 = ConvBlock(numout, numout)
        self.conv4_3 = ConvBlock(numout, numout)
        out_ch = 15
        # self.fc = nn.Linear(numout,out_ch)
        self.reduce1 = nn.Conv2d(numout, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        x = x / 255.0 - 0.5
        x = self.bn1(x)
        x_conv11 = self.conv1_1(x)
        x_conv12 = self.conv1_2(x_conv11)
        x_conv21 = self.conv2_1(self.max_pool(x_conv12))
        x_conv22 = self.conv2_2(x_conv21)
        x_conv31 = self.conv3_1(self.max_pool(x_conv22))
        x_conv32 = self.conv3_2(x_conv31)
        x_conv33 = self.conv3_3(x_conv32)
        x_conv41 = self.conv4_1(self.max_pool(x_conv33))
        x_conv42 = self.conv4_2(x_conv41)
        x_conv43 = self.conv4_3(x_conv42)
        x_reduce = self.reduce1(self.max_pool(x_conv43))
        x_out = torch.squeeze(F.max_pool2d(x_reduce, kernel_size=[4, 4]))
        # x_pro = F.sigmoid(x_out)

        return x_out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = VGG()
    image = Variable(torch.FloatTensor(np.zeros((1, 3, 64, 64))))
    start_time = time.time()
    for i in range(100):
        x_out = model(image)
        print(x_out.shape)
    print(time.time() - start_time)
    # print(model.state_dict().keys())
    print(x_out.size())

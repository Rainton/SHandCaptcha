import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple
import functools
import torch.nn.functional as F
import numpy as np
import time
import math
import classification_net.pytorchparser

Conv = namedtuple('Conv', ['stride', 'depth'])
Conv_crelu = namedtuple('Conv_crelu', ['stride', 'depth', 'kernel'])

DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't'])  # t is the expension factor
DilatedResidual = namedtuple('DilatedResidual', ['stride', 'depth', 'num', 't'])

Pool2d = namedtuple('Pool2d', ['kernel_size', 'stride'])
V1_CONV_DEFS = [
    Conv(stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]
V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=1, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

# 3-29
V2_CONV_DEFS = [
    # 1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=32, num=3, t=4),
    InvertedResidual(stride=2, depth=40, num=3, t=4),
    InvertedResidual(stride=1, depth=52, num=1, t=4),
]
# 3-30
V2_CONV_DEFS = [
    # 1InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't'])
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=30, num=5, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
]
# 4-2    def forward(self, x):

V2_CONV_DEFS = [
    # 1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=42, num=1, t=4),
]
# 4-3Conv
V2_CONV_DEFS = [
    # 1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=28, num=3, t=4),
    InvertedResidual(stride=2, depth=32, num=3, t=4),
    InvertedResidual(stride=1, depth=36, num=1, t=4),
]

# 4-7
V2_CONV_DEFS = [
    # 1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=2, depth=24, num=5, t=4),
    InvertedResidual(stride=1, depth=28, num=3, t=4),
    InvertedResidual(stride=2, depth=32, num=3, t=4),
    InvertedResidual(stride=1, depth=36, num=1, t=4),
]

# 4-7-2
V2_CONV_DEFS = [

    # 1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=5, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]

'''

#4-8 quick stride=2-fail
V2_CONV_DEFS = [
    #1Conv
    Conv(stride=2, depth=12),
    InvertedResidual(stride=2, depth=12, num=1, t=1),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=5, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
#4-8-2
V2_CONV_DEFS = [
    #1
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=16, num=1, t=1),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=5, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
'''
# 4-11-3
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=5, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
# 4-12-1
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]
# 4-12-2-2
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]
# 4-13-1-2
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=20, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]
# 4-14-1-2
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=40, num=1, t=2),
]
# 4-16-1-2
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=2),
]

# 4-16-2-2
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=3, t=2),

]
# 4-17-1
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
    InvertedResidual(stride=2, depth=28, num=3, t=4),
    InvertedResidual(stride=1, depth=32, num=1, t=4),
]
# 4-18-2
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=14, num=2, t=4),
    InvertedResidual(stride=1, depth=18, num=4, t=4),
    InvertedResidual(stride=2, depth=22, num=4, t=4),
    InvertedResidual(stride=1, depth=26, num=4, t=4),
]
# 4-19-1
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    InvertedResidual(stride=1, depth=16, num=4, t=4),
    InvertedResidual(stride=2, depth=20, num=4, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
]
# 5-4-1
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    DilatedResidual(stride=1, depth=10, num=1, t=4),
    DilatedResidual(stride=2, depth=12, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=4, t=4),
    DilatedResidual(stride=2, depth=20, num=3, t=4),
    DilatedResidual(stride=1, depth=24, num=1, t=4),
]
# 5-7-1
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    DilatedResidual(stride=1, depth=10, num=1, t=4),
    DilatedResidual(stride=2, depth=12, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=4, t=4),
    DilatedResidual(stride=2, depth=20, num=4, t=4),
    DilatedResidual(stride=1, depth=24, num=4, t=4),
]

# 5-8-1
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    InvertedResidual(stride=1, depth=16, num=3, t=4),
    DilatedResidual(stride=1, depth=16, num=1, t=4),
    InvertedResidual(stride=2, depth=20, num=4, t=4),
    InvertedResidual(stride=1, depth=24, num=3, t=4),
    DilatedResidual(stride=1, depth=24, num=1, t=4),
]

# 5-10-1
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),
    InvertedResidual(stride=2, depth=12, num=2, t=4),
    InvertedResidual(stride=1, depth=16, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=1, t=4),
    DilatedResidual(stride=1, depth=24, num=1, t=4),
]
# 5-14-1

V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=1, depth=16),
    InvertedResidual(stride=1, depth=10, num=1, t=4),

    InvertedResidual(stride=2, depth=12, num=2, t=4),

    InvertedResidual(stride=1, depth=16, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=2, t=4),

    InvertedResidual(stride=1, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),
]

# 6-15-2
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=10, num=2, t=4),

    InvertedResidual(stride=2, depth=12, num=2, t=4),

    InvertedResidual(stride=2, depth=16, num=2, t=4),
    DilatedResidual(stride=1, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=2, t=4),

    InvertedResidual(stride=1, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),
]

# 6-16-3
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=12, num=2, t=4),

    InvertedResidual(stride=2, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=2, t=4),

    InvertedResidual(stride=2, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),

    InvertedResidual(stride=1, depth=32, num=2, t=4),
    DilatedResidual(stride=1, depth=32, num=2, t=4),
]

# 6-18-3
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=12, num=2, t=4),

    InvertedResidual(stride=2, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),

    InvertedResidual(stride=2, depth=36, num=2, t=4),
    DilatedResidual(stride=1, depth=36, num=2, t=4),

    InvertedResidual(stride=1, depth=48, num=2, t=4),
    DilatedResidual(stride=1, depth=48, num=2, t=4),
]

V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=1, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

# 6-18-3
V2_CONV_DEFS = [
    # 1
    # nn.MaxPool2d(2)
    Conv(stride=2, depth=16),
    InvertedResidual(stride=2, depth=12, num=2, t=4),

    InvertedResidual(stride=2, depth=16, num=2, t=4),

    InvertedResidual(stride=2, depth=24, num=2, t=4),
    DilatedResidual(stride=1, depth=24, num=2, t=4),

    InvertedResidual(stride=2, depth=36, num=2, t=4),
    DilatedResidual(stride=1, depth=36, num=2, t=4),

    InvertedResidual(stride=1, depth=48, num=2, t=4),
    DilatedResidual(stride=1, depth=48, num=2, t=4),
]
# 6-19-3
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    DilatedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    DilatedResidual(stride=1, depth=24, num=1, t=4),
    InvertedResidual(stride=1, depth=30, num=2, t=4),
    DilatedResidual(stride=1, depth=30, num=1, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    DilatedResidual(stride=1, depth=48, num=1, t=4),
]

# 6-21
V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

# 6-21
V2_CONV_DEFS = [
    Conv(stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
]

# 6-20
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=1, t=4),
    InvertedResidual(stride=1, depth=30, num=2, t=4),
    InvertedResidual(stride=1, depth=30, num=1, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=48, num=1, t=4),
]
# 6-24-0
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=2, t=4),
    InvertedResidual(stride=1, depth=20, num=1, t=4),
    InvertedResidual(stride=2, depth=24, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=1, t=4),
    InvertedResidual(stride=1, depth=30, num=2, t=4),
    InvertedResidual(stride=1, depth=30, num=1, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=48, num=1, t=4),
]
# 6-24-0-2
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=1, depth=24, num=4, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=48, num=1, t=4),
]

# 6-28-0
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=4, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    InvertedResidual(stride=1, depth=48, num=1, t=4),
]

# 6-29-0
V2_CONV_DEFS = [
    Conv(stride=4, depth=16),
    InvertedResidual(stride=2, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=1, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=4, t=4),
    InvertedResidual(stride=1, depth=32, num=3, t=4),
    InvertedResidual(stride=2, depth=48, num=3, t=4),
]

# 6-27-0
V2_CONV_DEFS = [
    Conv(stride=2, depth=16),
    InvertedResidual(stride=1, depth=12, num=1, t=4),
    InvertedResidual(stride=2, depth=16, num=2, t=4),
    InvertedResidual(stride=2, depth=20, num=3, t=4),
    InvertedResidual(stride=2, depth=24, num=4, t=4),
    InvertedResidual(stride=1, depth=30, num=3, t=4),
    InvertedResidual(stride=2, depth=36, num=3, t=4),
    DilatedResidual(stride=1, depth=48, num=1, t=4),
]

# 7-2-0
V2_CONV_DEFS = [
    Conv_crelu(stride=1, depth=10, kernel=3),
    Conv_crelu(stride=1, depth=10, kernel=3),
    # Conv_crelu(stride=2, depth=12, kernel=5),
    InvertedResidual(stride=2, depth=16*2, num=1, t=1),
    InvertedResidual(stride=1, depth=16*2, num=1, t=1),
    InvertedResidual(stride=2, depth=24*2, num=1, t=1),
    InvertedResidual(stride=1, depth=24*2, num=1, t=1),
    InvertedResidual(stride=1, depth=24*2, num=1, t=1),
    InvertedResidual(stride=1, depth=24*2, num=1, t=1),
    InvertedResidual(stride=2, depth=30*2, num=1, t=1),
    InvertedResidual(stride=1, depth=30*2, num=1, t=1),
    InvertedResidual(stride=1, depth=30*2, num=1, t=1),
    InvertedResidual(stride=1, depth=30*2, num=1, t=1),
]
# 7-30
V2_CONV_DEFS = [
    Conv(stride=1, depth=20),
    Conv(stride=1, depth=20),
    Conv(stride=2, depth=32),
    Conv(stride=1, depth=32),
    Conv(stride=2, depth=40),
    Conv(stride=1, depth=40),
    Conv(stride=1, depth=40),
    Conv(stride=2, depth=60),
    Conv(stride=1, depth=60),
    Conv(stride=1, depth=60),
]
# 7-30-1
# V2_CONV_DEFS = [
#     Conv(stride=1, depth=20),
#     Conv(stride=1, depth=20),
#     Conv(stride=1, depth=20),
#     Conv(stride=2, depth=32),
#     Conv(stride=1, depth=32),
#     Conv(stride=1, depth=32),
#     Conv(stride=2, depth=40),
#     Conv(stride=1, depth=40),
#     Conv(stride=1, depth=40),
#     Conv(stride=1, depth=40),
#     Conv(stride=2, depth=60),
#     Conv(stride=1, depth=60),
#     Conv(stride=1, depth=60),
#     Conv(stride=1, depth=60),
# ]
# # 7-30-2
# V2_CONV_DEFS = [
#     Conv(stride=1, depth=20*2),
#     Conv(stride=1, depth=20*2),
#     Conv(stride=2, depth=32*2),
#     Conv(stride=1, depth=32*2),
#     Conv(stride=2, depth=40*2),
#     Conv(stride=1, depth=40*2),
#     Conv(stride=1, depth=40*2),
#     Conv(stride=2, depth=60*2),
#     Conv(stride=1, depth=60*2),
#     Conv(stride=1, depth=60*2),
# ]
class _conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _conv_bn_crelu(nn.Module):
    def __init__(self, inp, oup, stride, kernel):
        super(_conv_bn_crelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x1 = F.relu(x)
        x2 = F.relu(-x)
        return torch.cat((x1, x2), 1)


class _pool2d(nn.Module):
    def __init__(self):
        super(_pool2d, self).__init__()
        self.pool2d = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.pool2d(x)


class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            # pw
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _inverted_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _dilated_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_dilated_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, groups=inp * expand_ratio, padding=2,
                      dilation=2, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.depth = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def mobilenet(conv_defs, depth_multiplier=1.0, min_depth=8):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    layers = []
    in_channels = 3
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, Conv_crelu):
            layers += [_conv_bn_crelu(in_channels, depth(conv_def.depth), conv_def.stride, conv_def.kernel)]
            in_channels = depth(conv_def.depth * 2)
        elif isinstance(conv_def, InvertedResidual):
            for n in range(conv_def.num):
                stride = conv_def.stride if n == 0 else 1
                layers += [_inverted_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
                in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DilatedResidual):
            for n in range(conv_def.num):
                stride = conv_def.stride if n == 0 else 1
                layers += [_dilated_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
                in_channels = depth(conv_def.depth)
    return layers


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


class Small_Net(nn.Module):
    def __init__(self, pretrained=False, num_classes = 15):
        super(Small_Net, self).__init__()
        self.ops = nn.ModuleList(mobilenet_v2())
        self.class_max_pool = nn.MaxPool2d(8)
        self.class_conv = nn.Conv2d(60, num_classes, 1)
        self.__init_weight()

    def forward(self, x):
        start_time = time.time()
        for op in self.ops:
            x = op(x)
            # print("0", time.time() - start_time, x.size())
            # start_time = time.time()
            # print(x.size())
        x = self.class_max_pool(x)
        # print("1", time.time() - start_time, x.size())
        start_time = time.time()
        x = self.class_conv(x)
        # classification_net.pytorchparser.pytorch2ncnn(x, "test.param", "test.bin")
        # print("2", time.time() - start_time, x.size())
        start_time = time.time()
        x = x.squeeze_()
        # print("3", time.time() - start_time, x.size())
        start_time = time.time()
        return x


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


mobilenet_v1 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v1_075 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet, conv_defs=V1_CONV_DEFS, depth_multiplier=0.25)

mobilenet_v2 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=1.0)
mobilenet_v2_075 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.75)
mobilenet_v2_050 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.50)
mobilenet_v2_025 = wrapped_partial(mobilenet, conv_defs=V2_CONV_DEFS, depth_multiplier=0.25)


if __name__ == '__main__':
    images = Variable(torch.from_numpy(np.full((1, 3, 64, 64), 0.)).float()).cuda()
    net = Small_Net().cuda()
    net.eval()
    net.load_state_dict(torch.load("E:/code/hands_small_net/classification_net/logs/model/SMALL_NETacc0.928.pkl"))
    # net = net.cuda()
    start_time = time.time()
    out = net(images)
    print(time.time() - start_time)
    print(out.size())
    # print(images.size())
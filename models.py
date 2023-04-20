#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################
# filename: models.py
# datetime: 2023-04-07 14:07:05
######################


import torch
from torch import nn


class ConvReLU(nn.Module):
    '''
    Convolution and ReLU
    '''
    def __init__(self, inch, outch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ILN(nn.Module):
    '''
    Iris Localization Network (ILN)
    input: torch tensor (b,1,480,640)
    output: torch tensor (b,6), 6: x_pupil, y_pupil, r_pupil, x_iris, y_iris, r_iris
    '''
    def __init__(self, num_classes=6, m=0.25, s=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((int(480*s), int(640*s))),

            ConvReLU(1, int(64*m)),
            ConvReLU(int(64*m), int(64*m)),
            nn.MaxPool2d(kernel_size=2, stride=2), # (480sx640s)->(240sx320s)

            ConvReLU(int(64*m), int(128*m)),
            ConvReLU(int(128*m), int(128*m)),
            nn.MaxPool2d(kernel_size=2, stride=2), # (240sx320s)->(120sx160s)

            ConvReLU(int(128*m), int(256*m)),
            ConvReLU(int(256*m), int(256*m)),
            ConvReLU(int(256*m), int(256*m)),
            nn.MaxPool2d(kernel_size=2, stride=2), # (120sx160s)->(60sx80s)

            ConvReLU(int(256*m), int(512*m)),
            ConvReLU(int(512*m), int(512*m)),
            ConvReLU(int(512*m), int(512*m)),
            nn.MaxPool2d(kernel_size=2, stride=2), # (60sx80s)->(30sx40s)

            ConvReLU(int(512*m), int(512*m)),
            ConvReLU(int(512*m), int(512*m)),
            ConvReLU(int(512*m), int(512*m)),
            nn.AdaptiveAvgPool2d((3, 4)),

            nn.Flatten(),
            nn.Linear(int(512*m) * 3 * 4, int(4096*m)),
            nn.ReLU(True),
            nn.Linear(int(4096*m), int(4096*m)),
            nn.ReLU(True),
            nn.Linear(int(4096*m), num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    pass


if __name__ == '__main__':
    main()

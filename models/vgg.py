import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # 750(input)x20 -> 375x128 -> 187x256 -> 93x512 -> 46x512 -> 23x512
        self.architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(self.architecture)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1*512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1)           # 10개의 클래스가 아니라 3개의 클래스로 reduce했기 때문에.
        )
        
    def forward(self, x):
       # input이 3x1x750x40인데, 1d 로 처리할때 shape는 B x C x L이 되므로, 3x40x750으로 바꿔줘야
        x = x.squeeze(1).permute(0,2,1)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, 512)
        out = self.classifier(x)

        return out.view(-1)

    def _make_layers(self, cfg):
        layers = []
        in_channel = 20
        for num_channel in cfg:
            if num_channel == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)] # list 덧셈으로 layer 추가
            else:
                layers += [nn.Conv1d(in_channels= in_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1),
                              nn.BatchNorm1d(num_channel),
                              nn.ReLU(inplace=True)] # cfg 이용하여여 channel 크기에 맞는 convolution layer 추가
                in_channel = num_channel
        
        return nn.Sequential(*layers) # unpacking list using *

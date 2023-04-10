import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import random
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()

        # self.output_size = output_size
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        self.mean_only = mean_only

        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):

        batch_size = inputs.size(0)
        weights = torch.bmm(inputs, self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1))

        if inputs.size(0) == 1:
            attentions = F.softmax(torch.tanh(weights), dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()), dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        if self.mean_only:
            return weighted.sum(1)
        else:
            noise = 1e-5 * torch.randn(weighted.size())

            if inputs.is_cuda:
                noise = noise.to(inputs.device)
            avg_repr, std_repr = weighted.sum(1), (weighted + noise).std(1)

            representations = torch.cat((avg_repr, std_repr), 1)

            return representations


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


RESNET_CONFIGS = {
                  'recon': [[1, 1, 1, 1], PreActBlock],
                  '18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }


def setup_seed(random_seed, cudnn_deterministic=True):
    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False


class ResNet(nn.Module):
    def __init__(self, num_nodes, enc_dim, resnet_type='18', nclasses=2, dropout1d=False, dropout2d=False, p=0.01):
        self.in_planes = 16
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.if_dropout1d = dropout1d
        self.if_dropout2d = dropout2d
        if self.if_dropout2d:
            self.dropout2d = nn.Dropout2d(p=p, inplace=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = nn.Conv2d(512 * block.expansion, 256, kernel_size=(num_nodes, 3), stride=(1, 1), padding=(0, 1),
                               bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        if self.if_dropout1d:
            self.dropout1d = nn.Dropout(p=p, inplace=True)
        self.fc = nn.Linear(256 * 2, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = SelfAttention(256)
        self.GRU = nn.GRU(94, 94, 5)

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.activation(self.bn5(x))
        stats = self.attention(x.mean(2).squeeze(2).permute(0, 2, 1).contiguous())
        feat = self.fc(stats)
        mu = self.fc_mu(feat)
        return feat, mu


class Reconstruction_autoencoder(nn.Module):
    def __init__(self, enc_dim, resnet_type='18', nclasses=2):
        super(Reconstruction_autoencoder, self).__init__()

        self.fc = nn.Linear(enc_dim, 4 * 10 * 125)
        self.bn1 = nn.BatchNorm2d(4)
        self.activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            PreActBlock(4, 16, 1),
            PreActBlock(16, 64, 1),
            PreActBlock(64, 128, 1)
        )

        self.layer2 = nn.Sequential(
            PreActBlock(128, 64, 1)
        )
        self.layer3 = nn.Sequential(
            PreActBlock(64, 32, 1),
            PreActBlock(32, 16, 1)
        )
        self.layer4 = nn.Sequential(
            PreActBlock(16, 8, 1),
            PreActBlock(8, 4, 1),
            PreActBlock(4, 1, 1),
        )

    def forward(self, z):
        z = self.fc(z).view((z.shape[0], 4, 10, 125))
        z = self.activation(self.bn1(z))
        z = self.layer1(z)
        z = self.layer2(z)
        z = nn.functional.interpolate(z, scale_factor=3, mode="bilinear", align_corners=True)
        z = self.layer3(z)
        z = nn.functional.interpolate(z, scale_factor=2, mode="bilinear", align_corners=True)
        z = self.layer4(z)
        return z


class compress_Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(compress_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out



class compress_block(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride,*args, **kwargs):
        super(compress_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))


    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Conversion_autoencoder(nn.Module):
    def __init__(self, num_nodes, enc_dim, nclasses=2):
        self.in_planes = 16
        super(Conversion_autoencoder, self).__init__()

        layers, block = RESNET_CONFIGS['recon']

        self._norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(9, 3), stride=(3, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            compress_block(16, 16, 1),
            compress_block(16, 32, (2,3)),
        )

        self.layer2 = nn.Sequential(
            compress_block(32, 32, 1),
            compress_block(32, 64, 2),
        )

        self.layer3 = nn.Sequential(
            compress_block(64, 64, 1),
            compress_block(64, 128, 2),
        )

        self.layer4 = nn.Sequential(
            compress_block(128, 256, 1),
            compress_block(256, 128, 1),
        )

        # connect x_1

        self.layer1_i = nn.Sequential(
            nn.ConvTranspose2d(256,256,3,2,1),
            PreActBlock(256, 128, 1),
            PreActBlock(128, 64, 1),
        )

        self.layer2_i = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            PreActBlock(128, 64, 1),
            PreActBlock(64, 32, 1),
        )
        self.layer3_i = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, (2,3), 1,output_padding=(1,2)),
            PreActBlock(64, 32, 1),
            PreActBlock(32, 16, 1)
        )
        self.layer4_i = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (9, 3), (3, 2), 1,output_padding=(2,1)),
            PreActBlock(32, 8, 1),
            PreActBlock(8, 4, 1),
            PreActBlock(4, 1, 1),
        )



    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.in_planes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 1, 64, 1, norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, base_width=64, dilation=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x_1 = self.activation(self.bn1(x))
        x_2 = self.layer1(x_1)
        x_3 = self.layer2(x_2)
        x_4 = self.layer3(x_3)
        x_5 = self.layer4(x_4)
        y_1 = torch.cat([x_5,x_4],dim=1)
        y_2 = self.layer1_i(y_1)
        y_2 = torch.cat([y_2,x_3],dim=1)
        y_3 = self.layer2_i(y_2)
        y_3 = torch.cat([y_3,x_2],dim=1)
        y_4 = self.layer3_i(y_3)
        y_5 = torch.cat([y_4,x_1],dim=1)
        result = self.layer4_i(y_5)
        return result


class Speaker_classifier(nn.Module):
    def __init__(self, enc_dim, nclasses):
        super(Speaker_classifier, self).__init__()
        self.fc_1 = nn.Linear(enc_dim, 128)
        self.bn_1 = nn.BatchNorm1d(128)
        self.fc_2 = nn.Linear(128, 64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.fc_3 = nn.Linear(64, nclasses)
    def forward(self, x):
        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        y = self.fc_3(x)
        return y

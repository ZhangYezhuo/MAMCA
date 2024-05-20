import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1))
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage)
        # shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )
            
    def forward(self, x):
         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),)

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class denosing_unit(nn.Module):

    def __init__(self, block, num_block, in_channel=2, out_channel=16):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_channel, self.out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, self.out_channel, num_block, 1)

    def _make_layer(self, block, out_channel, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(out_channel, out_channel, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv2_x(self.conv1(x))
        return x
    
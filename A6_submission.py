import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Define a basic residual block.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the residual network classifier.
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, num_blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * BasicBlock.expansion, 10)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Params:
    def __init__(self):
        self.use_gpu = 1

        self.train = Params.Training()
        self.val = Params.Validation()
        self.test = Params.Testing()
        self.ckpt = Params.Checkpoint()
        self.optim = Params.Optimization()

    def process(self):
        self.val.vis = self.val.vis.ljust(3, '0')
        self.test.vis = self.test.vis.ljust(2, '0')
    
    class Optimization:
        def __init__(self):
            self.type = 'adam'  # or 'sgd'
            self.lr = 0.001
            self.momentum = 0.9
            self.eps = 1e-8
            self.weight_decay = 5e-4

    class Training:
        def __init__(self):
            self.probs_data = ''
            self.batch_size = 128
            self.n_workers = 4  
            self.n_epochs = 300

    class Validation:
        def __init__(self):
            self.batch_size = 16
            self.n_workers = 1
            self.gap = 1
            self.ratio = 0.2
            self.vis = '0'
            self.tb_samples = 10

    class Testing:
        def __init__(self):
            self.enable = 0
            self.batch_size = 24
            self.n_workers = 1
            self.vis = '0'

    class Checkpoint:
        def __init__(self):
            self.load = 1
            self.path = './checkpoints/model.pt'
            self.save_criteria = [
                'train-acc',
                'train-loss',
                'val-acc',
                'val-loss',
            ]

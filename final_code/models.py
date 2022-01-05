import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

def ResNet34Pre(n_classes):
    model = timm.create_model('resnet34', pretrained=True)
    model.fc.out_features = n_classes
    return model

# The ResNet models. RNet allows for manually specifying layers

def RNet(n_classes, layers):
    return ResNet(BasicBlock, layers, num_classes=n_classes)

def ResNet18(n_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes)

def ResNet34(n_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)

def ResNet50(n_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes)

# Convolution Block for Stacked CNN
class ConvLayer(nn.Module):
    def __init__(self, n_in, n_out, k_size, s, p, max_pool=None):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(n_in, n_out, k_size, s, p)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=max_pool[0], stride=max_pool[1]) if max_pool else None

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        
        if self.pool:
            x = self.pool(x)

        return x

# Stacked CNN Model
class NetConv(nn.Module):
    def __init__(self, n_classes):
        super(NetConv, self).__init__()
        
        # *3, *2, *2/3, *1

        n_start = 64

        self.convL1 = ConvLayer(3, n_start, 11, 4, 2, (3, 2))
        self.convL2 = ConvLayer(n_start, n_start * 3, 5, 1, 2, (3, 2))
        self.convL3 = ConvLayer(n_start * 3, n_start * 3 * 2, 3, 1, 1)
        self.convL4 = ConvLayer(n_start * 3 * 2, int(n_start * 3 * 2 * (2/3)), 3, 1, 1)
        self.convL5 = ConvLayer(int(n_start * 3 * 2 * (2/3)), int(n_start * 3 * 2 * (2/3)), 3, 1, 1, (3, 2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(int(n_start * 3 * 2 * (2/3)) * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes)
        )
        
    def forward(self, x):
        
        x = self.convL1(x)
        x = self.convL2(x)
        x = self.convL3(x)
        x = self.convL4(x)
        x = self.convL5(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
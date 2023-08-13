#仅提供网络结构（主要看结构，代码并不优雅）
import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256),
            nn.ReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1),
            nn.ReLU(True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

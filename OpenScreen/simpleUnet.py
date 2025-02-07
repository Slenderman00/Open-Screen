import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet, self).__init__()

        # Down conv
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.down_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.down_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.down_conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(2, 2)
        self.down_conv6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )

        # Up conv
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up_conv4 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up_conv5 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Downsample
        x1 = self.down_conv1(x)
        x1p = self.pool1(x1)
        x2 = self.down_conv2(x1p)
        x2p = self.pool2(x2)
        x3 = self.down_conv3(x2p)
        x3p = self.pool3(x3)
        x4 = self.down_conv4(x3p)
        x4p = self.pool4(x4)
        x5 = self.down_conv5(x4p)
        x5p = self.pool5(x5)
        x6 = self.down_conv6(x5p)

        # Upsample
        x8 = self.up_conv1(x6)
        x8u = F.interpolate(x8, scale_factor=2, mode='bilinear', align_corners=False)
        x8c = torch.cat([x8u, x5], dim=1)
        x9 = self.up_conv2(x8c)
        x9u = F.interpolate(x9, scale_factor=2, mode='bilinear', align_corners=False)
        x9c = torch.cat([x9u, x4], dim=1)
        x10 = self.up_conv3(x9c)
        x10u = F.interpolate(x10, scale_factor=2, mode='bilinear', align_corners=False)
        x10c = torch.cat([x10u, x3], dim=1)
        x11 = self.up_conv4(x10c)
        x11u = F.interpolate(x11, scale_factor=2, mode='bilinear', align_corners=False)
        x11c = torch.cat([x11u, x2], dim=1)
        x12 = self.up_conv5(x11c)
        x12u = F.interpolate(x12, scale_factor=2, mode='bilinear', align_corners=False)
        x12c = torch.cat([x12u, x1], dim=1)

        x_out = torch.sigmoid(self.final_conv(x12c))
        return x_out

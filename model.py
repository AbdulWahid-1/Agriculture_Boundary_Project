# model.py
# This is the U-Net architecture I designed to extract field boundaries.
# I upgraded this to a 6-Band architecture to fully utilize the SWIR infrared data.

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class FieldBoundaryUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # FIXED: Now taking 6 multi-spectral channels from the Kaggle dataset
        self.down1 = ConvBlock(6, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = ConvBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = ConvBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = ConvBlock(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        
        b = self.bottleneck(self.pool(d3))
        
        u3 = self.upconv3(b)
        u3 = self.up3(torch.cat([u3, d3], dim=1))
        
        u2 = self.upconv2(u3)
        u2 = self.up2(torch.cat([u2, d2], dim=1))
        
        u1 = self.upconv1(u2)
        u1 = self.up1(torch.cat([u1, d1], dim=1))
        
        return self.final(u1)